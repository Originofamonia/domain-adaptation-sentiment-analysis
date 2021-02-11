# http://proceedings.mlr.press/v37/ganin15.pdf
# https://medium.com/analytics-vidhya/domain-adaptation-for-sentiment-analysis-d1930e6548f4
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.autograd import Function


class ReviewDataset(Dataset):
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        review = self.df.iloc[index]["text"]
        sentiment = self.df.iloc[index]["sentiment"]
        sentiment_dict = {
            "pos": 0,
            "neg": 1,
        }
        label = sentiment_dict[sentiment]
        encoded_input = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config["max_length"],
            padding='max_length',
            return_overflowing_tokens=True,
        )
        if "num_truncated_tokens" in encoded_input and encoded_input["num_truncated_tokens"] > 0:
            # print("Attention! you are cropping tokens")
            pass

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        token_type_ids = encoded_input["token_type_ids"] if "token_type_ids" in encoded_input else None

        data_input = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "token_type_ids": torch.tensor(token_type_ids),
            "label": torch.tensor(label),
        }

        return data_input["input_ids"], data_input["attention_mask"], data_input["token_type_ids"], data_input["label"]

    def __len__(self):
        return self.df.shape[0]


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DomainAdaptationModel(nn.Module):
    def __init__(self, config):
        super(DomainAdaptationModel, self).__init__()

        num_labels = config["num_labels"]
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], num_labels),
            nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(config["hidden_size"], 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            grl_lambda=1.0,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        reversed_pooled_output = GradientReversalFn.apply(pooled_output, grl_lambda)

        sentiment_pred = self.sentiment_classifier(pooled_output)
        domain_pred = self.domain_classifier(reversed_pooled_output)

        return sentiment_pred, domain_pred


def compute_accuracy(logits, labels):
    predicted_labels_dict = {
        0: 0,
        1: 0,
    }

    predicted_label = logits.max(dim=1)[1]

    for pred in predicted_label:
        predicted_labels_dict[pred.item()] += 1
    acc = (predicted_label == labels).float().mean()

    return acc, predicted_labels_dict


def inference(model, training_parameters, config, percentage=5):
    device = config['device']
    # src_domain = config['src_domain']
    tgt_domain = config['tgt_domain']
    with torch.no_grad():
        predicted_labels_dict = {
            0: 0,
            1: 0,
        }

        dev_df = pd.read_csv(f"data/amzn_{tgt_domain}_test.tsv", sep="\t")

        data_size = dev_df.shape[0]
        selected_for_evaluation = int(data_size * percentage / 100)
        dev_df = dev_df.head(selected_for_evaluation)
        dataset = ReviewDataset(dev_df, config)

        dataloader = DataLoader(dataset=dataset, batch_size=training_parameters["batch_size"], shuffle=False,
                                num_workers=2)

        mean_accuracy = 0.0
        total_batches = len(dataloader)

        for input_ids, attention_mask, token_type_ids, labels in dataloader:
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            sentiment_pred, _ = model(**inputs)
            accuracy, predicted_labels = compute_accuracy(sentiment_pred, inputs["labels"])
            mean_accuracy += accuracy
            predicted_labels_dict[0] += predicted_labels[0]
            predicted_labels_dict[1] += predicted_labels[1]
        # print(predicted_labels_dict)
    return mean_accuracy / total_batches


def train_src(training_params, config, src_dataloader):
    lr = training_params["learning_rate"]
    n_epochs = training_params["epochs"]
    device = config['device']

    model = DomainAdaptationModel(config)
    # if torch.cuda.device_count() > 1:
    #     print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)
    loss_fn_sentiment_classifier = torch.nn.NLLLoss()
    '''
    In one training step we will update the model using both the source labeled data and target unlabeled data
    We will run it till the batches last for any of these datasets

    In our case target dataset has more data. Hence, we will leverage the entire source dataset for training

    If we use the same approach in a case where the source dataset has more data then the target dataset then we will
    under-utilize the labeled source dataset. In such a scenario it is better to reload the target dataset when it 
    finishes. This will ensure that we are utilizing the entire source dataset to train our model.
    '''
    num_batches = len(src_dataloader)

    for epoch_idx in range(n_epochs):

        src_iter = iter(src_dataloader)
        pbar = tqdm(range(num_batches))

        for batch_idx in pbar:
            p = float(batch_idx + epoch_idx * num_batches) / (training_params["epochs"] * num_batches)
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
            grl_lambda = torch.tensor(grl_lambda)

            model.train()
            optimizer.zero_grad()

            # Source dataset training update
            input_ids, attention_mask, token_type_ids, labels = next(src_iter)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
                "grl_lambda": grl_lambda,
            }

            for k, v in inputs.items():
                inputs[k] = v.to(device)

            sentiment_pred, domain_pred = model(**inputs)
            loss_s_sentiment = loss_fn_sentiment_classifier(sentiment_pred, inputs["labels"])

            if batch_idx % training_params["print_after_steps"] == 0:
                desc = f'E: {epoch_idx}/{n_epochs} step: {batch_idx}/{num_batches} ' \
                       f'loss_s_sentiment={loss_s_sentiment:.4f}'
                pbar.set_description(desc=desc)

            loss_s_sentiment.backward()
            optimizer.step()

        # Evaluate the model after every epoch
        accuracy = inference(model, training_params, config, percentage=1).item()
        print(f"Accuracy on tgt eval set {config['tgt_domain']} after epoch {epoch_idx} is {accuracy:.4f}")

        # torch.save(model.state_dict(), os.path.join(training_parameters["output_folder"],
        #                                             "epoch_" + str(epoch_idx) + training_parameters["output_file"]))

    return model


def train_tgt(training_params, config, src_dataloader, tgt_dataloader):
    lr = training_params["learning_rate"]
    n_epochs = training_params["epochs"]
    device = config['device']

    model = DomainAdaptationModel(config)
    # if torch.cuda.device_count() > 1:
    #     print('Let\'s use {} GPUs!'.format(torch.cuda.device_count()))
    #     model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)

    loss_fn_sentiment_classifier = torch.nn.NLLLoss()
    loss_fn_domain_classifier = torch.nn.NLLLoss()
    '''
    In one training step we will update the model using both the source labeled data and target unlabeled data
    We will run it till the batches last for any of these datasets

    In our case target dataset has more data. Hence, we will leverage the entire source dataset for training

    If we use the same approach in a case where the source dataset has more data then the target dataset then we will
    under-utilize the labeled source dataset. In such a scenario it is better to reload the target dataset when it 
    finishes. This will ensure that we are utilizing the entire source dataset to train our model.
    '''

    max_batches = min(len(src_dataloader), len(tgt_dataloader))

    for epoch_idx in range(n_epochs):

        src_iter = iter(src_dataloader)
        tgt_iter = iter(tgt_dataloader)

        pbar = tqdm(range(max_batches))
        for batch_idx in pbar:
            p = float(batch_idx + epoch_idx * max_batches) / (training_params["epochs"] * max_batches)
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
            grl_lambda = torch.tensor(grl_lambda)

            model.train()
            optimizer.zero_grad()

            # Source dataset training update
            input_ids, attention_mask, token_type_ids, labels = next(src_iter)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
                "grl_lambda": grl_lambda,
            }

            for k, v in inputs.items():
                inputs[k] = v.to(device)

            sentiment_pred, domain_pred = model(**inputs)
            loss_s_sentiment = loss_fn_sentiment_classifier(sentiment_pred, inputs["labels"])
            y_s_domain = torch.zeros(training_params["batch_size"], dtype=torch.long).to(device)
            loss_s_domain = loss_fn_domain_classifier(domain_pred, y_s_domain)

            # Target dataset training update
            input_ids, attention_mask, token_type_ids, labels = next(tgt_iter)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
                "grl_lambda": grl_lambda,
            }

            for k, v in inputs.items():
                inputs[k] = v.to(device)

            _, domain_pred = model(**inputs)

            # Note that we are not using the sentiment predictions here for updating the weights
            y_t_domain = torch.ones(training_params["batch_size"], dtype=torch.long).to(device)
            # print(domain_pred.size(), y_t_domain.size())
            loss_t_domain = loss_fn_domain_classifier(domain_pred, y_t_domain)

            # Combining the loss
            loss = loss_s_sentiment + loss_s_domain + loss_t_domain
            if batch_idx % training_params["print_after_steps"] == 0:
                desc = f'E: {epoch_idx}/{n_epochs} step: {batch_idx}/{max_batches} l={loss:.4f} ' \
                       f'l_s_s={loss_s_sentiment:.4f} l_s_d={loss_s_domain:.4f} ' \
                       f'l_t_d={loss_t_domain:.4f}'
                pbar.set_description(desc=desc)

            loss.backward()
            optimizer.step()

        # Evaluate the model after every epoch
        accuracy = inference(model, training_params, config, percentage=100).item()
        print(f"Accuracy on tgt eval set {config['tgt_domain']} after epoch {epoch_idx} is {accuracy:.4f}")

        # torch.save(model.state_dict(), os.path.join(training_parameters["output_folder"],
        #                                             "epoch_" + str(epoch_idx) + training_parameters["output_file"]))

    return model


def evaluate_model(model, training_params, config):
    accuracy = inference(model, training_params, config, percentage=100).item()
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_domain = 'books'
    tgt_domain = 'dvd'

    config = {
        'device': device,
        'src_domain': src_domain,
        'tgt_domain': tgt_domain,
        "num_labels": 2,
        "hidden_dropout_prob": 0.2,
        "hidden_size": 768,
        "max_length": 512,
    }

    training_params = {
        "batch_size": 4,
        "epochs": 100,
        "output_folder": "./models/",
        "output_file": "tgt_model.pt",
        "learning_rate": 2e-5,
        "print_after_steps": 10,
        "save_steps": 5000,
    }

    src_df = pd.read_csv(f"data/amzn_{src_domain}_train.tsv", sep='\t')
    source_dataset = ReviewDataset(src_df, config)
    src_train_dataloader = DataLoader(dataset=source_dataset,
                                      batch_size=training_params["batch_size"],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=2)

    tgt_df = pd.read_csv(f"data/amzn_{tgt_domain}_train.tsv", sep="\t")
    target_dataset = ReviewDataset(tgt_df, config)
    tgt_train_dataloader = DataLoader(dataset=target_dataset,
                                      batch_size=training_params["batch_size"],
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=2)

    print(f"train src model on {config['src_domain']}")
    src_model = train_src(training_params, config, src_train_dataloader)
    print(f"train tgt model on {config['tgt_domain']}")
    tgt_model = train_tgt(training_params, config, src_train_dataloader, tgt_train_dataloader)

    print(f"src data: {config['src_domain']}; tgt data: {config['tgt_domain']}")
    src_acc = evaluate_model(src_model, training_params, config)
    print(f"src_model {config['src_domain']}'s accuracy on tgt eval set {config['tgt_domain']} is {src_acc:.4f}")
    tgt_acc = evaluate_model(tgt_model, training_params, config)
    print(f"tgt_model {config['tgt_domain']}'s accuracy on tgt eval set {config['tgt_domain']} is {tgt_acc:.4f}")


if __name__ == '__main__':
    main()
