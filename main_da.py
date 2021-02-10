import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
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
            max_length=self.config["max_length"],
            pad_to_max_length=True,
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


def inference(model, training_parameters, config, dataset="imdb", percentage=5):
    device = config['device']
    domain = config['domain']
    with torch.no_grad():
        predicted_labels_dict = {
            0: 0,
            1: 0,
        }
        if dataset == 'amzn':
            dev_df = pd.read_csv(f"data/{dataset}_{domain}_test.tsv", sep="\t")
        else:
            dev_df = pd.read_csv(f"data/{dataset}_dev.tsv", sep="\t")
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
        print(predicted_labels_dict)
    return mean_accuracy / total_batches


def train(training_parameters, config, source_dataloader, target_dataloader):
    lr = training_parameters["learning_rate"]
    n_epochs = training_parameters["epochs"]
    device = config['device']

    model = DomainAdaptationModel(config)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr)

    loss_fn_sentiment_classifier = torch.nn.NLLLoss()
    loss_fn_domain_classifier = torch.nn.NLLLoss()
    '''
    In one training step we will update the model using both the source labeled data and target unlabeled data
    We will run it till the batches last for any of these datasets

    In our case target dataset has more data. Hence, we will leverage the entire source dataset for training

    If we use the same approach in a case where the source dataset has more data then the target dataset then we will
    under-utilize the labeled source dataset. In such a scenario it is better to reload the target dataset when it finishes
    This will ensure that we are utilizing the entire source dataset to train our model.
    '''

    max_batches = min(len(source_dataloader), len(target_dataloader))

    for epoch_idx in range(n_epochs):

        source_iterator = iter(source_dataloader)
        target_iterator = iter(target_dataloader)

        for batch_idx in range(max_batches):

            p = float(batch_idx + epoch_idx * max_batches) / (training_parameters["epochs"] * max_batches)
            grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
            grl_lambda = torch.tensor(grl_lambda)

            model.train()

            if batch_idx % training_parameters["print_after_steps"] == 0:
                print("Training Step:", batch_idx)

            optimizer.zero_grad()

            # Souce dataset training update
            input_ids, attention_mask, token_type_ids, labels = next(source_iterator)
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
            y_s_domain = torch.zeros(training_parameters["batch_size"], dtype=torch.long).to(device)
            loss_s_domain = loss_fn_domain_classifier(domain_pred, y_s_domain)

            # Target dataset training update
            input_ids, attention_mask, token_type_ids, labels = next(target_iterator)
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
            y_t_domain = torch.ones(training_parameters["batch_size"], dtype=torch.long).to(device)
            print(domain_pred.size(), y_t_domain.size())
            loss_t_domain = loss_fn_domain_classifier(domain_pred, y_t_domain)

            # Combining the loss

            loss = loss_s_sentiment + loss_s_domain + loss_t_domain
            loss.backward()
            optimizer.step()

        # Evaluate the model after every epoch

        torch.save(model.state_dict(), os.path.join(training_parameters["output_folder"],
                                                    "epoch_" + str(epoch_idx) + training_parameters["output_file"]))
        accuracy = inference(model, training_parameters, config, dataset="amzn", percentage=1).item()
        print("Accuracy on amazon after epoch " + str(epoch_idx) + " is " + str(accuracy))

        accuracy = inference(model, training_parameters, config, dataset="imdb", percentage=1).item()
        print("Accuracy on imdb after epoch " + str(epoch_idx) + " is " + str(accuracy))

    return model


def evaluate_model(model, training_parameters, config):
    accuracy = inference(model, training_parameters, config, dataset="amzn", percentage=100).item()
    print("Accuracy on full amazon is " + str(accuracy))

    accuracy = inference(model, training_parameters, config, dataset="imdb", percentage=100).item()
    print("Accuracy on full imdb is " + str(accuracy))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = 'dvd'

    config = {
        'device': device,
        'domain': domain,
        "num_labels": 2,
        "hidden_dropout_prob": 0.2,
        "hidden_size": 768,
        "max_length": 512,
    }

    training_params = {
        "batch_size": 2,
        "epochs": 1,
        "output_folder": "./models/",
        "output_file": "model.pt",
        "learning_rate": 2e-5,
        "print_after_steps": 5,
        "save_steps": 5000,
    }

    imdb_df = pd.read_csv("./data/imdb_train.tsv", sep='\t')
    source_dataset = ReviewDataset(imdb_df, config)
    source_dataloader = DataLoader(dataset=source_dataset,
                                   batch_size=training_params["batch_size"],
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=2)

    amazon_df = pd.read_csv(f"./data/amzn_{domain}_train.tsv", sep="\t")
    target_dataset = ReviewDataset(amazon_df, config)
    target_dataloader = DataLoader(dataset=target_dataset,
                                   batch_size=training_params["batch_size"],
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=2)

    model = train(training_params, config, source_dataloader, target_dataloader)
    evaluate_model(model, training_params, config)


if __name__ == '__main__':
    main()
