import pandas as pd


def remove_tabs(x):
    return x.replace("\t", " ").replace("\n", ". ")


def preprocess_imdb(split):
    imdb_folder = "data/imdb/"

    filename = split + ".csv"

    imdb_df = pd.read_csv(imdb_folder + filename)
    imdb_df["text"] = imdb_df["text"].apply(remove_tabs)
    return imdb_df


def func(score):
    if score > 0:
        return "pos"
    else:
        return "neg"


def preprocess_amzn(domain, split):
    # folder = "../amazon-reviews/home_n_kitchen_splits/"
    filename = f'data/processed/{domain}/{split}.txt'

    amazon_df = pd.read_csv(filename, sep="\t")

    amazon_df["label"] = amazon_df["label"].apply(func)

    amazon_df["review"] = amazon_df["review"].apply(remove_tabs)
    amazon_df.columns = ['text', 'sentiment']
    return amazon_df


def main():
    # imdb_train_df = preprocess_imdb("train")
    # imdb_dev_df = preprocess_imdb("test")
    # imdb_train_df.to_csv("./data/imdb_train.tsv", sep="\t", index=False)
    # imdb_dev_df.to_csv("./data/imdb_dev.tsv", sep="\t", index=False)
    # print(imdb_train_df.head())
    # print(imdb_train_df.shape)
    # print(imdb_dev_df.head())
    # print(imdb_dev_df.shape)
    domain = 'kitchen'
    amazon_train_df = preprocess_amzn(domain, "train")
    amazon_dev_df = preprocess_amzn(domain, "test")

    amazon_train_df.to_csv(f"./data/amzn_{domain}_train.tsv", sep="\t", index=False)
    amazon_dev_df.to_csv(f"./data/amzn_{domain}_test.tsv", sep="\t", index=False)
    print(amazon_train_df.head())
    print(amazon_train_df.shape)
    print(amazon_dev_df.head())
    print(amazon_dev_df.shape)


if __name__ == '__main__':
    main()
