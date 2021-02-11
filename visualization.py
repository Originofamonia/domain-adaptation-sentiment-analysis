
import matplotlib.pyplot as plt
from transformers import pipeline
import numpy as np
import pprint
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns


def main():
    nlp_features = pipeline('feature-extraction')
    amazon_df = pd.read_csv("data/amzn_books_test.tsv", sep="\t")  # .head(1000)

    amazon_features = []

    # total_count = 1000
    # print_step = 100
    # count = 0
    for index, row in amazon_df.iterrows():
        review = row["text"]
        inputs = nlp_features.tokenizer.cls_token + review
        # count += 1
        # if count > total_count:
        #     break
        # if count % print_step == 0:
        #     print("Processed count:", count)
        try:
            output = nlp_features(inputs)
            output_array = np.array(output)
            feature = list(output_array[0][0])
            amazon_features.append(feature)

        except Exception as e:
            pass

    imdb_df = pd.read_csv("data/amzn_dvd_test.tsv", sep="\t")  # .head(1000)

    imdb_features = []
    # count = 0
    for index, row in imdb_df.iterrows():
        review = row["text"]
        inputs = nlp_features.tokenizer.cls_token + review
        # count += 1
        # if count > total_count:
        #     break
        # if count % print_step == 0:
        #     print("Processed count:", count)
        try:
            output = nlp_features(inputs)
            output_array = np.array(output)
            feature = list(output_array[0][0])
            imdb_features.append(feature)

        except Exception as e:
            pass

    print(len(amazon_features), len(imdb_features))

    total_features = amazon_features + imdb_features
    X_embedded = TSNE(n_components=2).fit_transform(total_features)
    print(X_embedded.shape)
    y = ['Domain 0' for i in np.arange(len(amazon_features))] + ['Domain 1' for i in np.arange(len(imdb_features))]
    # y is the domain label, not class label
    sns.set(rc={'figure.figsize': (11.7, 8.27)})
    palette = sns.color_palette("bright", 2)
    sns.scatterplot(X_embedded[:, 0], X_embedded[:, 1], hue=y, legend='full', palette=palette)
    plt.show()


if __name__ == '__main__':
    main()
