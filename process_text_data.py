"""
Script to process and tokenize document classification dataset.
"""

import os
import pickle
import string

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sklearn.datasets
import sklearn.feature_extraction
import textblob


#### SET STOP WORDS AND CLEAN NUMBERS AND SPECIAL CHARACTERS
stop_words = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS.union(
    [
        "'s",
        "th",
        "anywh",
        "becau",
        "el",
        "elsewh",
        "everywh",
        "ind",
        "otherwi",
        "plea",
        "somewh",
        "abov",
        "afterward",
        "alon",
        "alreadi",
        "alway",
        "ani",
        "anoth",
        "anyon",
        "anyth",
        "anywher",
        "becam",
        "becaus",
        "becom",
        "befor",
        "besid",
        "cri",
        "describ",
        "dure",
        "els",
        "elsewher",
        "empti",
        "everi",
        "everyon",
        "everyth",
        "everywher",
        "fifti",
        "formerli",
        "forti",
        "ha",
        "henc",
        "hereaft",
        "herebi",
        "hi",
        "howev",
        "hundr",
        "inde",
        "latterli",
        "mani",
        "meanwhil",
        "moreov",
        "mostli",
        "nobodi",
        "noon",
        "noth",
        "nowher",
        "onc",
        "onli",
        "otherwis",
        "ourselv",
        "perhap",
        "pleas",
        "seriou",
        "sever",
        "sinc",
        "sincer",
        "sixti",
        "someon",
        "someth",
        "sometim",
        "somewher",
        "themselv",
        "thenc",
        "thereaft",
        "therebi",
        "therefor",
        "thi",
        "thu",
        "togeth",
        "twelv",
        "twenti",
        "veri",
        "wa",
        "whatev",
        "whenc",
        "whenev",
        "wherea",
        "whereaft",
        "wherebi",
        "wherev",
        "whi",
        "yourselv",
    ]
    + list(string.digits + string.ascii_lowercase)
)

# get stemming of words
def textblob_tokenizer(str_input):
    blob = textblob.TextBlob(str_input.lower())
    words = [token.stem() for token in blob.words]
    return words


def get_dataset_vocab(dataset_x, max_features=1000, min_df=10):
    # Get TFIDF Representation. The purpose of this is to get the
    # top `max_features` most common non-stop words above `min_df`
    # and turn them into the vocabulary.
    print("Computing vocabulary...")
    tfidf = TfidfVectorizer(
        sublinear_tf=True,
        tokenizer=textblob_tokenizer,
        min_df=min_df,
        max_features=max_features,
        norm="l2",
        encoding="latin-1",
        ngram_range=(1, 1),
        stop_words=stop_words,
        lowercase=True,
        strip_accents="ascii",
        analyzer="word",
    )
    tfidf.fit_transform(dataset_x)
    vocabulary = tfidf.get_feature_names()
    return vocabulary


def bow_process_dataset(dataset, dataset_name, vocabulary):
    """ parameter `dataset` is a dict containing 'x', 'y', 'label_ids' """

    # Convert to bag of words representation using vocabulary.
    print(f"Converting to {dataset_name} BOW...")
    vectorizer = CountVectorizer(
        vocabulary=vocabulary,
        tokenizer=textblob_tokenizer,
        encoding="latin-1",
        ngram_range=(1, 1),
        stop_words=stop_words,
        lowercase=True,
        strip_accents="ascii",
        analyzer="word",
        binary=True,
    )
    dataset["x"] = vectorizer.fit_transform(dataset["x"])

    print("Saving...")
    dataset["vocab"] = vocabulary
    with open(f"data/doc_classification/{dataset_name}.pkl", "wb") as f:
        pickle.dump(dataset, f)


def get_dataset():
    # Load dataset
    df_huffpost = pd.read_json(
        "data/doc_classification/News_Category_Dataset_v2.json", lines=True
    )

    # Remove categories with ambiguous meanings
    categories_to_remove = [
        "IMPACT",
        "LATINO VOICES",
        "EDUCATION",
        "COLLEGE",
        "GREEN",
        "THE WORLDPOST",
        "WORLDPOST",
        "FIFTY",
    ]
    df_huffpost = df_huffpost[
        ~df_huffpost["category"].isin(categories_to_remove)
    ].reset_index()

    # Merge duplicate categories
    def rename(x):
        if x == "PARENTS":
            return "PARENTING"
        if x in ["ARTS", "CULTURE & ARTS"]:
            return "ARTS & CULTURE"
        if x == "STYLE":
            return "STYLE & BEAUTY"
        if x == "TASTE":
            return "FOOD & DRINK"
        if x == "HEALTHY LIVING":
            return "HOME & LIVING"
        return x

    df_huffpost["category"] = df_huffpost["category"].map(rename)

    # Prune to keep only top 10 most frequent categories
    top10_categories = (
        df_huffpost["category"].value_counts(ascending=True).index.tolist()[-10:]
    )
    df_huffpost = df_huffpost[
        df_huffpost["category"].isin(top10_categories)
    ].reset_index()

    df_huffpost["Text"] = (
        df_huffpost["headline"] + " " + df_huffpost["short_description"]
    )
    # For future convenience
    df_huffpost.to_json(
        "data/doc_classification/News_Category_Dataset_Cleaned_Categories10.json",
        orient="records",
        lines=True,
    )

    x = df_huffpost["Text"]
    y, label_ids = df_huffpost["category"].factorize()

    return dict(x=x, y=y, label_ids=list(label_ids))


def main():
    huffpost_10 = get_dataset()
    vocabulary = get_dataset_vocab(huffpost_10["x"])
    bow_process_dataset(huffpost_10, "cleaned_categories10", vocabulary)


if __name__ == "__main__":
    main()