from keybert import KeyBERT
import re
import pandas as pd


kw_model = KeyBERT()


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def concatentate_text(text):
    text = text.replace(" ", "_")
    return text


def get_bag_of_words(authors, publisher):
    for i, author in enumerate(authors):
        authors[i] = preprocess_text(author)
        authors[i] = concatentate_text(authors[i])
    authors = " ".join(authors)
    publisher = preprocess_text(publisher)
    publisher = concatentate_text(publisher)
    bag_of_words = " ".join([authors, publisher])
    return bag_of_words


def get_keywords(text):
    keywords = kw_model.extract_keywords(
        text, keyphrase_ngram_range=(1, 1), stop_words="english"
    )
    keywords = " ".join([k[0] for k in keywords])
    return keywords


def get_keywords(new_books):
    for book in new_books:
        bag_of_words = get_bag_of_words(
            [author["author"]["name"] for author in book["authors"]], book["publisher"]
        )
        keywords = get_keywords(book["preprocessed_description"])
        book["keywords"] = bag_of_words + " " + keywords
