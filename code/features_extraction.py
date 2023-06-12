import numpy as np
import os
import build_vocabulary as bv
import porter
from sklearn.feature_extraction.text import CountVectorizer


def stem_words(words: list):
    """Apply stemming to all words in the list"""
    stemmed_words = []
    for word in words:
        word = porter.stem(word)
        if (word not in stemmed_words):
            stemmed_words.append(word)
    return stemmed_words


def read_document(filename, voc):
    """Read a document and return its BoW representation."""
    f = open(filename, encoding="utf8")
    text = []
    for line in f:
        line = bv.remove_punctuation(line.lower())
        text.append(line)
    f.close()
    vectorizer = CountVectorizer(vocabulary=voc)
    bow_matrix = vectorizer.fit_transform(text)
    return bow_matrix.toarray()


def read_document_stemming(filename, voc):
    """Read a document and return its BoW representation."""
    f = open(filename, encoding="utf8")
    text = []
    for line in f:
        line = bv.remove_punctuation(line.lower())
        stemmed_line = ''
        for w in line.split():
            w = porter.stem(w)
            stemmed_line += w + ' '
        text.append(stemmed_line)
    f.close()
    vectorizer = CountVectorizer(vocabulary=voc)
    bow_matrix = vectorizer.fit_transform(text)
    return bow_matrix.toarray()


def get_bow_representation(vocabulary, no_clickbait_path='../dataset/non_clickbait_train.txt', clickbait_path='../dataset/clickbait_train.txt', save=False, stemming=False, getClasses=False):
    """Read all documents and return the BoW representation, as well as the kind of each document (positive or negative)."""
    labels = []
    classes = []
    if stemming:
        vocabulary = stem_words(vocabulary)
        documents = np.ndarray((0, len(vocabulary)))
        new_samples = read_document_stemming(clickbait_path, vocabulary)
        documents = np.concatenate([documents, new_samples])
        labels = [1]*len(new_samples)
        classes.append("clickbait")

        new_samples = read_document_stemming(no_clickbait_path, vocabulary)
        documents = np.concatenate([documents, new_samples])
        labels = labels + [0]*len(new_samples)
        classes.append("non_clickbait")

    else:
        documents = np.ndarray((0, len(vocabulary)))
        new_samples = read_document(clickbait_path, vocabulary)
        documents = np.concatenate([documents, new_samples])
        labels = [1]*len(new_samples)
        classes.append("clickbait")

        new_samples = read_document(no_clickbait_path, vocabulary)
        documents = np.concatenate([documents, new_samples])
        labels = labels + [0]*len(new_samples)
        classes.append("non_clickbait")

    # np.stack transforms the list of vectors into a 2D array.
    X = np.stack(documents)
    Y = np.array(labels)
    if save:
        data = np.concatenate([X, Y[:, None]], 1)
        np.savetxt("train.txt.gz", data)
    if getClasses:
        return X, Y, classes

    return X, Y
