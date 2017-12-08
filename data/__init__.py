import pickle

def load(file):
    with open(file, "rb") as fin:
        return pickle.load(fin)

glovevec = load("data/glovevec.pkl")
word2vec = load("data/word2vec.pkl")
askubuntu_label = load("data/askubuntu_label.pkl")
askubuntu_corpus = load("data/askubuntu_corpus.pkl")
android_label = load("data/android_label.pkl")
android_corpus = load("data/android_corpus.pkl")
