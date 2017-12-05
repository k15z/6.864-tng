"""
Start by downloading the datasets into *this* directory:

    git clone https://github.com/taolei87/askubuntu
    git clone https://github.com/jiangfeng1124/Android
    wget http://nlp.stanford.edu/data/glove.42B.300d.zip -O glove.zip
    unzip glove.zip; rm glove.zip; gzip glove.42B.300d.txt

Then, run `preprocess.py` to pack the data into Python pickles for easier loading.
"""
import gzip
import pickle
from tqdm import tqdm
from collections import defaultdict

def load(file):
    with open(file, "rb") as fin:
        return pickle.load(fin)

def save(file, data):
    with open(file, "wb") as fout:
        pickle.dump(data, fout, protocol=pickle.HIGHEST_PROTOCOL)

def word_embedding(path):
    embedding = {}
    with gzip.open(path, "rt", encoding="utf8") as fin:
        for row in tqdm(map(lambda line: line.split(), fin), path):
            word, vec = row[0].lower(), list(map(float, row[1:]))
            assert word not in embedding
            embedding[word] = vec
    return embedding

class AskUbuntu(object):

    def load_corpus():
        askubuntu = {}
        with gzip.open("askubuntu/text_tokenized.txt.gz", "rt", encoding="utf8") as fin:
            for qid, question, body in tqdm(map(lambda line: line.split("\t"), fin)):
                body_tokens = body.strip().split(" ")
                askubuntu[int(qid)] = {
                    "question": question.split(" "),
                    "body": body_tokens if len(body_tokens) < 100 else body_tokens[:100]
                }
        return askubuntu

    def load_dataset(mode):
        dataset = []
        if mode == "train": mode = "train_random"
        with open("askubuntu/%s.txt" % mode) as fin:
            for line in fin:
                qid, positive, negative = line.split("\t")[0:3]
                positive = set(map(int, positive.split()))
                negative = set(map(int, negative.split()))
                if len(negative - positive) > 0:
                    dataset.append({
                        "qid": int(qid),
                        "pos_qids": list(positive),
                        "neg_qids": list(negative - positive)
                    })
        return dataset

class Android(object):

    def load_corpus():
        android = {}
        with gzip.open("Android/corpus.tsv.gz", "rt", encoding="utf8") as fin:
            for qid, question, body in tqdm(map(lambda line: line.split("\t"), fin)):
                body_tokens = body.strip().split(" ")
                android[int(qid)] = {
                    "question": question.split(" "),
                    "body": body_tokens if len(body_tokens) < 100 else body_tokens[:100]
                }
        return android

    def load_dataset(neg_file, pos_file):
        dataset = defaultdict(lambda: {"pos_qids": [], "neg_qids": []})
        with open(neg_file, "rt", encoding="utf8") as fin:
            for row in tqdm(map(lambda line: line.split(), fin)):
                qid, negid = int(row[0]), int(row[1])
                dataset[qid]["pos_qids"].append(negid)
        with open(pos_file, "rt", encoding="utf8") as fin:
            for row in tqdm(map(lambda line: line.split(), fin)):
                qid, posid = int(row[0]), int(row[1])
                dataset[qid]["neg_qids"].append(posid)
        return [{**{"qid": k}, **v} for k, v in dataset.items()]

"""
save("word2vec.pkl", word_embedding("askubuntu/vector/vectors_pruned.200.txt.gz"))

save("askubuntu_label.pkl", {
    "dev": AskUbuntu.load_dataset("dev"),
    "test": AskUbuntu.load_dataset("test"),
    "train": AskUbuntu.load_dataset("train_random")
})
save("askubuntu_corpus.pkl", AskUbuntu.load_corpus())

save("android_label.pkl", {
    "dev": Android.load_dataset("Android/dev.neg.txt", "Android/dev.pos.txt"),
    "test": Android.load_dataset("Android/test.neg.txt", "Android/test.pos.txt")
})
save("android_corpus.pkl", Android.load_corpus())
"""
