import json
import meter
import numpy as np
from helpers import *
from tqdm import tqdm
from random import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

class GradReverse(autograd.Function):
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return (-grad_output)

class PoolingCNN(nn.Module):
    def __init__(self, input_size, hidden_size, pooling):
        super(PoolingCNN, self).__init__()
        assert pooling in ["mean", "max"]

        self.pooling = pooling
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)

    def forward(self, question):
        output = F.tanh(self.conv1(question))
        if self.pooling == "max":
            return torch.max(output, 2)[0]
        return torch.mean(output, 2)

def load_cnn_dataset(mode, forever=False):
    done = False
    while not done:
        for question, positives, negatives, _ in load_dataset(mode):
            if len(positives) == 0 or len(negatives) == 0: continue
            question = torch.from_numpy(np.array(question).astype(np.float32).transpose()).unsqueeze_(0)
            positives = torch.from_numpy(np.array(positives).astype(np.float32)).permute(0, 2, 1)
            negatives = torch.from_numpy(np.array(negatives).astype(np.float32)).permute(0, 2, 1)
            yield question, positives, negatives
        if not forever: 
            done = True

def max_margin_loss(positives, negatives):
    loss = 0.0
    for pos in positives:
        for neg in negatives:
            pairwise_loss = neg - pos + 0.2
            if pairwise_loss.data[0] > 0.0:
                loss += pairwise_loss
    return loss / (len(positives) * len(negatives))

hidden_size = 128
encoder = PoolingCNN(200, hidden_size, "mean")

classifier = nn.Linear(hidden_size, 1)
classifier_loss = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()))

android_dev_loader = load_cnn_dataset("android.dev", forever=True)
for epoch in range(10):
    print("Epoch", epoch)
    
    i = 0
    for question, positives, negatives in tqdm(load_cnn_dataset("train")):
        # use the encoder
        question = encoder(autograd.Variable(question, requires_grad=True))
        positives = encoder(autograd.Variable(positives, requires_grad=True))
        negatives = encoder(autograd.Variable(negatives, requires_grad=True))

        # train the classifier
        loss = 0.0
        loss += classifier_loss(GradReverse()(classifier(positives)), autograd.Variable(torch.Tensor(positives.size(0), 1).fill_(1.0))) / positives.size(0)
        loss += classifier_loss(GradReverse()(classifier(negatives)), autograd.Variable(torch.Tensor(negatives.size(0), 1).fill_(1.0))) / negatives.size(0)
        _, poss, negg = next(android_dev_loader)
        poss = encoder(autograd.Variable(poss, requires_grad=True))
        negg = encoder(autograd.Variable(negg, requires_grad=True))
        loss += classifier_loss(GradReverse()(classifier(poss)), autograd.Variable(torch.Tensor(poss.size(0), 1).fill_(0.0))) / poss.size(0)
        loss += classifier_loss(GradReverse()(classifier(negg)), autograd.Variable(torch.Tensor(negg.size(0), 1).fill_(0.0))) / negg.size(0)
        if type(loss) != type(0.0):
            (loss * 0.1).backward(retain_graph=True)
            if random() < 0.001:
                print("classifier loss", loss.data[0])

        # train the encoder
        positives = [cosine_similarity(question[0], positives[i], dim=0) for i in range(positives.size(0))]
        negatives = [cosine_similarity(question[0], negatives[i], dim=0) for i in range(negatives.size(0))]
        loss = max_margin_loss(positives, negatives)
        if type(loss) != type(0.0):
            loss.backward()
            if random() < 0.001:
                print("question retrieval loss", loss.data[0])

        i += 1

        if i % 16 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if i % 2048 == 0:
            print()
            for mode in ["dev", "test", "android.dev", "android.test"]:
                all_scores, all_expected = [], []
                stats = {"loss": [], "mrr": [], "map": [], "p@1": [], "p@5": []}
                for question, positives, negatives in load_cnn_dataset(mode):
                    question = encoder(autograd.Variable(question))
                    positives = encoder(autograd.Variable(positives))
                    negatives = encoder(autograd.Variable(negatives))

                    positives = [cosine_similarity(question[0], positives[i], dim=0) for i in range(positives.size(0))]
                    negatives = [cosine_similarity(question[0], negatives[i], dim=0) for i in range(negatives.size(0))]
                    loss = max_margin_loss(positives, negatives)
                    stats["loss"].append(loss.data[0] if type(loss) != type(0.0) else 0.0)

                    positives, negatives = [x.data[0] for x in positives], [x.data[0] for x in negatives]
                    stats["mrr"].append(reciprocal_rank(positives + negatives, [1.0] * len(positives) + [0.0] * len(negatives)))
                    stats["map"].append(average_precision(positives + negatives, [1.0] * len(positives) + [0.0] * len(negatives)))
                    stats["p@1"].append(precision_at_k(positives + negatives, [1.0] * len(positives) + [0.0] * len(negatives), 1))
                    stats["p@5"].append(precision_at_k(positives + negatives, [1.0] * len(positives) + [0.0] * len(negatives), 5))

                    all_scores.extend(positives + negatives)
                    all_expected.extend([1.0] * len(positives) + [0.0] * len(negatives))
                    # if random() < 0.001: print("pos", positives[0], "neg", negatives[0])
                stats = {k: sum(v) / len(v) for k, v in stats.items()}
                stats["mode"] = mode
                stats["roc"], stats["roc@5"] = meter.auroc(np.array(all_scores), np.array(all_expected))
                print(json.dumps(stats))
