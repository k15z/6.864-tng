import json
import numpy as np
from helpers import *
from tqdm import tqdm
from random import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

use_gpu = torch.cuda.is_available()

class PoolingRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pooling):
        super(PoolingRNN, self).__init__()
        assert pooling in ["mean", "final"]

        self.pooling = pooling
        self.hidden_size = hidden_size
        self.lstm1 = nn.LSTM(input_size, hidden_size)

    def forward(self, question):
        batch_size = question.size(0)
        hidden = (autograd.Variable(torch.randn(batch_size, 1, self.hidden_size)), autograd.Variable(torch.randn((batch_size, 1, self.hidden_size))))
        output, hidden = self.lstm1(question, hidden)

        if self.pooling == "final":
            return output[-1,:,:]
        return torch.mean(output[:,:,:], 2)

def load_cnn_dataset(mode, forever=False):
    dataset = []
    for question, positives, negatives, _ in load_dataset(mode):
        if len(positives) == 0 or len(negatives) == 0: continue
        question = torch.from_numpy(np.array(question).astype(np.float32)).unsqueeze_(1)
        positives = torch.from_numpy(np.array(positives).astype(np.float32)).permute(1, 0, 2)
        negatives = torch.from_numpy(np.array(negatives).astype(np.float32)).permute(1, 0, 2)
        yield question, positives, negatives
        dataset.append((question, positives, negatives))
    while forever:
        yield from dataset

def max_margin_loss(positives, negatives):
    loss = 0.0
    for pos in positives:
        for neg in negatives:
            pairwise_loss = neg - pos + 0.2
            if pairwise_loss.data[0] > 0.0:
                loss += pairwise_loss
    return loss / (len(positives) * len(negatives))

encoder = PoolingRNN(200, 32, "mean")
optimizer = optim.Adam(encoder.parameters())

i = 0
for question, positives, negatives in tqdm(load_cnn_dataset("train")):
    question = encoder(autograd.Variable(question, requires_grad=True))
    positives = encoder(autograd.Variable(positives, requires_grad=True))
    negatives = encoder(autograd.Variable(negatives, requires_grad=True))

    positives = [cosine_similarity(question[0], positives[i], dim=0) for i in range(positives.size(0))]
    negatives = [cosine_similarity(question[0], negatives[i], dim=0) for i in range(negatives.size(0))]
    loss = max_margin_loss(positives, negatives)
    if type(loss) != type(0.0):
        loss.backward()

    i += 1

    if i % 16 == 0:
        optimizer.step()
        optimizer.zero_grad()

    if i % 256 == 0:
        print()
        for mode in ["dev", "test"]:
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

                # if random() < 0.001: print("pos", positives[0], "neg", negatives[0])
            stats = {k: sum(v) / len(v) for k, v in stats.items()}
            stats["mode"] = mode
            print(json.dumps(stats))
