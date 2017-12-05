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

parser = argparse.ArgumentParser(description='Train the CNN model.')
parser.add_argument('--pooling', type=str, default="mean")
parser.add_argument('--hidden_dims', type=int, default=128)
args = parser.parse_args()

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

def load_cnn_dataset(mode):
    for question, positives, negatives, _ in load_dataset(mode):
        if len(positives) == 0 or len(negatives) == 0: continue
        question = torch.from_numpy(np.array(question).astype(np.float32).transpose()).unsqueeze_(0)
        positives = torch.from_numpy(np.array(positives).astype(np.float32)).permute(0, 2, 1)
        negatives = torch.from_numpy(np.array(negatives).astype(np.float32)).permute(0, 2, 1)
        yield question, positives, negatives

def max_margin_loss(positives, negatives):
    loss = 0.0
    for pos in positives:
        maxNeg = negatives[0]
        for neg in negatives:
            if neg.data[0] > maxNeg.data[0]:
                maxNeg = neg
        pairwise_loss = maxNeg - pos + 1.0
        if pairwise_loss.data[0] > 0.0:
            loss += pairwise_loss
    return loss / (len(positives) * len(negatives))

encoder = PoolingCNN(200, args.hidden_size, args.pooling)
optimizer = optim.Adam(encoder.parameters(), lr=0.0001, weight_decay=1e-5)

for epoch in range(16):
    print("Epoch", epoch)
    
    i = 0
    for question, positives, negatives in tqdm(load_cnn_dataset("train")):
        encoder.train(True)
        question = encoder(autograd.Variable(question, requires_grad=True))
        positives = encoder(autograd.Variable(positives, requires_grad=True))
        negatives = encoder(autograd.Variable(negatives, requires_grad=True))

        positives = [cosine_similarity(question[0], positives[i], dim=0) for i in range(positives.size(0))]
        negatives = [cosine_similarity(question[0], negatives[i], dim=0) for i in range(negatives.size(0))]
        loss = max_margin_loss(positives, negatives)
        if type(loss) != type(0.0):
            loss.backward()
            if random() < 0.001:
                print("question retrieval loss", loss.data[0])

        i += 1

        if i % 32 == 0:
            optimizer.step()
            optimizer.zero_grad()

        if i % 2048 == 0:
            print()
            encoder.train(False)
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
