import torch
import random
import numpy as np
from sklearn import metrics
from data import glovevec, word2vec, askubuntu_label, askubuntu_corpus, android_label, android_corpus

MISS, HIT = 0, 0
ZEROS = [0.0] * 300
def vectorize(tokens, pad_to, embedding=glovevec):
    global MISS, HIT
    vectors = []
    for token in tokens:
        token = token.lower()
        vectors.append(embedding[token] if token in embedding else ZEROS)
        if token not in embedding:
            MISS += 1
        else: HIT += 1
    while len(vectors) < pad_to:
        vectors.insert(0, ZEROS)
    return vectors

tokens = lambda x: x["question"]# + x["body"]
def load_dataset(mode):
    # positives = batch_size * nb_words * embedding_dims
    if "android" in mode:
        for sample in android_label[mode.replace("android.", "")]:
            negids = sample["neg_qids"] if len(sample["neg_qids"]) < 20 or "train" not in mode else random.sample(sample["neg_qids"], 20)
            pad_to = max(
                [len(tokens(android_corpus[sample["qid"]]))] + 
                [len(tokens(android_corpus[qid])) for qid in sample["pos_qids"]] + 
                [len(tokens(android_corpus[qid])) for qid in negids]
            )
            question = vectorize(tokens(android_corpus[sample["qid"]]), pad_to=pad_to)
            positives = [vectorize(tokens(android_corpus[qid]), pad_to=pad_to) for qid in sample["pos_qids"]]
            negatives = [vectorize(tokens(android_corpus[qid]), pad_to=pad_to) for qid in negids]
            yield question, positives, negatives, {}
    else:
        for sample in askubuntu_label[mode]:
            negids = sample["neg_qids"] if len(sample["neg_qids"]) < 20 or "train" not in mode else random.sample(sample["neg_qids"], 20)
            pad_to = max(
                [len(tokens(askubuntu_corpus[sample["qid"]]))] + 
                [len(tokens(askubuntu_corpus[qid])) for qid in sample["pos_qids"]] + 
                [len(tokens(askubuntu_corpus[qid])) for qid in negids]
            )
            question = vectorize(tokens(askubuntu_corpus[sample["qid"]]), pad_to=pad_to)
            positives = [vectorize(tokens(askubuntu_corpus[qid]), pad_to=pad_to) for qid in sample["pos_qids"]]
            negatives = [vectorize(tokens(askubuntu_corpus[qid]), pad_to=pad_to) for qid in negids]
            yield question, positives, negatives, {}

def normalize(input, p=2, dim=1, eps=1e-12):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
    """
    return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)
    
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    r"""Returns cosine similarity between x1 and x2, computed along dim.
    .. math ::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}
    Args:
        x1 (Variable): First input.
        x2 (Variable): Second input (of size matching x1).
        dim (int, optional): Dimension of vectors. Default: 1
        eps (float, optional): Small value to avoid division by zero.
            Default: 1e-8
    Shape:
        - Input: :math:`(\ast_1, D, \ast_2)` where D is at position `dim`.
        - Output: :math:`(\ast_1, \ast_2)` where 1 is at position `dim`.
    Example::
        >>> input1 = autograd.Variable(torch.randn(100, 128))
        >>> input2 = autograd.Variable(torch.randn(100, 128))
        >>> output = F.cosine_similarity(input1, input2)
        >>> print(output)
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def reorder(scores, expected):
    """
    Produce a list of (score, expected) where we first sort by score in descending order and
    then sort by expected in ascending order. The reason for sorting expected by ascending is
    to assume the worst-case scenario - namely that if we predict the same score for multiple
    samples, we don't have a well-defined order so we should assume the worst ordering.

    Examples:
        reorder([0, 0], [0, 1]) -> [(0,0), (0,1)]
        reorder([0, 1], [0, 1]) -> [(1,1), (0,0)]
        reorder([0, 1, 1], [0, 1, 0]) -> [(1,1), (1,0), (0,0)]
    """
    return list(reversed(sorted(zip(scores, expected), key=lambda x: (x[0], -x[1]))))

def average_precision(scores, expected):
    """
    See https://en.wikipedia.org/wiki/Information_retrieval#Average_precision

    scores - a list of rank scores
    expected - a list of 0 or 1
    """
    return metrics.average_precision_score(expected, scores)

def precision_at_k(scores, expected, k):
    """
    scores - a list of rank scores
    expected - a list of 0 or 1
    k - an integer
    """
    first_k = reorder(scores, expected)[0:k]
    targets = [t for s, t in first_k]
    return sum(targets) / len(targets)

def reciprocal_rank(scores, expected):
    """
    See https://en.wikipedia.org/wiki/Mean_reciprocal_rank

    scores - a list of rank scores
    expected - a list of 0 or 1
    """
    for i, pair in enumerate(reorder(scores, expected)):
        score, target = pair
        if target:
            return 1.0 / (i + 1.0)
    return 0.0

def auroc_at_fpr(scores, expected, max_fpr=0.05):
    fpr, tpr, thresholds = metrics.roc_curve(expected, scores)
    for i in range(fpr.shape[0]):
        if fpr[i] > max_fpr:
            break
    return metrics.auc(fpr[:i], tpr[:i])

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
    return loss / len(positives) if len(positives) > 0 else loss
