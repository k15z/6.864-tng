import torch
import numpy as np
from sklearn import metrics
from data import word2vec, askubuntu_label, askubuntu_corpus

ZEROS = [0.0] * 200
MAX_TIMESTEPS = max([len(corpus["question"]) + len(corpus["body"]) for corpus in askubuntu_corpus.values()])

def vectorize(tokens, embedding=word2vec, pad_to=MAX_TIMESTEPS):
    vectors = []
    for token in tokens:
        vectors.append(embedding[token] if token in embedding else ZEROS)
    while len(vectors) < pad_to:
        vectors.insert(0, ZEROS)
    return vectors

tokens = lambda x: x["question"] + x["body"]
def load_dataset(mode):
    for sample in askubuntu_label[mode]:
        pad_to = max(
            [len(tokens(askubuntu_corpus[sample["qid"]]))] + 
            [len(tokens(askubuntu_corpus[qid])) for qid in sample["pos_qids"]] + 
            [len(tokens(askubuntu_corpus[qid])) for qid in sample["neg_qids"]]
        )
        question = vectorize(tokens(askubuntu_corpus[sample["qid"]]), pad_to=pad_to)
        positives = [vectorize(tokens(askubuntu_corpus[qid]), pad_to=pad_to) for qid in sample["pos_qids"]]
        negatives = [vectorize(tokens(askubuntu_corpus[qid]), pad_to=pad_to) for qid in sample["neg_qids"]]
        yield question, positives, negatives, {}

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
    total = 0.0
    for k in range(len(scores)):
        total += precision_at_k(scores, expected, k + 1) * expected[k]
    return 0.0 if sum(expected) == 0.0 else total/sum(expected)

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
