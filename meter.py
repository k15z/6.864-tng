
'''
Code took from PyTorchNet (https://github.com/pytorch/tnt)

'''

import math
import numbers
import numpy as np
import torch

class Meter(object):
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class AUCMeter(Meter):
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.

    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """
    def __init__(self):
        super(AUCMeter, self).__init__()
        self.reset()

    def reset(self):
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
        assert np.ndim(output) == 1, \
            'wrong output size (1D expected)'
        assert np.ndim(target) == 1, \
            'wrong target size (1D expected)'
        assert output.shape[0] == target.shape[0], \
            'number of outputs and targets does not match'
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
            'targets should be binary (0, 1)'

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)
        self.sortind = None


    def value(self, max_fpr=1.0):
        assert max_fpr > 0

        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5

        # sorting the arrays
        if self.sortind is None:
            scores, sortind = torch.sort(torch.from_numpy(self.scores), dim=0, descending=True)
            scores = scores.numpy()
            self.sortind = sortind.numpy()
        else:
            scores, sortind = self.scores, self.sortind

        # creating the roc curve
        tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            if self.targets[sortind[i - 1]] == 1:
                tpr[i] = tpr[i - 1] + 1
                fpr[i] = fpr[i - 1]
            else:
                tpr[i] = tpr[i - 1]
                fpr[i] = fpr[i - 1] + 1

        tpr /= (self.targets.sum() * 1.0)
        fpr /= ((self.targets - 1.0).sum() * -1.0)

        for n in range(1, scores.size + 1):
            if fpr[n] > max_fpr:
                break

        # calculating area under curve using trapezoidal rule
        #n = tpr.shape[0]
        h = fpr[1:n] - fpr[0:n - 1]
        sum_h = np.zeros(fpr.shape)
        sum_h[0:n - 1] = h
        sum_h[1:n] += h
        area = (sum_h * tpr).sum() / 2.0

        return area / max_fpr

def auroc(scores, expected):
    meter = AUCMeter()
    meter.add(scores, expected)
    return meter.value(1.0), meter.value(0.5)

if __name__ == '__main__':
    # there's something wrong with this but the staff said to use it so ¯\_(ツ)_/¯
    from sklearn.metrics import roc_auc_score

    pos = [0.9799557328224182]
    neg = [0.9866231679916382, 0.9863668084144592, 0.9841463565826416, 0.9835896492004395, 0.982826292514801, 0.9825848340988159, 0.9817935228347778, 0.9811779260635376, 0.9801144599914551, 0.9799917936325073, 0.9796085357666016, 0.9784649014472961, 0.9772071838378906, 0.9771943688392639, 0.9767858982086182, 0.9749810695648193, 0.9738849997520447, 0.9707192182540894, 0.9698690176010132, 0.967475950717926]
    #pos = [0.9922550320625305, 0.9916551113128662, 0.9903582334518433, 0.9878537654876709, 0.9932836294174194, 0.9908018708229065]
    #neg = [0.9894577264785767, 0.9884077906608582, 0.9854131937026978, 0.9849563241004944, 0.9846494793891907, 0.9846468567848206, 0.9846099019050598, 0.98403400182724, 0.9836485981941223, 0.9830759763717651, 0.9828921556472778, 0.9827064871788025, 0.9823583960533142, 0.9822470545768738, 0.9821904897689819, 0.9819778203964233, 0.9818161129951477, 0.9812932014465332, 0.9806071519851685, 0.9804866313934326]

    scores = np.array(pos + neg)
    expected = np.array([1.0] * len(pos) + [0.0] * len(neg))

    # standard sklearn implementation
    print(roc_auc_score(expected, scores))

    # meter implementation
    meter = AUCMeter()
    meter.add(scores, expected)
    print(meter.value(max_fpr=1.0))
    print(meter.value(max_fpr=0.9))
    print(meter.value(max_fpr=0.5))
    print(meter.value(max_fpr=0.1))
    print(meter.value(max_fpr=0.05))
