from random import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

scores = [random()/5 for _ in range(100)] + [random() for _ in range(100)]
expected1 = [0.0] * 100 + [1.0] * 100
expected2 = [0.0] * 150 + [1.0] * 50

fpr, tpr, _ = roc_curve(expected1, scores)
plt.plot(fpr, tpr, label="baseline")

fpr, tpr, _ = roc_curve(expected2, scores)
plt.plot(fpr, tpr, label="high false negative rate")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Examples')
plt.legend(loc="lower right")
plt.show()
