import numpy as np
import matplotlib.pyplot as plt
import sys


def get_model_summary(model, test_x, test_y, threshold=0.5):
    """
    returns:
        Accuracy, Precision, Recall, F1 and AUC
    """
    if not isinstance(test_x, list):
        print("input should be a list")
        return
    pred = model.predict(test_x)[:, 1]
    pred=np.where(pred > threshold, 1, 0).astype(np.int)
    # compare the pred with test_y
    assert len(pred) == len(test_y), \
        "prediction size is not consistent with test label size"

    pred=pred.astype(np.int)
    test_y=test_y.astype(np.int)

    TP=np.sum((pred == 1) & (test_y == 1))
    TN=np.sum((pred == 0) & (test_y == 0))
    FP=np.sum((pred == 1) & (test_y == 0))
    FN=np.sum((pred == 0) & (test_y == 1))

    acc=np.sum(pred == test_y)/len(pred)
    precision=TP/(TP+FP)
    recall=TP/(TP+FN)
    F1=2.0/(1.0/precision+1.0/recall)
    return {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1': F1}


def get_stats(model,
              g_test_x,
              l_test_x,
              test_y,
              num_points = 30):
    """
    return: acc, precision, recall, f1
    """
    x=np.linspace(0.01, 0.99, num_points)
    acc=[]
    precision=[]
    recall=[]
    f1=[]
    for i, value in enumerate(x[:-1]):
        sys.stdout.write(f'\r {i+1}/{len(x[:-1])}')
        stats=get_model_summary(
            model, [g_test_x, l_test_x], test_y, value
        )
        acc.append(stats['Accuracy'])
        precision.append(stats['Precision'])
        recall.append(stats['Recall'])
        f1.append(stats['F1'])
    return acc, precision, recall, f1


def get_roc(model, test_x, test_y, num_points=101):
    pred = model.predict(test_x)[:,1]
    x, y = [], []
    for threshold in np.linspace(0, 1, num_points)[::-1]:
        pred_class = np.where(pred >= threshold, 1, 0).astype(np.int)
        TP = np.sum((pred_class == 1) & (test_y == 1))
        FN = np.sum((pred_class == 0) & (test_y == 1))
        FP = np.sum((pred_class == 1) & (test_y == 0))
        TN = np.sum((pred_class == 0) & (test_y == 0))
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        x.append(FPR)
        y.append(TPR)
    return x, y


def auc(x, y):
    assert len(x) == len(y)
    res = 0 
    for i in range(len(x)-1):
        res += (x[i+1]-x[i])*(y[i]+y[i-1])
    return res * 0.5 
