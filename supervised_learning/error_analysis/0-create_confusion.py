#!/usr/bin/env python3
import numpy as np

def create_confusion_matrix(labels, logits):
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    num_classes = labels.shape[1]
    num_classes = int(num_classes)  # ensure integer type

    confusion = np.zeros((num_classes, num_classes))

    for t, p in zip(true_classes, pred_classes):
        confusion[t][p] += 1

    return confusion
