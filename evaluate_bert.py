#!/usr/bin/env python
import sys, os, os.path
import json
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

if __name__ == "__main__":

    # Load student code
    from mp3 import run_train_test

    train_data = json.load(open('train.json'))
    dev_data = json.load(open('dev.json'))
    labels = [_.pop('label') for _ in dev_data]

    prediction = run_train_test(train_data, dev_data)

    f1_score = f1_score(prediction, labels, average=None)
    for i, _ in enumerate(f1_score):
        print(f"F1 score of class {i}: {_}")
    print("Average F1 score:", np.mean(f1_score))
