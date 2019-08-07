# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt


def sigmoid(y_pred):
    return 1/(1+np.exp(-y_pred))


def lg_loss_gbm(gbm_score, y_true):
    y_loss = []
    min_loss = 1
    for i, y_pred in enumerate(gbm_score):
        y_sigmed = []
        for y_iter in y_pred:
            y_sigmed.append(sigmoid(y_iter))
        loss = log_loss(y_true, y_sigmed)
        if loss < min_loss:
            min_loss = loss
            min_iter = i
        y_loss.append(loss)
    return y_loss, min_loss, min_iter


data = pd.read_csv('gbm-data.csv').to_numpy()
y = data[:, 0]
X = data[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8,
                                                    random_state=241)

for lr in [0.2 ]: # [1, 0.5, 0.3, 0.2, 0.1]: # todo uncomment
    gbm = GradientBoostingClassifier(n_estimators=250, verbose=True,
                                     random_state=241, learning_rate=lr)
    gbm.fit(X_train, y_train)

    train_loss, _, _ = lg_loss_gbm(gbm.staged_decision_function(X_train), y_train)
    test_loss, min_loss, min_iter = lg_loss_gbm(gbm.staged_decision_function(X_test), y_test)
    print(min_loss, min_iter)
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()

# forest

rfc = RandomForestClassifier(n_estimators=36, random_state=241)
rfc.fit(X_train,y_train)
print(log_loss(y_test, rfc.predict_proba(X_test)))
