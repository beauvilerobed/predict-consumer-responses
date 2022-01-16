import argparse
import json
import logging
import os
import sys
import pickle as pkl

import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb


def _get_train_data():
    train = pd.read_csv ('data/train/train.csv')
    return xgb.DMatrix(
        train.loc[:, train.columns != "target"], label=train["target"]
    )


def _get_test_data():
    test_data = pd.read_csv ('data/test/test.csv')
    df_test = xgb.DMatrix(
        test_data.loc[:, test_data.columns != "target"], label=test_data["target"]
    )
    return test_data, df_test

def train():
    param = {"max_depth": 5, "eta": 0.1, "objective": "multi:softmax", "num_class": 8}
    num_round = 50
    train_loader = _get_train_data()
    test_data, df_test = _get_test_data()
    bst = xgb.train(param, train_loader, num_round)

    test(bst, test_data, df_test)
    save_model(bst)


def test(model, test_data, df_test):
    preds = model.predict(df_test)
    score=f1_score(test_data["target"], preds, average=None)
    print(f"Testing score: {score}")


def save_model(model):
    print("Saving the model.")
    pkl.dump(model, open('xgboost-model', "wb"))
    print("Stored trained model at {}".format('xgboost-model'))


if __name__ == "__main__":
    train()