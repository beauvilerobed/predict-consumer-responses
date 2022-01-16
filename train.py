import argparse
import json
import logging
import os
import sys
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _get_train_data(training_dir):
    logger.info("Get train data loader in {}".format(training_dir))
    data_paths = [i for i in (os.path.join(training_dir, f) for f in os.listdir(training_dir)) if os.path.isfile(i)]
    train = pd.read_csv (data_paths[0])
    return xgb.DMatrix(
        train.loc[:, train.columns != "target"], label=train["target"]
    )


def _get_test_data(validation):
    logger.info("Get test data loader in {}".format(validation))
    data_paths = [i for i in (os.path.join(validation, f) for f in os.listdir(validation)) if os.path.isfile(i)]
    test_data = pd.read_csv (data_paths[0])
    df_test = xgb.DMatrix(
        test_data.loc[:, test_data.columns != "target"], label=test_data["target"]
    )
    return test_data, df_test

def train(args):
    logger.info("Hyperparameters: max_depth: {}, eta: {}, objective: {}, num_class: {}".format(
                    args.max_depth, args.eta, args.objective, args.num_class)
    )
    param = {"max_depth": args.max_depth, "eta": args.eta, "objective": args.objective, "num_class": args.num_class}
    num_round = args.num_round
    train_data = _get_train_data(args.train_dir)
    test_data, df_test = _get_test_data(args.validation)
    bst = xgb.train(param, train_data, num_round)

    test(bst, test_data, df_test)
    save_model(bst, args.model_dir)


def test(model, test_data, df_test):
    logger.info("Testing Model on Testing Dataset")
    preds = model.predict(df_test)
    score1=f1_score(test_data["target"], preds, average='weighted')
    logger.info(f"f1-score:{score1}")
    logger.info(f"[2]#011validation-f1:{score1}")
    

def save_model(model, model_dir):
    logger.info("Saving the model.")
    model_location = model_dir + "/xgboost-model"
    pkl.dump(model, open(model_location, "wb"))
    logger.info("Stored trained model at {}".format(model_location))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="multi:softmax")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument( "--num_class", type=int, default=8)
    parser.add_argument('--num_round', type=int, default=50)

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    train(parser.parse_args())

def model_fn(model_dir):
    with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
        booster = pkl.load(f)
    return booster