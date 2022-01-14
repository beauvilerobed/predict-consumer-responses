import argparse
import json
import logging
import os
import sys
import pickle as pkl

from sklearn.metrics import precision_score
import xgboost as xgb

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _get_train_data_loader(training_dir):
    logger.info("Get train data loader")
    return xgb.DMatrix(
        training_dir.loc[:, training_dir.columns != "target"], label=training_dir["target"]
    )


def _get_test_data_loader(test_dir):
    logger.info("Get test data loader")
    return xgb.DMatrix(
        test_dir.loc[:, test_dir.columns != "target"], label=test_dir["target"]
    )

def train(args):
    logger.info("Hyperparameters: epoch: {}, eta: {}, objective: {}, num_class: {}".format(
                    args.max_depth, args.eta, args.objective, args.num_class)
    )
    param = {"max_depth": args.max_depth, "eta": args.eta, "objective": args.objective, "num_class": args.num_class}
    num_round = 100
    train_loader = _get_train_data_loader(args.train_dir)
    test_loader = _get_test_data_loader(args.test_dir)
    bst = xgb.train(param, train_loader, num_round)

    test(bst, test_loader)
    save_model(bst, args.model_dir)


def test(model, test_loader):
    logger.info("Testing Model on Whole Testing Dataset")
    preds = model.predict(test_loader)
    score=precision_score(test_loader["target"], preds, average='weighted')
    logger.info(f"Testing Percision: {score}")

def model_fn(model_dir):
    with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
        booster = pkl.load(f)
    return booster

def save_model(model, model_dir):
    logger.info("Saving the model.")
    model_location = model_dir + "/xgboost-model"
    pkl.dump(model, open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="multi:softmax")
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument( "--num_class", type=int, default=8)
    parser.add_argument('--num_round', type=int)

    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])

    train(parser.parse_args())