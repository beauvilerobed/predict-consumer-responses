# Starbucks Rewards: Predicting Consumer Responses

## Dataset

### Overview
The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data.

## Model Training
Using Gradient Boosting, a supervised learning method, we will analyze the attributes of customers to create customer classifications.
This is a multi-class classification problem so the key metric we will use is the f1-score. The simulating dataset only has one product, while Starbucks offers dozens of products. Therefore, this data set is a simplified version of the real Starbucks app.

## Machine Learning Pipeline
- Cleaning
- Feature engineering
- Split
- Training
- Inference

To finish this project, we perform the following tasks:

1. Upload Training Data: First you will have to upload the training data to an S3 bucket.
2. Model Training Script: Once we have done that, we will have to write a script to train a model on that dataset.
3. Train in SageMaker: Finally, we will have to use SageMaker to run that training script and train your model


## Standout Suggestions

* **Model Deployment:** Once we have trained your model, we can deploy our model to a SageMaker endpoint and then query it with an image to get a prediction.
* **Hyperparameter Tuning**: To improve the performance of our model, we can use SageMaker’s Hyperparameter Tuning to search through a hyperparameter space and get the value of the best hyperparameters.

