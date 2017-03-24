import re
import os               # for environ variables in Part 3
%load_ext autoreload
%autoreload 2
import numpy as np
import pandas as pd
from pyspark.sql.functions import rand
from sparktools import SparkNLPClassifier



# TESTING

### TRAINING RANDOM FOREST on Prefered
model = SparkNLPClassifier()

model.vectorize_train('stars', 4)
testrf = model.train_test_split()

model.train_random_forest()
predictionrf = model.predict(testrf)

rf_pop = predictionrf.select('probability','label').to_pandas


rf_pop.to_json('../data/rf_model_performance/rf_popular.json')

### TRAINING RANDOM FOREST on Relevant
model = SparkNLPClassifier()

model.vectorize_train('useful + funny + cool', 3)
testrf = model.train_test_split()

model.train_random_forest()
predictionrf = model.predict(testrf)

rf_rel = predictionrf.select('probability','label').to_pandas

rf_rel.to_json('../data/rf_model_performance/rf_relevant.json')

### TRAIN NAIVE BAYES ON relevance
model = SparkNLPClassifier()

model.vectorize_train('useful + funny + cool', 3)
testrf = model.train_test_split()

model.train_naive_bayes()
predictionnb = model.predict(testrf)

nb_rel = predictionnb.select('probability','label').to_pandas

nb_rel.to_json('../data/rf_model_performance/nb_relevant.json')

### TRAIN NAIVE BAYES ON Prefered
model = SparkNLPClassifier()

model.vectorize_train('stars', 4)
testrf = model.train_test_split()

model.train_naive_bayes()
predictionnb = model.predict(testrf)

nb_pop = predictionnb.select('probability','label').to_pandas

nb_pop.to_json('../data/rf_model_performance/nb_popular.json')