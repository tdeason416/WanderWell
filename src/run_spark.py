import re
import os               # for environ variables in Part 3
import numpy as np
import pandas as pd
from sparktools import SparkNLPClassifier
import build_df



# TESTING

### TRAINING RANDOM FOREST on Prefered
model = SparkNLPClassifier()

model.vectorize_train('stars', 4)
testrf = model.train_test_split()

model.train_random_forest()
predictionrf = model.predict(testrf)

rf_pop = predictionrf.select('probability','label').toPandas()


rf_pop.to_json('../data/rf_model_performance/rf_popular.json')

build_df.save_file_to_s3('../data/rf_model_performance/rf_popular.json', 'wanderwell-ready')

### TRAINING RANDOM FOREST on Relevant
model = SparkNLPClassifier()

model.vectorize_train('useful + funny + cool', 3)
testrf = model.train_test_split()

model.train_random_forest()
predictionrf = model.predict(testrf)

rf_rel = predictionrf.select('probability','label').toPandas()

rf_rel.to_json('../data/rf_model_performance/rf_relevant.json')

build_df.save_file_to_s3('../data/rf_model_performance/rf_relevant.json', 'wanderwell-ready')

### TRAIN NAIVE BAYES ON relevance
model = SparkNLPClassifier()

model.vectorize_train('useful + funny + cool', 3)
testrf = model.train_test_split()

model.train_naive_bayes()
predictionnb = model.predict(testrf)

nb_rel = predictionnb.select('probability','label').toPandas()

nb_rel.to_json('../data/rf_model_performance/nb_relevant.json')

build_df.save_file_to_s3('../data/rf_model_performance/nb_relevant.json', 'wanderwell-ready')

### TRAIN NAIVE BAYES ON Prefered
model = SparkNLPClassifier()

model.vectorize_train('stars', 4)
testrf = model.train_test_split()

model.train_naive_bayes()
predictionnb = model.predict(testrf)

nb_pop = predictionnb.select('probability','label').toPandas()

nb_pop.to_json('../data/rf_model_performance/nb_popular.json')

build_df.save_file_to_s3('../data/rf_model_performance/nb_popular.json', 'wanderwell-ready')
