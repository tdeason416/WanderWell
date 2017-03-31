import re
import os               # for environ variables in Part 3
import numpy as np
import pandas as pd
from sparktools import SparkNLPClassifier
import build_df

path = '../data/model_performance' 

if not os.path.exists(path):
    os.mkdir(path)
if path[-1] != '/':
    path += '/'

for n in np.arange(10,352,50):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=n)
    predictionnb = model.predict(test)
    floc = '{}{}_trees_rf_3upvotes_1learn_2depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=n)
    predictionnb = model.predict(testrf)
    floc = '{}{}_trees_gbr_3upvotes_1learn_2depth_.json'.format(path,n)
    nb_rel = predictionnb.select('prediction', 'label').toPandas()
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

for n in np.arange(2,14,3):
    model = SparkNLPClassifier()
    model.vectorize_train('useful + funny + cool', thres=3, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=15, max_cats=n)
    predictionnb = model.predict(test)
    floc = '{}{}max_cats_15_trees_rf_3upvotes_1learn_15depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train('useful + funny + cool', thres=3, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=15, max_cats=n)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('prediction','label').toPandas()
    floc = '{}{}max_cats_15trees_gbr_3upvotes_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

for n in np.arange(1,19,3):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=15, max_cats=6)
    predictionnb = model.predict(test)
    floc = '{}{}maxcats_15trees_rf_3upvotes_1learn_15depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=15, max_cats=n)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('prediction','label').toPandas()
    floc = '{}{}_trees_gbr_3upvotes_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

## this one seems to crash memory

for n in np.linspace(.5,5,8):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=n, n_features=8)
    model.train_test_split()
    testrf = model.train_naive_bayes(smoothing = n)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('probability','label').toPandas()
    floc = '{}{}upvotes_nb_1smoothing_.json'.format(int(n*10))
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')
    

for n in np.arange(1,27,5):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=n, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=15, max_cats=6)
    predictionnb = model.predict(test)
    floc = '{}{}maxcats_50trees_rf_3upvotes_1learn_15depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=n, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=15, max_cats=6)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('prediction','label').toPandas()
    floc = '{}{}upvotes_trees_gbr_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

for i in np.logspace(.001,.1, 3):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=15, max_cats=6, learning_rate= n)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('prediction','label').toPandas()
    floc = '{}{}learn_3upvotes_trees_gbr_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')
