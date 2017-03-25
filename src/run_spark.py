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


### train max number of trees for RF
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
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

for n in np.linspace(1,12,3):
    model = SparkNLPClassifier()
    model.vectorize_train('useful + funny + cool', thres=3, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=50, max_cats=n)
    predictionnb = model.predict(test)
    floc = '{}max_cats_50_trees_rf_3upvotes_1learn_15depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=50, max_cats=n)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('probability','label').toPandas()
    floc = '{}{}max_cats_50trees_gbr_3upvotes_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

for n in np.arange(1,19,3):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=50, max_cats=6)
    predictionnb = model.predict(test)
    floc = '{}{}maxcats_50trees_rf_3upvotes_1learn_15depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=50, max_cats=n)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('probability','label').toPandas()
    floc = '{}{}_trees_gbr_3upvotes_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

for n in np.linspace(.5,5,8):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=n, n_features=8)
    model.train_test_split()
    testrf = model.train_naive_bayes(smoothing = 1.0)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('probability','label').toPandas()
    floc = '{}{}upvotes_nb_1smoothing_.json'.format(n*100)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')
    
for n in np.arange(1,27,5):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=n, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=50, max_cats=6)
    predictionnb = model.predict(test)
    floc = '{}{}maxcats_50trees_rf_3upvotes_1learn_15depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=n, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=50, max_cats=6)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('probability','label').toPandas()
    floc = '{}{}upvotes_trees_gbr_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=n, n_features=8)
    test = model.train_test_split()
    model.train_random_forest(depth=15, n_trees=50, max_cats=6)
    predictionnb = model.predict(test)
    floc = '{}{}upvotes_50trees_rf_1learn_15depth_.json'.format(path,n)
    nb_rel = predictionnb.select('probability', 'label').toPandas()
    nb_rel.to_json(floc.format(path,n))
    build_df.save_file_to_s3(floc, 'wanderwell-ready')

for i in np.logspace(.0001,.1, 10):
    model = SparkNLPClassifier()
    model.vectorize_train( 'useful + funny + cool', thres=3, n_features=8)
    testrf = model.train_test_split()
    model.train_boosted_regression(depth=2, n_trees=100, max_cats=6)
    predictionnb = model.predict(testrf)
    nb_rel = predictionnb.select('probability','label').toPandas()
    floc = '{}{}learn_3upvotes_trees_gbr_1learn_2depth_.json'.format(path,n)
    nb_rel.to_json(floc)
    build_df.save_file_to_s3(floc, 'wanderwell-ready')
