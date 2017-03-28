import re
import pyspark as ps
from pyspark import SparkConf
import os 
import numpy as np
import pandas as pd

import build_df
from pyspark.sql.functions import rand
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover, StringIndexer, VectorIndexer
# from pyspark.ml.feature import NGram # maybe

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics

class SparkNLPClassifier(object):
    '''
    Builds trained model in spark based on labeled yelp_academic_dataset, 
    and evaluates data based on output of the training set.
    '''
    def __init__(self, local_file=False):
        '''
        Generates Spark Dataframe from wander-wall for NLP analysis
        --------
        PARAMETERS
        json_location: string -  local file or s3 file to use when creating df
        s3_bucket:
        --------
        RETURNS
        self.data: spark dataframe containing training set.
        '''
        self.spark = ps.sql.SparkSession.builder \
            .master("spark://master:7077") \
            .appName("nlp_reviews") \
            .getOrCreate()

        if not local_file:
            train_url = 's3n://wanderwell-ready/yelp_academic_dataset_review.json'
            self.data = self.spark.read.json(train_url)
        else:
            self.data = self.spark.read.json(local_file)
        self.train = None
        self.train_split = None
        self.model = None 

    def generate_binary_labels(self, df, label, thres=1):
        '''
        Converts given label in dataframe into binary value, and drops all other labels
        '''
        def _rem_non_letters(text):
            lets = []
            ntext = text.lower()
            ntext = re.sub("[^a-z' ]", ' ', ntext)
            return ntext.split()
        self.spark.udf.register('imbin', lambda x: 1 if x >= thres else 0)
        self.spark.udf.register('words_only', _rem_non_letters)
        df.registerTempTable('df')
        r_bin = self.spark.sql('''
            SELECT array(words_only(text)) as content, int(imbin({})) as label
            FROM df
                    '''.format(label))
        return r_bin

    def split_labeled_sets(self, df, label):
        '''
        Seperates labeled sets into negative and positive values
        --------
        parameters
        label: str - column id for labeled set
        thres: int - value to split label on
        --------
        returns: sparkdf -  modifys original self.train dataframe
        '''
        mincount_pos = df.filter('label = 1').count()
        mincount_neg = df.filter('label = 0').count()
        if mincount_pos > mincount_neg:
            mincount = mincount_neg
        else:
            mincount = mincount_pos
        dataset_neg = df.filter('label = 0').orderBy(rand()).limit(mincount)
        dataset_pos = df.filter('label = 1').orderBy(rand()).limit(mincount)
        return dataset_pos.union(dataset_neg)

    def vectorize_train(self, label, thres=1, n_features=20):
        '''
        Applies generate_binary_labels, split labeled sets
        and vectorize to the training set
        --------
        Parameters:
        label: str - name of the label variable
        thres: inclusive mininmum value for positive label
        n_features: number of terms to be vectorized per set
        --------
        Returns
        populates self.train
        '''
        self.train = self.generate_binary_labels(self.data, label, thres)
        self.train = self.split_labeled_sets(self.train, label)
        self.train = self.vectorize(self.train, n_features)

    def vectorize(self, df, n_features=8):
        '''
        generates vectorized features from the self.traindf dataframe
        --------
        Parameters
        df: spark dataframe - object to be featurized
        n_features: int -  max number of words to be used as features
        --------
        Returns
        None - Vectorized and rescaled data.
        '''
        self.spark.udf.register('listjoin', lambda x: ' '.join(x))
        remover = StopWordsRemover(inputCol="content", outputCol="filtered")
        df_lab_stopped = remover.transform(df)
        df_lab_stopped.registerTempTable('df_lab_stopped')
        stop_strings = self.spark.sql('''
                    SELECT listjoin(filtered) as filtered, content, label
                    FROM df_lab_stopped
                    ''')
        tokenizer = Tokenizer(inputCol="filtered", outputCol="words")
        wordsData = tokenizer.transform(stop_strings)
        hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=n_features)
        featurizedData = hashingTF.transform(wordsData)
        featurizedData.cache()
        idf = IDF(inputCol="rawFeatures", outputCol="features")
        idfModel = idf.fit(featurizedData)
        rescaledData = idfModel.transform(featurizedData)
        return rescaledData

    def train_test_split(self, test_size= .2):
        '''
        Split dataset into training values and testing values
        --------
        Parameters
        test_size: float - percent of data to use for testing
        --------
        Returns
        test_set: pulls test set from training_values
        '''
        training_data, testing_data = self.train.randomSplit([1-test_size, test_size])
        self.train = training_data
        return testing_data


    def cross_val_eval(self, df, model_classifier, paramgrid, number_of_folds=5):
        '''
        Performs Kfold split on training set for cross validation
        --------
        Parameters
        df: spark.df - training df to cross_val
        model_classifier - spark ML classsifier
        paramgrid - dict - terms to be optimized
        number_of_folds: int - number of cross val cells
        --------
        Returns
        crossval.fit() - fitted cross validation function
        '''
        model = model_classifier(labelCol='label', featuresCol='features')
        evaluator = BinaryClassificationEvaluator()
        pipe = Pipeline(stages=[model])
        crossval = CrossValidator(
                        estimator= pipe,
                        estimatorParamMaps= paramgrid,
                        evaluator= evaluator,
                        numFolds= number_of_folds)
        return crossval.fit(df)

    def train_boosted_regression(self, depth=2, n_trees=100,
                                 learning_rate= .01, max_cats= 6):
        '''
        train dataset on boosted decision trees
        --------
        Parameters
        depth: int -  max_allowable depth of decision tree leafs
        n_trees: int - max number of iterations
        learning_rate: int - rate which the model fits
        --------
        '''
        featureIndexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=max_cats).fit(self.train)
        gbr = GBTRegressor(labelCol='label', featuresCol="features",
                             maxDepth=depth, maxIter=n_trees, stepSize=learning_rate, maxMemoryInMB=2000)
        pipeline = Pipeline(stages=[featureIndexer, gbr])
        # Train model.  This also runs the indexer.
        self.model = pipeline.fit(self.train)

    def train_random_forest(self, depth=3, n_trees=100, max_cats=6):
        '''
        train dataset on random forest classifiers
        --------
        Parameters
        depth: int -  max_allowable depth of decision tree leafs
        n_trees: int - max number of trees
        --------
        '''
        featureIndexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=max_cats).fit(self.train)
        gbr = RandomForestClassifier(labelCol='label', featuresCol="features", probabilityCol="probability",
                                maxDepth=depth, numTrees=n_trees, maxMemoryInMB=2000)
        pipeline = Pipeline(stages=[featureIndexer, gbr])
        # Train model.  This also runs the indexer.
        self.model = pipeline.fit(self.train)

    def train_naive_bayes(self, smoothing= 1.0):
        '''
        train dataset on naive bayes algo
        --------
        Parameters
        smoothing = float
        --------
        Returns
        None
        '''
        # create the trainer and set its parameters
        nb = NaiveBayes(smoothing=smoothing, modelType="multinomial")
        self.model = nb.fit(self.train)

    def predict(self, test):
        '''
        Generates probabilties for test set based on the fitted models of this class
        --------
        Parameters
        test: spark.df - test data
        --------
        Returns
        predict - spark.df with probabbility row added
        '''
        probability = self.model.transform(test)
        return probability

    def evaluate_model_simple(self, test):
        '''
        generate tpr, fpr, fnr, and tpr for each threshold
        --------
        Parameters:
        test: spark.df post vectorization
        number_of_iterations: number of threshold values between .001 and 1.00 utilized in roc curve
        --------
        Returns:
        list-of-dict - containing rate of pthres, tp, fp, fn, tn
        '''
        score_model = {}
        predictionAndLabels = test.rdd.map(lambda lp: (float(self.model.predict(lp.features)), lp.label))
        # Instantiate metrics object
        metrics = BinaryClassificationMetrics(predictionAndLabels)
        metrics2 = MulticlassMetrics(predictionAndLabels)
        # Area under precision-recall curve
        score_model['precision_recall'] = metrics.areaUnderPR
        # Area under ROC curve
        score_model["ROC_area"] = metrics.areaUnderROC
        score_model['tpr'] = metrics2.truePositiveRate('label')
        score_model['fpr'] = metrics2.falsePositiveRate('label')
        return score_model

    def _rem_non_letters(self, text):
        '''
        removes all non lowercase letters and spaces from string
        --------
        Parameters
        text: string - text to convert
        --------
        Returns: ntext.split() - list of lowercase single word strings
        '''
        ntext = text.lower()
        ntext = re.sub("[^a-z' ]", ' ', ntext)
        return ntext.split()

