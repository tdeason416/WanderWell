import re
import pyspark as ps
import os 
import numpy as np
import pandas as pd

from src import build_df
from pyspark.sql import functions as F
from pyspark.sql.functions import rand
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
from pyspark.ml.classification import NaiveBayes
# from pyspark.ml.feature import NGram # maybe

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer

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
        TODO
        '''
        self.spark = spark = ps.sql.SparkSession.builder \
            .appName("reviews_nlp") \
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
        Applies vectorize to the training set
        '''
        self.train = self.generate_binary_labels(self.data, label, thres)
        self.train = self.split_labeled_sets(self.train, label)
        self.train = self.vectorize(self.train, n_features)

    def vectorize(self, df, n_features=30):
        '''
        generates vectorized features from the self.traindf dataframe
        --------
        Parameters
        df: spark dataframe - object to be featurized
        n_features: int -  max number of words to be used as features
        --------
        Returns
        None - this function augments original self.train df
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
        number_of_folds: int - number of cross val cells
        --------
        Returns
        None - populates the self.Kfolds arguement
        '''
        model = model_classifier(labelCol='label', featuresCol='features')
        evaluator = MulticlassClassificationEvaluator()
        pipe = Pipeline(stages=[model])
        crossval = CrossValidator(
                        estimator= pipe,
                        estimatorParamMaps= paramgrid,
                        evaluator= evaluator,
                        numFolds= number_of_folds)
        return crossval.fit(df)

    def train_boosted_regression(self, depth=3, n_trees=100, 
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
        gbr = GBTClassifier(labelCol='label', featuresCol="features",
                             maxDepth=depth, maxIter=n_trees, stepSize=learning_rate, maxMemoryInMB=10000)
        pipeline = Pipeline(stages=[featureIndexer, gbr])
        # Train model.  This also runs the indexer.
        self.model = pipeline.fit(self.train)

    def train_random_forest(self, depth=3, n_trees=100, max_cats= 6):
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
                                maxDepth=depth, numTrees=n_trees, maxMemoryInMB=10000)
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

    def generate_confusion_matrix(self, test, thres):
        '''
        generates confusion matrix from trained model
        --------
        Parameters
        thres: float - threshold probability for positve outcome
        --------
        Returns
        dict - containing rate of probs at the given thres
        '''
        cm = {}
        cm['thres'] = thres
        prediction = self.predict(test)
        self.spark.udf.register('flbin', lambda x: 1 if x[1] > thres else 0) # this does not work
        print prediction.select('probability').show(20)
        prediction.registerTempTable('prob')
        cfm = self.spark.sql('''SELECT flbin(probability) as pred, label FROM prob''')
        print cfm.select('pred', 'label').show(20)
        cfm.registerTempTable('cfm')
        sql_temp = 'SELECT label - pred as result FROM cfm WHERE label = {}'
        is_pos = self.spark.sql(sql_temp.format(1))
        is_neg = self.spark.sql(sql_temp.format(0))
        cm['tp'] = is_pos.filter('result = 0').count()
        cm['fp'] = is_pos.count() - cm['tp']
        cm['fn'] = is_neg.filter('result = -1').count()
        cm['tn'] = is_neg.count() - cm['fn']
        return cm

    def evaluate_model(self, test, number_of_iterations):
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
        acclist = []
        self.spark.udf.register('getsecond', lambda x: x[1])
        probs = self.predict(test)
        probs.registerTempTable('probs')
        tlat = '''SELECT getsecond(probability) as probs, label FROM probs WHERE {} {}'''
        # print tlat.format('label', '> 0')
        c_true = self.spark.sql(tlat.format('label', '> 0'))
        c_false = self.spark.sql(tlat.format('label', '< 1'))
        for thres in np.linspace(.001, .999, number_of_iterations):
            cfdict = {'thres': thres}
            cfdict['tp'] = c_true.filter('probs > {}'.format(thres)).count()
            cfdict['fn'] = c_true.filter('probs < {}'.format(thres)).count()
            cfdict['fp'] = c_false.filter('probs > {}'.format(thres)).count()
            cfdict['fn'] = c_false.filter('probs < {}'.format(thres)).count()
            cfdict['tpr'] = cfdict['tp']/(cfdict['tp'] + cfdict['fn'])
            cfdict['fpr'] = cfdict['fp']/(cfdict['fp'] + cfdict['tn'])
            print 'tp= {}, fn= {}, fp= {}'.format(cfdict['tp'], cfdict['fn'], cfdict['fp']) 
            acclist.append(cfdict)
        return acclist

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

