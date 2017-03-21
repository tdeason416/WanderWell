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
            train_url = 's3n://wanderwell-ready/yelp_academic_reviews.json'
            self.train = self.spark.read.json(train_url)
        else:
            self.train = self.spark.read.json(local_file)
        self.train_popular = None
        self.train_positive = None

    def generate_binary_labels(self, df, label, thres=0):
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

    def train_vectorize(self):
        '''
        Applies vectorize to the training set
        '''
        self.train_popular = self.generate_binary_labels(self.train, 'useful + funny + cool', 3)
        self.train_popular = self.split_labeled_sets(self.train_popular, 'useful + funny + cool')
        self.train_popular = self.vectorize(self.train_popular, 2500)
        self.train_positive = self.generate_binary_labels(self.train, 'stars', 4)
        self.train_positive = self.split_labeled_sets(self.train_positive, 'stars')
        self.train_positive = self.vectorize(self.train_positive, 2500)

    def vectorize(self, df, n_features=2500):
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
    
    def train(self):
        '''

        '''
        pass

    def _rem_non_letters(self, text):
        '''
        removes all non lowercase letters and spaces from string
        --------
        Parameters
        text: string - text to convert
        --------
        Returns: ntext.split() - list of lowercase single word strings
        '''
        lets = []
        ntext = text.lower()
        ntext = re.sub("[^a-z' ]",' ',ntext)
        return ntext.split()

