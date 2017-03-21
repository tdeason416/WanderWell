'''
Up here there be docstrings ye best be reading them
'''



import re
import pyspark as ps
import os 
import numpy as np
import pandas as pd

from src import build_df
from pyspark.sql import functions as F
from pyspark.sql.functions import rand
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StopWordsRemover
# from pyspark.ml.feature import NGram # maybe



class SparkNLPClassifier(object):
    '''
    this allows you to classify a spark object
    '''
    def __init__(self, local_file=None, s3_bucket='False'):
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
            .appName("yelp_explore") \
            .getOrCreate() 
        if local_file is None
            train_url = 's3n://wanderwell-ready/yelp_academic_reviews.json'
            self.train = spark.read.json(train_url)
            self.test_df = None 
        else:
            self.df = spark.read.json(local_file)

    def split_labeled_sets(self, label, thres=0):
        '''
        Seperates labeled sets into negative and positive values
        --------
        parameters
        label: str - column id for labeled set
        thres: int - value to split label on
        --------
        returns: sparkdf -  modifys original self.train dataframe
        '''
        self.spark.udf.register('imbin', lambda x: 1 if x >= thres else 0)
        self.spark.udf.register('mkascii', self.rem_non_ascii)
        train_df = self.train_df.copy()
        yelp.registerTempTable('train')
        r_bin =  self.spark.sql('''
            SELECT array(mkascii(text)) as content, int(imbin({})) as label
            FROM train
                    '''.format(label))
        mincount_pos = r_bin.filter('label = 1')count()
        mincouint_neg = r_bin.filter('label = 0')count()
        if mincount_pos > mincount_neg:
            mincount = mincount_neg
        else:
            mincount = mincount_pos
        dataset_neg = r_bin.filter('relevant = 0').orderBy(rand()).limit(mincount)
        dataset_pos = r_bin.filter('relevant = 1').orderBy(rand()).limit(mincount)
        df_labels = dataset_pos.union(dataset_neg)
        self.train = df_labels


    def vectorize(self, n_features=2500):
        '''
        generates vectorized features from the self.traindf dataframe
        --------
        Parameters
        n_features: int -  max number of words to be used as features
        --------
        Returns
        None - this function augments original self.train df
        '''
        self.spark.udf.register('listjoin', lambda x: ' '.join(x))
        remover = StopWordsRemover(inputCol="content", outputCol="filtered")
        df_lab_stopped = remover.transform(self.train)
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
        self.train = rescaledData
    
    def train_sentiment(self):
        '''
        '''
        pass


    def rem_non_ascii(self, text):
        lets = []
        ntext = text.lower()
        ntext = re.sub("[^a-z' ]",' ',ntext)
        return ntext.split()

