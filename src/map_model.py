'''
DOCSTRINGS 4 DAYS
'''
import build_df
import boto3
import os
import json
import numpy as np
import pandas as pd

class CityValues(object):
    '''
    A modeling framework for city object which outputs heatmap criteria
    '''
    def __init__(self, city, endtime=None, S3=False):
        '''
        Generate city object.  Extracts data from data folder if exists
        Otherwise information is saved from s3 to the data folder.
        --------
        Parameters
        city: string - name of city
        endtime: str - time of data extraction
        S3: bool - if True, data to be extracted from wanderwell-ready S3 bucket.
        -------
        Returns
        None
        '''
        if S3:
            build_df.extract_from_s3('wanderwell-ready', '../data/', city.lower())
        self.general = pd.read_json('../data/{}-clean.json'.format(city.lower()))
        self.bnb = pd.read_json('../data/{}-bnb.json'.format(city.lower()))
        self.grid = pd.read_json('../data/{}-grid.json'.format(city.lower()))
        self.comments = pd.read_json('../data/{}-comments.json'.format(city.lower()))
        self.nlp_ratings = pd.read_json('../data/{}-c_ratings.json'\
                                                            .format(city.lower()))
        self.weighted_ratings = None
        self.time_periods = [30, 90, 180, 360, 720]
        if endtime is None:
            self.endtime = pd.Timestamp('2017, 3, 5')
        else:
            self.endtime = pd.Timestamp(endtime)

    def _add_comments_ratings(self):
        '''
        Adds comment relevance ratings derived from SPARK nlp to self.comments
        '''
        self.comments['relevance'] = self.nlp_ratings.values

    def _apply_rating_frequency(self, df, group_col):
        '''
        function to convert datetime to days since a given endtime
        --------
        Parameters
        df: pd.Dataframe -  dataframe which contains items to be grouped
        group_col: string - column name of groupby col
        --------
        Returns
        grouped: Pd.Series - integer timedelta in days to the endtime
        '''
        df_ = df.copy()
        agg_dict = {'rating': ['count', 'median', 'std'], 'date': ['min', 'max']}
        for num in self.time_periods:
            df_['date{}'.format(num)] = df_['date'] <= num * df_['date']
            agg_dict['date{}'.format(num)] = 'sum'
        grouped = df_.groupby(group_col).agg(agg_dict)
        grouped.fillna(0, inplace=True)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped['date_range'] = grouped['date_max'] - grouped['date_min']
        for num in self.time_periods:
            grouped['rpd_{}'.format(num)] = grouped['date{}_sum'.format(num)]/num
            grouped.drop('date{}_sum'.format(num), axis=1, inplace=True)
        return grouped


    def _identify_fraud_users(self):
        '''
        Identifys fradulant users in the self.comments df
        '''
        by_date = self.comments.groupby(['user', 'date']).count()
        by_date_ = by_date.reset_index()
        by_users = by_date_.groupby('user').agg({'content' : ['median', 'std', 'max']}).fillna(10)
        by_users.columns = [col[1].strip() for col in by_users.columns.values]
        fraud = by_users[(by_users['median'] <= 1) & (by_users['max'] > 15) & (by_users['std'] < 5)]
        return fraud.index

    def rate_user_comments(self):
        '''
        Assign rank to comments by user
        --------
        Parameters
        None - defined through init
        --------
        Returns
        pd.Dataframe - Creates self.weighted_ratings dataframe
        '''
        self._add_comments_ratings()
        no_text = self.comments.drop('content', axis=1)
        no_text['date'] = (pd.Timestamp('2017, 2, 28') - no_text['date']).apply(lambda x: x.days)
        no_text['positive'] = no_text['rating'].apply(lambda x: True if x > 4 else False)
        by_user = self._apply_rating_frequency(no_text, 'user')
        by_user = by_user[by_user['rating_std'] != 0]
        counts = by_user['rating_count'].describe().values
        fraudsters = self._identify_fraud_users()
        by_user.drop(fraudsters, inplace=True)
        s_users = by_user['rating_count'] > 30
        twosig = counts[1] + 1.5 * counts[2]
        pow_user = (by_user['rating_count'] > twosig + s_users)
        active_user = (by_user['rpd_30'] > .25 + s_users) * 1.5
        endurance_user = (by_user['rpd_720'] > .05) * 2.0
        by_user['weight'] = (pow_user*1.5 + active_user*1.25 + endurance_user*2.0)
        no_text_neg = no_text[no_text['positive'] == False]
        no_text_neg['rating'].apply(lambda x: 4 - x)
        no_text_pos = no_text[no_text['positive']]
        no_text_pos['rating'] = no_text_pos['rating'].apply(lambda x: x - 3)
        for user_rating in by_user['weight'].value_counts().index:
            no_text_neg['weighted_rating'] = -(user_rating * no_text_neg['rating'])
            no_text_pos['weighted_rating'] = user_rating * no_text_pos['rating']
        rating_sum = pd.concat([no_text_neg, no_text_pos])
        rating_sum['rating'] = rating_sum['weighted_rating'] * rating_sum['relevance']
        self.weighed_ratings = rating_sum.drop('weighted_rating', axis=1)

    def assign_poi_ratings(self):
        '''
        Assigns user reviews to buisness locations
        --------
        Parameters
        trending_boost: float
        --------
        Returns
        biz_ratings: pd.DataFrame - ratings of buisneses
        '''
        bus_ratings = self._apply_rating_frequency(self.weighed_ratings, 'bus_id')
        print bus_ratings.head()




        