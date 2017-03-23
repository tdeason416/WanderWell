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
        Otherwise information is extracted from s3.
        --------
        Parameters
        city: string - name of city
        endtime: str - time of data extraction
        S3: bool - if True, data to be extracted from Wanderwell-ready S3 bucket.
        --------
        Returns
        None
        '''
        if S3:
            build_df.extract_from_s3('wanderwell-ready', '../data/', city.lower())
        self.general = pd.read_json('../data/{}-clean.json'.format(city.lower()))
        self.bnb = pd.read_json('../data/{}-bnb.json'.format(city.lower()))
        self.grid = pd.read_json('../data/{}-grid.json'.format(city.lower()))
        self.comments = pd.read_json('../data/{}-comments.json'.format(city.lower()))
        self.comments_nlp_ratings = pd.read_json('../data/{}-c_ratings.json'\
                                                                .format(city.lower()))
        self.time_periods = [30, 90, 180, 360, 720]
        self.user_comment_ratings = None
        if endtime is None:
            self.endtime = pd.Timestamp('2017, 3, 5')
        else:
            self.endtime = pd.Timestamp(endtime)

    def _add_comments_ratings(self):
        '''
        Adds comment relevance ratings derived from SPARK nlp to self.comments
        '''
        self.comments['relevance'] = self.comments_nlp_ratings['relevance']

    def _apply_rating_frequency(self, df):
        '''
        function to convert datetime to days since a given endtime
        --------
        Parameters
        time : pd.Series - data which time delta applies to
        endtime: datetime object - date of relative measurement
        --------
        Returns
        days: Pd.Series - integer timedelta in days to the endtime
        '''
        df_ = df.copy()
        agg_dict = {'rating': ['count', 'median', 'std'], 'date': ['min', 'max']}
        for num in self.time_periods:
            df_['date{}'.format(num)] = df_['date'] <= num * df_['date']
            agg_dict['date{}'.format(num)] = 'sum'
        grouped = df_.groupby('user').agg(agg_dict)
        grouped.fillna(0, inplace=True)
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped['date_range'] = grouped['date_max'] - grouped['date_min']
        for num in self.time_periods:
            grouped['rpd_{}'.format(num)] = grouped['date{}_sum'.format(num)]/num
            grouped.drop('date{}_sum'.format(num), axis=1, inplace=True)
        return grouped

    def rate_user_comments(self):
        '''
        Assign rank to comments by user
        --------
        Parameters
        None - defined through init
        --------
        Returns
        None
        '''
        self._add_comments_ratings()
        no_text = self.comments.drop('content', axis=1)
        no_text['date'] = (pd.Timestamp('2017, 2, 28') - no_text['date']).apply(lambda x: x.days)
        by_user = self._apply_rating_frequency(no_text)
        # by_bus = self._apply_rating_frequency(no_text)
        by_user = by_user[by_user['rating_std'] != 0]
        # by_bus = by_bus[by_bus['rating-std'] != 0]
        counts = by_user['rating_count'].describe().values
        fraud_user = by_user['date_range'] < 30
        # fraud_user = by_user['rating_count'] > 30 + fraud_user
        by_user = by_user[fraud_user]
        s_users = by_user['rating_count'] > 30
        twosig = counts[1] + 1.5 * counts[2]
        pow_user = (by_user['rating_count'] > twosig + s_users)
        active_user = (by_user['rpd_30'] > .25 + s_users) * 1.5
        endurance_user = (by_user['rpd_720'] > .05 + s_users) * 2.0
        ###apply user rating weights
        # print by_user['rating_count'].describe()
        # print by_user['date_range'].describe()
        print by_user['rating_count'].nlargest(20)
        print pow_user.sum()
        print active_user.sum()
        print endurance_user.sum()
        by_user['weight'] = (pow_user*1.25 + active_user*1.5 + endurance_user*2.0)
        # by_user['weight'] = by_user['weight']*5 /by_user['weight'].sum() *100
        # print by_user.weight.value_counts()
        #### this is bad, dont do this
        for user_rating in by_user['weight'].value_counts().index:
            no_text['weighted_rating'] = user_rating * no_text['rating']
        self.user_comment_ratings = no_text

    def Assign_poi_ratings(self, trending_boost):
        '''
        Assigns user reviews to buisness locations
        --------
        Parameters
        trending_boost: float
        --------
        Returns
        biz_ratings: pd.DataFrame - ratings of buisneses
        '''
        # u_ratings = self.comment_ratings
        pass

city = 'Seattle'
seattle = CityValues(city)
seattle.rate_user_comments()



        