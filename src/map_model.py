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
        Generate city object
        --------
        Parameters
        city: string - name of city
        endtime: str - time of data extraction
        S3: bool - if True, data to be extracted from Wanderwell S3 bucket
        --------
        Returns
        None
        '''
        if S3:
            build_df.extract_from_s3('wanderwell-ready', 'data/', city.lower())
        self.general = pd.read_json('data/{}-clean.json'.format(city.lower()))
        self.bnb = pd.read_json('data/{}-bnb.json'.format(city.lower()))
        self.grid = pd.read_json('data/{}-grid.json'.format(city.lower()))
        self.comments = pd.read_json('data/{}-comments.json'.format(city.lower()))
        self.time_periods = [30, 90, 180, 360, 720]
        if endtime is None:
            self.endtime = pd.Timestamp('2017, 3, 5')
        else:
            self.endtime = pd.Timestamp(endtime)

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
        none
        --------
        Returns
        none
        '''
        no_text = self.comments.drop('content', axis=1)
        no_text['date'] = (pd.Timestamp('2017, 2, 28') - no_text['date']).apply(lambda x: x.days)
        by_user = self._apply_rating_frequency(no_text)
        # by_bus = self._apply_rating_frequency(no_text)
        by_user = by_user[by_user['rating_std'] != 0]
        # by_bus = by_bus[by_bus['rating-std'] != 0]
        fraud_user = by_user['date_range'] > 5
        fraud_user = fraud_user + by_user['rating_count'] < 30
        by_user = by_user[fraud_user]
        s_users = by_user['rating_count'] > 30
        pow_user = (by_user['rating_count'] > 30 + s_users)
        active_user = (by_user['rpd_30'] > .25 + s_users) * 1.5
        endurance_user = (by_user['rpd_720'] > .05 + s_users) * 2.0
        ###apply user rating weights
        by_user['weight'] = pow_user*1.25 + active_user*1.5 + endurance_user*2.0
        by_user['weight'] = by_user['weight']*5 /by_user['weight'].sum() *100
        print by_user.weight.value_counts()
        #### this is bad, dont do this
        for user_rating in by_user['weight'].value_counts().index:
            no_text['rating'] = user_rating * by_user['weight']
        self.user_comment_ratings = no_text



        