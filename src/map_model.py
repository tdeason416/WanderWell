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
            build_df.extract_from_s3('wanderwell-ready', 'data/', city)
        self.general = pd.read_json('data/{}-clean.json'.format(city.lower()))
        self.bnb = pd.read_json('data/{}-bnb.json'.format(city.lower()))
        self.grid = pd.read_json('data/{}-grid.json'.format(city.lower()))
        self.comments = pd.read_json('data/{}-comments'.format(city.lower()))
        if endtime is None:
            self.endtime = pd.Timestamp('2017, 3, 5')
        else:
            self.endtime = pd.Timestamp(endtime)

    def _days_since(self, time):
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
        since = self.endtime - time
        return since.apply(lambda x: x.days)

    def rate_comments(self):
        '''
        Assign rank to comments and users
        --------
        Parameters
        self
        --------
        Returns
        something
        '''
        no_text = self.comments.drop('content', axis=1)
        no_text['date'] = (pd.Timestamp('2017, 2, 28') - no_text['date']).apply(lambda x: x.days)
        no_text['date'].apply(self._days_since)
        users = no_text.groupby('user').agg({'median': 'rating', 'stdev': 'rating' ,
                                            'self._days_since': 'date'})
        bus = no_text.groupby('bus_id')
        