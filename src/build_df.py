import boto3
import os

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
# sys.setdefaultencoding('UTF8')


### only run this code if you need to update from s3 bucket
def extract_from_s3(bucket_name, local_folder):
    ''' 
    saves json files from s3 bucket locally requires acccess key to bucket.
    ========
    PARAMETERS:
    bucket_name: str - s3 bucket where json is stored
    local_folder: location to save files locally
    ========
    RETURNS:
    None - output is sent to local file
    '''
    fnames = []
    files = set(os.listdir(local_folder))
    aws = boto3.resource('s3')
    awsbucket = aws.Bucket(bucket_name)
    for name in awsbucket.objects.all():
        fnames.append(name.key)
    for name in awsbucket.objects.all():
        fname = name.key
        if fname not in files and '.json' in fname:
            aws.meta.client.download_file(bucket_name,
                                fname,'{}{}'.format(local_folder, fname))

def create_flattened_dataframe(json_folder):
    '''
    loads locally stored single line .json files into flattned dataframe
    ========
    PARAMETERS
    json_folder: local folder containing only json to analyze
    ========
    RETURNS
    city_df:  pandas dataframe - coarse dataframe of output data
    '''
    daylist = ['mon', 'tues', 'wed', 'thurs', 'fri', 'sat', 'sun']
    poi_list = []
    cat_set = set()
    for bus in os.listdir(json_folder):
        if bus.split('.')[-1] == 'json':
            with open(json_folder+bus) as busfile:
                try: 
                    bus_json = json.load(busfile)
                # if json file is currupt, or is missing data
                except Exception:
                    continue
                cats = bus_json.pop('categories')
                ### seperate hours into daily columns
                if 'hours' in bus_json:
                    hours_dict = bus_json.pop('hours')[0]['open']
                else:
                    hours_dict = None
                for idx, day in enumerate(daylist):
                    try: 
                        bus_json['hours_open_{}'.format(day)] = hours_dict[idx]['start']
                    except Exception: 
                        bus_json['hours_open_{}'.format(day)] = False
                    try: 
                        bus_json['hours_closed_{}'.format(day)] = hours_dict[idx]['end']
                    except Exception: 
                        bus_json['hours_closed_{}'.format(day)] = False
                ### seprate catagories into individual columns (3 total)
                for idx, val in enumerate(cats):
                    bus_json['category-{}'.format(idx)] = val['alias']
                    cat_set.add('cat: {}'.format(val['alias']))
                poi_list.append(pd.io.json.json_normalize(bus_json))
    city_df = pd.concat(poi_list)
    for cat in ['category-0', 'category-1', 'category-2'] :
        city[cat].fillna(False, inplace=True)
    return city_df

def save_df_to_s3_json(df, bucket_name, file_name):
    '''
    Stores pandas dataframe as single .json file in save_df_to_s3_json
    ========
    PARAMETERS
    df: pandas dataframe - data to store in S3 bucket for later analysis
    bucket_name: string - s3 bucket to store .json file
    file_name: string - desired file name
    ========
    RETURNS
    None
    '''
    aws = boto3.resource('s3')



    


def extract_reviews_df():
    '''
    docstrings bitches
    '''
    pass