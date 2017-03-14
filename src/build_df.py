import boto3
import os

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import sys
# sys.setdefaultencoding('UTF8')


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
        city_df[cat].fillna(False, inplace=True)
    return city_df


# def split_dict_rows(cell, appendee):
#     '''
#     Splits cells from list of rows into 
#     ========
#     PARAMETERS
#     cell: cell containing dictionary
#     appendee: lit to add dictionaries to
#     =======
#     RETURNS
#     None
#     '''
#     appendee.append(dict)


def add_list_contents_to_df(row, appendee):
    '''
    appends rows in df to new columns
    ========
    PARAMETERS
    row: dataframe row containing list if items
    apandee: list to add items onto
    ========
    RETURNS
    None
    '''
    reviews = {}
    i= 0
    for review in row:
        i += 1
        reviews['review-{}'.format(i)] = review
    appendee.append(reviews)


def create_reviews_df(df):
    '''
    Generates reviews dataframe from existing object
    ========
    PARAMETERS
    df: pandas df - from create_flattened_dataframe function
    ========
    RETURNS
    review_df: pandas df - dataframe containing reviews for analysis
    '''
    reviews_1 = df['id']
    keys = df['reviews'].values[0][0].keys()
    review_rows = []
    df['reviews'].apply(add_list_contents_to_df, appendee= review_rows)
    reviews = pd.DataFrame(review_rows)
    reviews_ = reviews.set_index(reviews_1)
    reviews_ = reviews_.stack()
    reviews_ = reviews_.reset_index()
    expanded_reviews = [reviews_.drop(0, axis=1), pd.DataFrame(list(reviews_[0].values))]
    reviews_df = pd.concat(expanded_reviews, axis=1)
    reviews_df['rating-date'] = reviews_df['rating-date'].apply(pd.Timestamp)
    reviews_df['rating'] = reviews_df['rating'].apply(lambda x: float(x.split()[0]))
    reviews_df.drop('user_location', axis=1, inplace=True)
    reviews_df['level_1'] = reviews_df['level_1'].apply(lambda x: int(x.split('-')[-1]))
    reviews_df.columns = ['bus_id', 'review no', 'content', 'rating', 'date', 'user']
    return reviews_df


def create_photos_df(df):
    '''
    creates simple photos df associating photos with a bus id
    ========
    PARAMETERS
    df: pandas df - from create_flattened_dataframe function
    ========
    RETURNS
    new_df: pandas df - dataframe containing reviews
    '''
    pass


def create_general_df(df):
    '''
    Create general df for buisness Identification and rating.
    PARAMETERS
    df: pandas df - from create_flattened_dataframe function
    ========
    RETURNS
    new_df: pandas df - dataframe containing reviews
    '''
    pass


def save_df_to_json(df, file_location): # i now realize this function is stupid
    '''
    Stores pandas dataframe as single .json file
    ========
    PARAMETERS
    df: pandas dataframe 
    file_location: string - location of file to save df data
    ========
    RETURNS
    None
    '''
    df.to_json(file_location)


def save_file_to_s3(file_location, bucket_name, bucket_key):
    '''
    Stores pandas dataframe as single .json file in save_df_to_s3_json
    ========
    PARAMETERS
    file_location: string - location of file to upload to s3
    bucket_name: string - s3 bucket to store .json file
    bucket_key: string - desired file name on s3
    ========
    RETURNS
    None
    '''
    aws = boto3.resource('s3')
    ww_all = aws.Bucket(bucket_name)
    ww_all.upload_file(file_location, bucket_key)