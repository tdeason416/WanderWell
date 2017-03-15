'''
TOOLS FOR IMPORTING ORGANIZING AND STORING REVIEW DATA
---------------------
extract_from_s3: Saves files from directory to local folder for faster processing
create_flattened_dataframe: converts local folder of json files into pandas dataframe of reviews
create_reviews_df: generate dataframe of review for modeling
remote_unwanted_POIs: removes all unwanted Points of interest (points which do not add value to the model)
create_photos_df: **WORK IN PROGRESS**, generates dataframe of photos for modeling
create_general_df: creates analysis df from existing flatttned and 
save_df_to_json: saves pandas dataframe as json object
save_file to s3: archives file to s3 bucket
'''

import boto3
import os

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### only run this code if you need to update from s3 bucket
def extract_from_s3(bucket_name, local_folder):
    ''' 
    Saves all json files from s3 bucket to a local folder
    --------
    PARAMETERS:
    bucket_name: str - s3 bucket where json is stored
    local_folder: location to save files locally
    --------
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
    Loads locally stored single line .json files into flattned dataframe
    --------
    PARAMETERS
    json_folder: local folder containing only json to analyze
    --------
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



def _add_list_contents_to_df(row, appendee):
    '''
    Appends rows in df to new columns
    --------
    PARAMETERS
    row: dataframe row containing list if items
    apandee: list to add items onto
    --------
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
    --------
    PARAMETERS
    df: pandas df - from create_flattened_dataframe function
    --------
    RETURNS
    review_df: pandas df - dataframe containing reviews
    '''
    reviews_1 = df['id']
    keys = df['reviews'].values[0][0].keys()
    review_rows = []
    df['reviews'].apply(_add_list_contents_to_df, appendee= review_rows)
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

def _filter_items(filt_list, cats_df):
    '''
    Generates a bool array for the requested catagories
    --------
    PARAMETERS
    filt_list: list of terms to appear in parent or child of catagories list
    cats_df: pandas dataframe - catagories mated with thie parents.
    --------
    RETURNS
    set - contains relevant catagories
    '''
    blank_bool = np.zeros((cats_df.shape[0],1), dtype=np.int64).flatten()
    for cat in filt_list:
        blank_bool += cats_df['parent'].str.contains(cat).fillna(False)
        blank_bool += cats_df['catagory'].str.contains(cat).fillna(False)
    return set(cats_df[blank_bool > 0]['catagory'].values)

def _apply_filter(df, cat_set, cat):
    '''
    Adds a col to df as new object with bools to represent the given catagory
    --------
    PARAMETERS
    df: pd.Dataframe -  contains information on city POI's
    cat_set: set - contains related catagories to the interest
    cat: string - name of interest catagory
    --------
    RETURNS
    bool_array
    pd.Dataframe - similar to input with added row.
    '''
    df_ = df.copy()
    df_[cat] = df_['category-0'].isin(cat_set).fillna(False)
    df_[cat] = df_[cat] + df_['category-1'].isin(cat_set).fillna(False)
    df_[cat] = df_[cat] + df_['category-2'].isin(cat_set).fillna(False)
    return df_

def remove_unwanted_POIs(df, city):
    '''
    Removes non food/coffee/bar/recreation points of interest from dataframe
    --------
    PARAMETERS
    df: panads df  - to be cleaned
    city: string - name of city
    --------
    RETURNS
    new_df: pandas df - with less rows of data
    '''
    #--------#
    #if on s3 use
    aws = boto3.resource('s3')
    catfile = aws.Object('wanderwell-ready', 'poi-catagories').get()['Body']
    # #if local use
    # catfile = 'data/categories.json'
    #--------#
    #### remove all entries which are not in the target city
    df_ = df[df['location.city'].str.lower() == city.lower()]
    #### create catagories table
    cat_df = pd.read_json(catfile)
    cat_key = cat_df.set_index('alias')['parents']
    cat_key_m = cat_key.apply(lambda x: pd.Series(x))
    cat_key_m = cat_key_m[0]
    cats_df = cat_key_m.reset_index()
    cats_df.columns = ['catagory', 'parent']
    cats_df.dropna(inplace=True)
    #### split catagories
    subcat = {}
    subcat['food'] = _filter_items(['food', 'restaurants'], cats_df)
    subcat['coffee'] = _filter_items(['coffee'], cats_df)
    subcat['nightlife'] = _filter_items(['beer', 'wine', 'cocktail',
                                        'bars', 'pubs', 'breweries'], cats_df)
    #### add bool variables for specific catagories
    bools = np.zeros((df_.shape[0],1), dtype=np.int64).flatten()
    for key, value in subcat.iteritems():
        df_ = _apply_filter(df_, value, key)
        bools += df_[key]
    df_ = df_[bools > 0]
    return df_

def create_photos_df(df):
    '''
    Creates simple photos df associating photos with a bus id
    --------
    PARAMETERS
    df: pandas df - from create_flattened_dataframe function
    --------
    RETURNS
    new_df: pandas df - dataframe containing reviews
    '''
    pass

def create_general_df(df):
    '''
    Create general df for buisness Identification and rating.
    --------
    PARAMETERS
    df: pandas df - from create_flattened_dataframe function
    --------
    RETURNS
    pandas df - dataframe containing reviews
    '''
    keep_cols = ['name', 'category-0', 'coordinates.latitude', 'coordinates.longitude',
                'is_claimed', 'location.zip_code', 'price', 'review_count']
    hours_cols = [col for col in df.columns if col.startswith('hours')]
    keep_cols += hours_cols
    df_ = df[keep_cols]
    df_.columns = ['id', 'category', 'lat', 'long', 'claimed', 'zip', 
                   'price', 'review_count'] + hours_cols
    df_['price'].fillna('K', inplace=True)
    df_['price'] = df_['price'].apply(lambda x: len(x) if x is not 'K' else 0)
    df_['price'].fillna(False, inplace=True)
    return df_

def _find_min_distance(df1, df2):
    '''
    Finds Min distance between df1 and df2 values
    --------
    PARAMETERS
    df1: pd.DataFrame - Index dataframe
    df2: pd.DataFrame - being compared
    --------
    RETURNS
    index of items which contains the 'dist' column
    '''
    ## assume city is not straddling the equator or PM
    df1_ = df1.abs()
    ## there were 7 NA values in portland, discarding for now, may return later
    df2_ = df1.dropna(axis=0)
    df2_ = df2_[['lat', 'long']].abs()
    min_lat = []
    min_long = []
    for i in df1_['lat']:
        min_lat.append((np.abs(df2_ - i)).min())
    for j in df1_['long']:
        min_long.append((np.abs(df2_ - i)).min())
    df1_['dist'] = np.sqrt((np.array(min_lat)**2 + np.array(min_long)**2))
    return df1_



def build_grid(city_df, point_spacing, max_distance):
    '''
    Builds grid of points geographically based on proximity to points
    --------
    PARAMETERS
    city_df: pandas df - after it has been cleaned by create_general_df
    point_spacing: float - how far points should be located away from each other
    max_distance: float - how far a point is allowed to be away from any POIS
    --------
    RETURNS
    city_grid: pandas df contining long, and lat columns
    '''
    pnorth = city_df['lat'].max() + max_distance
    psouth = city_df['lat'].min() - max_distance
    pwest = city_df['long'].max() + max_distance
    peast = city_df['long'].min() - max_distance
    lats = np.arange(psouth, pnorth, point_spacing).reshape(-1, 1)
    longs = np.arange(peast, pwest, point_spacing).reshape(-1, 1)
    dummy = np.ones((lats.size, 1))
    empty = np.arange(2).reshape(1, 2)
    for value in longs:
        addcol = dummy * value
        addrow = np.append(lats, addcol, axis=1)
        empty = np.append(empty, addrow, axis=0)
    grid = pd.DataFrame(empty[1:,:])
    grid.columns = ['lat', 'long']
    gridex = _find_min_distance(grid, city_df)
    grid = grid[gridex['dist'] < max_distance]
    return grid
    

    # city_grid = pd.DataFrame({'lat': np.arange(pbot, ptop, point_spacing)
    # })

    




def save_df_to_json(df, file_location):
    '''
    Stores pandas dataframe as single .json file
    --------
    PARAMETERS
    df: pandas dataframe 
    file_location: string - location of file to save df data
    --------
    RETURNS
    None
    '''
    df.to_json(file_location)

def save_file_to_s3(file_location, bucket_name, bucket_key):
    '''
    Stores pandas dataframe as single .json file in save_df_to_s3_json
    -------
    PARAMETERS
    file_location: string - location of file to upload to s3
    bucket_name: string - s3 bucket to store .json file
    bucket_key: string - desired file name on s3
    -------
    RETURNS
    None
    '''
    aws = boto3.resource('s3')
    ww_all = aws.Bucket(bucket_name)
    ww_all.upload_file(file_location, bucket_key)

