'''
This script will clean and prepare data from json files to a format which can be
processed by the other files in this repository.  To run this file, the following
syntax must be used from the command line.  This file both saves those files
locally and updates the files saved on the wanderwell-ready s3 bucket.
--------
python extract_and_clean.py <city name>
--------
This script assumes you are running the file from it's home directory, and asssumes
you have allready collected files from the wanderwell-<city-name> s3 buckets.
'''
import pandas as pd
import numpy as np
import json
import sys
import build_df
city = sys.argv[1].lower()

### Build Clean DF
city_df = build_df.create_flattened_dataframe('../data/{}-json/'.format(city))
city_df_ = build_df.remove_unwanted_POIs(city_df, city)
city_df_clean = build_df.create_general_df(city_df_)
city_df_clean.to_json('../data/{}-clean.json'.format(city))
build_df.save_file_to_s3('../data/{}-clean.json'.format(city), 'wanderwell-ready')
### Build Comments DF
city_comments_df = build_df.create_reviews_df(city_df_)
city_comments_df.to_json('../data/{}-comments.json'.format(city))
build_df.save_file_to_s3('../data/{}-comments.json'.format(city), 'wanderwell-ready')
### Build BNB DF
city_bnb_df = build_df.create_bnb_df('../data/bnb-{}.json'.format(city), city)
city_bnb_df.to_json('../data/{}-bnb.json'.format(city))
build_df.save_file_to_s3('../data/{}-bnb.json'.format(city), 'wanderwell-ready')
## Build Grid
grid_build = pd.concat([city_df_clean[['lat', 'long']], city_bnb_df[['lat','long']]])
grid = build_df.build_grid(grid_build, .0027, .0009)
grid.to_json('../data/{}-grid.json.json'.format(city))
build_df.save_file_to_s3('../data/{}-grid.json'.format(city), 'wanderwell-ready')