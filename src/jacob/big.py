import pandas as pd
import numpy as np
import geopandas as gpd

# Calculations
import math
from math import atan2, radians, degrees, sin, cos
from haversine import haversine, Unit
from geopy.distance import geodesic
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

# ML libraries
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from gluonts.torch.model.tft import TemporalFusionTransformerEstimator  # TFT from Torch
from gluonts.torch.model.deepar import DeepAREstimator  # DeepAR from Torch
from lightning.pytorch import Trainer  # PyTorch Lightning Trainer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
from typing import List, Dict


import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import torch
import torch.nn as nn
import torch.optim as optim  # For optimization

# AutoML libraries
from autogluon.tabular import TabularPredictor

# Other libraries
from typing import List, Dict, Tuple
from datetime import timedelta
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to your file in the bucket
file_path = os.path.join(current_dir, '../../original_data/ais_train.csv'),

# Load the file into a pandas dataframe
ais_train_df = pd.read_csv(file_path, delimiter= '|', encoding= 'utf-8')

# Display the dataframe
ais_train_df.info()

ais_test_df = pd.read_csv(os.path.join(current_dir, '../../original_data/ais_test.csv'))
ais_test_df.info()

# Define the path to your file in the bucket
file_path = os.path.join(current_dir, '../../original_data/ports.csv'),

#Load the file into a pandas dataframe
ports_df = pd.read_csv(file_path, delimiter= '|', encoding= 'utf-8')

ports_df.info()

# Define the path to your file in the bucket
file_path = os.path.join(current_dir, '../../original_data/vessels.csv'),

#Load the file into a pandas dataframe
vessels_df = pd.read_csv(file_path, delimiter= '|', encoding= 'utf-8')

vessels_df.info()

# Define the path to your file in the bucket
file_path = os.path.join(current_dir, '../../original_data/schedules_to_may_2024.csv'),

#Load the file into a pandas dataframe
schedules_df = pd.read_csv(file_path, delimiter= '|', encoding= 'utf-8')

def preprocess_ais_train(ais_train_df, ports_df):
    """
    Preprocess the ais_train_df by converting columns, handling missing or invalid values,
    merging port information, and mapping NAVSTAT codes to descriptions.

    Additionally, set 'etaRaw' to NaN if its value is less than the current time.

    Parameters:
    - ais_train_df: DataFrame containing the raw AIS train data.
    - ports_df: DataFrame containing port information with portId, latitude, and longitude.

    Returns:
    - ais_train_df_cleaned: A cleaned and preprocessed version of ais_train_df.
    """
    # Step 1: Convert 'time' to datetime and drop 'etaRaw'
    ais_train_df['time'] = pd.to_datetime(ais_train_df['time'], format='%Y-%m-%d %H:%M:%S')
    ais_train_df.drop('etaRaw', axis=1, inplace=True)

    # Step 4: Convert relevant columns to float
    ais_train_df['cog'] = ais_train_df['cog'].astype(float)
    ais_train_df['sog'] = ais_train_df['sog'].astype(float)
    ais_train_df['rot'] = ais_train_df['rot'].astype(float)
    ais_train_df['heading'] = ais_train_df['heading'].astype(float)
    ais_train_df['latitude'] = ais_train_df['latitude'].astype(float)
    ais_train_df['longitude'] = ais_train_df['longitude'].astype(float)

    # Step 5: Replace invalid or default values with NaN
    ais_train_df['cog'] = np.where((ais_train_df['cog'] == 360) | (ais_train_df['cog'] > 360) | (ais_train_df['cog'] < 0), np.nan, ais_train_df['cog'])
    ais_train_df['sog'] = np.where((ais_train_df['sog'] == 1023) | (ais_train_df['sog'] < 0), np.nan, ais_train_df['sog'])
    ais_train_df['rot'] = np.where((ais_train_df['rot'] == -128), np.nan, ais_train_df['rot'])
    ais_train_df['heading'] = np.where((ais_train_df['heading'] > 360) | (ais_train_df['heading'] == 511) | (ais_train_df['heading'] < 0), np.nan, ais_train_df['heading'])

    # Step 6: Merge with ports to get port latitude and longitude

    # Renaming the latitude and longitude columns in ports_df to portLatitude and portLongitude
    ports_df = ports_df.rename(columns={'latitude': 'portLatitude', 'longitude': 'portLongitude'})

    # Merging ais_train_df with the updated ports_df on 'portId'
    ais_train_df = ais_train_df.merge(ports_df[['portId', 'portLatitude', 'portLongitude']], on='portId', how='left')

    # Step 7: Sort by vesselId and time
    ais_train_df = ais_train_df.sort_values(by=['vesselId', 'time']).reset_index(drop=True)

    return ais_train_df

ais_train_df = preprocess_ais_train(ais_train_df, ports_df)

ais_train_df.info()

def preprocess_vessels(vessels_df):
    """
    Preprocess the vessels_df by converting 'yearBuilt' to 'age', handling missing values,
    mapping 'homePort', and converting 'shippingLineId' into a categorical feature.

    Parameters:
    - vessels_df: DataFrame containing the raw vessels data.

    Returns:
    - vessels_df_cleaned: A cleaned and preprocessed version of vessels_df.
    """

    # Step 1: Convert 'yearBuilt' to 'age'
    current_year = 2024
    vessels_df['age'] = vessels_df['yearBuilt'].apply(lambda x: current_year - x if pd.notna(x) else np.nan)
    vessels_df.drop(columns=['yearBuilt'], inplace=True)

    # Step 2: Drop columns with high missing values and low predictive power
    columns_to_drop = ['NT', 'depth', 'draft', 'freshWater', 'enginePower', 'fuel',
                       'maxHeight', 'maxWidth', 'rampCapacity', 'maxSpeed']
    vessels_df.drop(columns=columns_to_drop, inplace=True)

    # Step 3: Make vesselType into category
    # Convert 'vesselType' from float to categorical without knowing the exact mapping
    vessels_df['vesselType'] = vessels_df['vesselType'].astype('category')

    # Optionally, handle missing values (nan) by filling them with 'Unknown' or leaving them as is
    vessels_df['vesselType'] = vessels_df['vesselType'].cat.add_categories('Unknown').fillna('Unknown')

    return vessels_df


def map_homePort_to_country(vessels_df):
    """
    Maps 'homePort' city names to their respective countries and groups rare countries into 'OTHER'.
    """
    initial_mapping = {
        'PANAMA': 'Panama', 'UNKNOWN': 'Unknown', 'PALERMO': 'Italy', 'NASSAU': 'Bahamas',
        'TOKYO': 'Japan', 'VALLETTA': 'Malta', 'OSLO': 'Norway', 'MONROVIA': 'Liberia',
        'MAJURO': 'Marshall Islands', 'JEJU CHEJU': 'South Korea', 'HELSINKI': 'Finland',
        # Add other mappings here...
    }

    vessels_df['homePort'] = vessels_df['homePort'].map(initial_mapping).fillna('OTHER')

    # Group rare countries into 'OTHER' (with fewer than 10 occurrences)
    country_counts = vessels_df['homePort'].value_counts()
    rare_countries = country_counts[country_counts < 10].index.tolist()
    vessels_df['homePort'] = vessels_df['homePort'].replace(rare_countries, 'OTHER')

    return vessels_df


def process_shippingLineId(vessels_df):
    """
    Converts 'shippingLineId' into a categorical feature, grouping those with less than 13 occurrences
    into 'Unknown'.
    """
    # Calculate the frequency of each shippingLineId
    shipping_freq = vessels_df['shippingLineId'].value_counts()

    # Map rare shippingLineIds (occurring less than 13 times) to 'Unknown'
    vessels_df['shippingLineId'] = vessels_df['shippingLineId'].apply(
        lambda x: x if shipping_freq[x] >= 13 else 'Unknown')

    return vessels_df


# Main Preprocessing Pipeline
def preprocess_all(vessels_df):
    """
    Full preprocessing pipeline for vessels_df including handling shippingLineId, homePort,
    KNN imputation, and maxSpeed adjustments.
    """
    # Step 1: Basic preprocessing (convert yearBuilt to age and drop unnecessary columns)
    vessels_df = preprocess_vessels(vessels_df)

    # Step 2: Map 'homePort' to country
    vessels_df = map_homePort_to_country(vessels_df)

    # Step 3: Convert 'shippingLineId' to categorical with grouping of rare values
    vessels_df = process_shippingLineId(vessels_df)

    return vessels_df


# Apply the full pipeline
vessels_df = preprocess_all(vessels_df)
vessels_df.info()

def merge_ais_with_vessels(ais_data, vessel_data):
    """
    Merges the AIS data with the vessel data on 'vesselId' using a left join.

    Parameters:
    - ais_data (pd.DataFrame): DataFrame containing AIS data.
    - vessel_data (pd.DataFrame): DataFrame containing vessel data.

    Returns:
    - baseDataset (pd.DataFrame): Merged DataFrame containing the AIS data with vessel information.
    """
    # Perform a left merge on 'vesselId'
    baseDataset = pd.merge(ais_data, vessel_data, how='left', on='vesselId')

    return baseDataset

# Example usage:
baseDataset = merge_ais_with_vessels(ais_train_df, vessels_df)

baseDataset.info()
baseDataset.head()

# Get the unique vesselIds from the test set
vessel_ids_test = set(ais_test_df['vesselId'].unique())

# Get the count of records per vesselId in the training set
vessel_record_counts = ais_train_df['vesselId'].value_counts()

# Get the 10 vessels with the lowest number of records
lowest_record_vessels = vessel_record_counts.nsmallest(20)

# Check if these vessels are in the test set
vessels_in_test = lowest_record_vessels.index.isin(vessel_ids_test)

# Combine the results into a dataframe for easy viewing
vessels_with_low_records = pd.DataFrame({
    'vesselId': lowest_record_vessels.index,
    'record_count': lowest_record_vessels.values,
    'in_test_set': vessels_in_test
})

# Display the result
print(vessels_with_low_records)

# List of vessel IDs to remove
vessels_to_remove = ['61e9f3cbb937134a3c4bff09', '61e9f3adb937134a3c4bfe37']

# Remove vessels from the dataset
baseDataset = baseDataset[~ais_train_df['vesselId'].isin(vessels_to_remove)]

def add_time_diff_and_gap_flag(df, time_col='time', vessel_col='vesselId', threshold=2):
    """
    Adds time_diff and large_gap_flag to the dataframe based on vessel-specific statistics.

    Parameters:
    df (pd.DataFrame): The input dataframe containing vesselId and timestamp columns.
    time_col (str): The name of the timestamp column.
    vessel_col (str): The name of the vessel identifier column.
    threshold (float): The number of standard deviations above the mean to flag large time gaps.

    Returns:
    pd.DataFrame: The original dataframe with 'time_diff' and 'large_gap_flag' added.
    """
    # Ensure the time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Calculate time_diff (difference in time between consecutive entries for each vessel)
    df['time_diff'] = df.groupby(vessel_col)[time_col].diff().dt.total_seconds().fillna(0)

    # Group by vesselId to calculate the mean and standard deviation of time_diff for each vessel
    vessel_stats = df.groupby(vessel_col)['time_diff'].agg(['mean', 'std']).reset_index()

    # Merge vessel statistics back to the original dataframe
    df = df.merge(vessel_stats, on=vessel_col, how='left')

    # Define a large time gap as threshold * standard deviations above the mean for each vessel
    df['large_gap_flag'] = (df['time_diff'] > df['mean'] + threshold * df['std']).astype(int)

    # Drop the mean and std columns (optional, you can keep them if needed)
    df = df.drop(columns=['mean', 'std'])

    return df

baseDataset = add_time_diff_and_gap_flag(baseDataset, time_col='time', vessel_col='vesselId', threshold=3)
baseDataset.info()

def map_navstat_to_movement(df, navstat_col='navstat'):
    """
    Create a new feature indicating whether the vessel is moving, anchored, or moored
    based on the navstat column and then remove the original navstat column.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the navstat column.
    navstat_col (str): The name of the navstat column.

    Returns:
    pd.DataFrame: The dataframe with a new 'vessel_status' column and without the original 'navstat' column.
    """
    # Define the mapping of navstat codes to the movement categories
    navstat_mapping = {
        0: 'moving',      # Underway using engine
        1: 'anchored',    # Anchored
        2: 'anchored',      # Not under command
        3: 'moving',      # Restricted manoeuverability
        4: 'moving',      # Constrained by her draught
        5: 'moored',      # Moored
        6: 'anchored',    # Aground (considered stationary)
        7: 'moving',      # Engaged in fishing
        8: 'moving',      # Underway sailing
        9: 'unknown',      # Reserved for future use
        15: 'unknown'      # Undefined
    }

    # Create a new column based on the mapping
    df['vessel_status'] = df[navstat_col].map(navstat_mapping)

    # Remove the original navstat column
    df = df.drop(columns=[navstat_col])

    return df

#Apply the function to create the vessel_status feature and remove the navstat column
baseDataset = map_navstat_to_movement(baseDataset, navstat_col='navstat')

baseDataset.info()

MAP_LAND_PATH = "gs://jacobsbucketformlproject/ML Competition/ne_10m_land.zip" # Path to the land zip file
MAP_OCEAN_PATH = "gs://jacobsbucketformlproject/ML Competition/ne_10m_ocean.zip" # Path to the ocean zip file
land_world = gpd.read_file(MAP_LAND_PATH)
ocean_world = gpd.read_file(MAP_OCEAN_PATH)

def classify_near_land(data: pd.DataFrame):
    """
    Classify vessels as being near land by checking if they fall within land polygons.

    Parameters:
    - data (pd.DataFrame): The vessel data containing latitude and longitude.

    Returns:
    - pd.DataFrame: A DataFrame with a new 'near_land' feature.
    """
    # Create a GeoSeries of points from the vessel's longitude and latitude
    _data = data.copy()
    _data['geometry'] = gpd.points_from_xy(_data['longitude'], _data['latitude'], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(_data, geometry='geometry')

    # Perform spatial join to classify vessels as near land or not
    gdf_with_land = gpd.sjoin(gdf, land_world, how='left', predicate='within')

    # If a vessel was classified as being "on land", we set "near_land" to True
    gdf_with_land['near_land'] = gdf_with_land['index_right'].isna() == False

    return gdf_with_land

# Classify vessels as being near land or not
gdf_with_near_land = classify_near_land(baseDataset)

# Convert GeoDataFrame to a regular DataFrame (if needed) and drop geometry
baseDataset = pd.DataFrame(gdf_with_near_land.drop(columns=['geometry', 'index_right', 'featurecla', 'scalerank', 'min_zoom']))
