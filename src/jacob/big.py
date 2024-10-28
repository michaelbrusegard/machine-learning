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

MAP_LAND_PATH = os.path.join(current_dir, '../../original_data/ne_10m_land.zip') # Path to the land zip file
MAP_OCEAN_PATH = os.path.join(current_dir, '../../original_data/ne_10m_ocean.zip') # Path to the ocean zip file
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

baseDataset.info()

def haversine_np(lat1, lon1, lat2, lon2):
    """
    Vectorized Haversine formula to calculate distance between two lat/lon points in kilometers.
    """
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # Calculate distance
    distance = R * c
    return distance

def calculate_distance_to_port_optimized(df):
    """
    Calculate the distance from the vessel's current location to the port based on existing port latitude and longitude.
    Optimized for large datasets using the Haversine formula and handles invalid/missing port information.

    Parameters:
    - df (pd.DataFrame): The dataframe containing vessel data (latitude, longitude, portLatitude, portLongitude, portId).

    Returns:
    - df (pd.DataFrame): The modified dataframe with a new 'distance_to_port' feature.
    """
    # Create a mask for valid port entries (i.e., portId is not NaN and port coordinates are valid)
    valid_port_mask = df['portId'].notna() & df['portLatitude'].between(-90, 90) & df['portLongitude'].between(-180, 180)

    # Initialize the 'distance_to_port' column with NaN for all rows
    df['distance_to_port'] = np.nan

    # Apply Haversine formula on valid rows
    valid_rows = df[valid_port_mask]

    df.loc[valid_port_mask, 'distance_to_port'] = haversine_np(
        valid_rows['latitude'].values,
        valid_rows['longitude'].values,
        valid_rows['portLatitude'].values,
        valid_rows['portLongitude'].values
    )

    return df

baseDataset = calculate_distance_to_port_optimized(baseDataset)

def impute_vessel_status(df, distance_threshold=0.1):
    # Ensure vessel_status is a string
    df['vessel_status'] = df['vessel_status'].astype(str)

    # Step 1: Identify inconsistencies
    inconsistent_status = (
        ((df['vessel_status'] == 'moored') & (df['sog'] > 0.1)) |
        ((df['vessel_status'] == 'moored') & (df['near_land'] == 0)) |
        ((df['vessel_status'] == 'anchored') & (df['sog'] > 0.1)) |
        ((df['vessel_status'] == 'moving') & (df['sog'] == 0))
    )
    df.loc[inconsistent_status, 'vessel_status'] = 'unknown'

    # Step 2: Impute 'unknown' statuses
    mask_unknown = df['vessel_status'] == 'unknown'

    # Impute based on proximity and SOG
    df.loc[
        mask_unknown & (df['sog'] <= 0.1) & (
            (df['near_land']) | (df['distance_to_port'] <= distance_threshold)
        ),
        'vessel_status'
    ] = 'moored'

    df.loc[
        mask_unknown & (df['sog'] != 0),
        'vessel_status'
    ] = 'moving'

    return df

# Apply the function to your dataset
baseDataset = impute_vessel_status(baseDataset)


# We use expert knowledge to infer anchored on unknown vessels that have sog = 0
def classify_unknown_as_anchored(df, distance_threshold=10, movement_radius=5):
    """
    Classifies 'unknown' vessel statuses as 'anchored' based on vessel behavior.

    Parameters:
    - df (pd.DataFrame): DataFrame containing vessel data.
    - distance_threshold (float): Maximum distance to port (in km) to consider a vessel for anchoring.
    - movement_radius (float): Radius within which subsequent rows must be stationary to confirm anchoring.

    Returns:
    - pd.DataFrame: Updated DataFrame with reclassified vessel statuses.
    """
    # Step 1: Filter unknown statuses with sog = 0 and within specified distance to port
    unknown_near_port = df[(df['vessel_status'] == 'unknown') &
                           (df['sog'] == 0) &
                           (df['distance_to_port'] <= distance_threshold)]

    # Step 2: Iterate through filtered rows to check subsequent behavior
    for idx in unknown_near_port.index:
        current_row = df.loc[idx]

        # Check the next row (if available) for movement or mooring status
        if idx + 1 in df.index:
            next_row = df.loc[idx + 1]

            # Conditions for reclassifying as 'anchored'
            # - If the next row shows movement or a close subsequent stationary position
            if ((next_row['sog'] > 0) or  # Vessel starts moving
                (next_row['vessel_status'] == 'moored') or  # Vessel moors next
                ((next_row['sog'] == 0) and  # Still stationary
                 (next_row['distance_to_port'] <= movement_radius))):  # Within close range

                # Set current unknown status to 'anchored'
                df.at[idx, 'vessel_status'] = 'anchored'

    return df

baseDataset = classify_unknown_as_anchored(baseDataset)

def handle_sog_anomalies(df):
    df = df.copy()

    # Ensure 'sog' is a float
    df['sog'] = df['sog'].astype(float)

    # Calculate vessel-specific time_diff thresholds
    vessel_time_stats = df.groupby('vesselId')['time_diff'].agg(['mean', 'std']).reset_index()
    vessel_time_stats['threshold'] = vessel_time_stats['mean'] + vessel_time_stats['std']

    # Define the moving mask to filter for 'moving' or 'unknown' vessel status
    moving_mask = df['vessel_status'].isin(['moving', 'unknown'])

    # Compute vessel-specific 'sog' statistics for moving status
    vessel_sog_stats = df.loc[moving_mask].groupby('vesselId')['sog'].agg(['median', 'mean', 'std'])
    vessel_sog_stats = vessel_sog_stats.rename(columns={'median': 'sog_median', 'mean': 'sog_mean', 'std': 'sog_std'})

    # Merge thresholds and stats back into the main DataFrame
    df = df.merge(vessel_time_stats[['vesselId', 'threshold']], on='vesselId', how='left')
    df = df.merge(vessel_sog_stats, on='vesselId', how='left')

    # Step 1: Identify anomalies (sog over threshold)
    sog_upper_limit = 2* df['sog_mean'] + df['sog_std']
    df.loc[df['sog'] > sog_upper_limit, 'sog'] = np.nan  # Set anomalies to NaN

    # Step 2: Interpolate sog where time_diff is within threshold
    def interpolate_sog(group):
        threshold = group['threshold'].iloc[0]
        # Only interpolate where time_diff is within threshold
        mask = group['time_diff'] <= threshold
        group.loc[mask, 'sog'] = group.loc[mask, 'sog'].interpolate(method='linear', limit_direction='both')
        return group

    df = df.groupby('vesselId').apply(interpolate_sog).reset_index(drop=True)

    # Step 3: Backfill sog where appropriate
    def backfill_sog(group):
        threshold = group['threshold'].iloc[0]
        for idx in group.index:
            if pd.isnull(group.loc[idx, 'sog']):
                # Check if the next row has time_diff within threshold and status is 'moving'
                next_time_diff = group['time_diff'].shift(-1)
                next_status = group['vessel_status'].shift(-1)
                if (next_time_diff[idx] <= threshold) and (next_status[idx] == 'moving'):
                    group.loc[idx, 'sog'] = group['sog'].shift(-1)[idx]
        return group

    df = df.groupby('vesselId').apply(backfill_sog).reset_index(drop=True)

    # Clean up
    df.drop(columns=['threshold', 'sog_median', 'sog_mean', 'sog_std'], inplace=True)

    return df

baseDataset = handle_sog_anomalies(baseDataset)

def handle_cog_interpolation(df, cog_col='cog', vessel_status_col='vessel_status'):
    df = df.copy()

    # Ensure COG is a float type
    df[cog_col] = df[cog_col].astype(float)

    # Set COG to 0 for moored vessels (stationary), as course is not meaningful
    df.loc[df[vessel_status_col] == 'moored', cog_col] = 0

    # Calculate vessel-specific time_diff thresholds
    vessel_time_stats = df.groupby('vesselId')['time_diff'].agg(['mean', 'std']).reset_index()
    vessel_time_stats['threshold'] = vessel_time_stats['mean'] + vessel_time_stats['std']

    # Merge thresholds back into the main DataFrame
    df = df.merge(vessel_time_stats[['vesselId', 'threshold']], on='vesselId', how='left')

    # Step 1: Interpolate COG where time_diff is within threshold
    def interpolate_cog(group):
        threshold = group['threshold'].iloc[0]
        # Only interpolate where time_diff is within threshold
        mask = group['time_diff'] <= threshold
        group.loc[mask, cog_col] = group.loc[mask, cog_col].interpolate(method='linear', limit_direction='both')
        return group

    df = df.groupby('vesselId').apply(interpolate_cog).reset_index(drop=True)

    #Clean up
    df.drop(columns=['threshold'], inplace =True)

    return df

# Apply the function to your dataset
baseDataset = handle_cog_interpolation(baseDataset)

baseDataset.info()

def handle_heading_interpolation(df, heading_col='heading', vessel_status_col='vessel_status'):
    df = df.copy()

    # Ensure heading is a float type
    df[heading_col] = df[heading_col].astype(float)

    # Set heading to 0 for moored vessels (stationary), as direction is not meaningful
    df.loc[df[vessel_status_col] == 'moored', heading_col] = 0

    # Calculate vessel-specific time_diff thresholds
    vessel_time_stats = df.groupby('vesselId')['time_diff'].agg(['mean', 'std']).reset_index()
    vessel_time_stats['threshold'] = vessel_time_stats['mean'] + vessel_time_stats['std']

    # Merge thresholds back into the main DataFrame
    df = df.merge(vessel_time_stats[['vesselId', 'threshold']], on='vesselId', how='left')

    # Step 1: Interpolate heading where time_diff is within threshold
    def interpolate_heading(group):
        threshold = group['threshold'].iloc[0]
        # Only interpolate where time_diff is within threshold
        mask = group['time_diff'] <= threshold
        group.loc[mask, heading_col] = group.loc[mask, heading_col].interpolate(method='linear', limit_direction='both')
        return group

    df = df.groupby('vesselId').apply(interpolate_heading).reset_index(drop=True)

    #Clean up
    df.drop(columns=['threshold'], inplace =True)

    return df

baseDataset = handle_heading_interpolation(baseDataset)

baseDataset.info()

def handle_rot_for_moored_vessels(df):
    """
    Set ROT (Rate of Turn) to 0 for all moored vessels.

    Parameters:
    df (pd.DataFrame): The input dataframe containing vessel_behaviour and rot columns.
    rot_col (str): The name of the ROT column.
    behaviour_col (str): The name of the vessel behaviour column (e.g., 'moored', 'moving').

    Returns:
    pd.DataFrame: The dataframe with ROT set to 0 for moored vessels.
    """
    # Set ROT to 0 where vessel_behaviour is 'moored'
    df.loc[df['vessel_status'] == 'moored', 'rot'] = 0

    return df

# Apply the function to the dataframe
baseDataset = handle_rot_for_moored_vessels(baseDataset)

baseDataset.info()

# Sample 10% of the data for clustering
df_sample = baseDataset.sample(frac=0.1, random_state=42)
X_sample = df_sample[['latitude', 'longitude']].values

# Standardize and apply DBSCAN to the sample
scaler = StandardScaler()
X_scaled_sample = scaler.fit_transform(X_sample)
dbscan = DBSCAN(eps=0.2, min_samples=40)
dbscan.fit(X_scaled_sample)

# Get the cluster labels from DBSCAN
labels_sample = dbscan.labels_

# Check for noise (-1 indicates noise points)
n_noise = list(labels_sample).count(-1)
print(f'Number of noise points: {n_noise}')

# Analyze the resulting clusters
n_clusters = len(set(labels_sample)) - (1 if -1 in labels_sample else 0)
print(f'Number of clusters found: {n_clusters}')

def feature_engineering_clusters(df, sample_df, cluster_labels, cluster_centers, eps, batch_size=10000):
    """
    Performs clustering-based feature engineering, including time spent in cluster and cluster transitions.

    Parameters:
    - df (pd.DataFrame): The full dataframe with vessel data.
    - sample_df (pd.DataFrame): The sampled dataframe used for clustering.
    - cluster_labels (np.array): Array of cluster labels from DBSCAN for the sampled data.
    - cluster_centers (np.array): Array of cluster centroids from DBSCAN.
    - eps (float): The epsilon value used in DBSCAN for maximum distance between points to form a cluster.
    - batch_size (int): Number of points to process in each batch to avoid memory errors.

    Returns:
    - df (pd.DataFrame): The modified dataframe with new features.
    """

    # Step 1: Assign the cluster labels for the points in the sample
    df.loc[sample_df.index, 'cluster_label'] = cluster_labels

    # Step 2: Use KDTree for efficient nearest-neighbor search for non-sampled points
    tree = KDTree(cluster_centers)  # Build KDTree from cluster centroids

    # Step 3: For non-sampled points, assign them to the nearest cluster if within 'eps'
    non_sampled_mask = df['cluster_label'].isna()
    non_sampled_points = df.loc[non_sampled_mask, ['latitude', 'longitude']].values

    if non_sampled_points.shape[0] > 0:
        # Initialize arrays to store results
        nearest_centroids = np.full(non_sampled_points.shape[0], -1)
        nearest_distances = np.full(non_sampled_points.shape[0], np.inf)

        # Process in batches to avoid memory overload
        for i in range(0, non_sampled_points.shape[0], batch_size):
            batch = non_sampled_points[i:i+batch_size]

            # Query KDTree for nearest centroids and distances for the current batch
            batch_distances, batch_nearest_centroids = tree.query(batch)

            # Assign points to the nearest cluster if the distance is within 'eps'
            within_eps = batch_distances <= eps
            nearest_centroids[i:i+batch_size][within_eps] = batch_nearest_centroids[within_eps]
            nearest_distances[i:i+batch_size][within_eps] = batch_distances[within_eps]

        # Assign the nearest centroids back to the dataframe
        df.loc[non_sampled_mask, 'cluster_label'] = nearest_centroids

    return df


baseDataset = feature_engineering_clusters(baseDataset, df_sample, dbscan.labels_, dbscan.components_, eps=0.2)

baseDataset.info()

def create_lag_features(df, lag_columns, lags):
    """
    Create lag features for the specified columns and lag values.

    Parameters:
    - df (pd.DataFrame): The dataframe containing spatial and time-dependent data.
    - lag_columns (list of str): List of columns for which to create lag features.
    - lags (list of int): List of lag intervals to create.

    Returns:
    - df (pd.DataFrame): The dataframe with new lag features.
    """
    for col in lag_columns:
        for lag in lags:
            # Create lagged features, keeping NaNs for missing values
            df[f'lag_{lag}_{col}'] = df.groupby('vesselId')[col].shift(lag)

    return df

def create_time_dependent_spatial_features(df):
    """
    Create lag, distance, and bearing features based on spatial and movement-related data.

    Parameters:
    - df (pd.DataFrame): The dataframe containing time-dependent spatial data (latitude, longitude, cog, sog, heading).

    Returns:
    - df (pd.DataFrame): The dataframe with new spatial features.
    """
    # Ensure the 'time' column is sorted by vesselId and time
    df = df.sort_values(by=['vesselId', 'time'])

    # Lag Feature Creation
    df = create_lag_features(df, lag_columns=['latitude', 'longitude'], lags=[1, 2, 3, 4, 5])
    df = create_lag_features(df, lag_columns=['cog', 'sog', 'heading'], lags=[1, 2])

    # Calculate the distance between consecutive points using the Haversine formula
    def haversine_np(lat1, lon1, lat2, lon2):
        R = 6371.0  # Radius of the Earth in kilometers
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    df['distance_traveled'] = haversine_np(df['lag_1_latitude'], df['lag_1_longitude'], df['latitude'], df['longitude'])

    # Calculate Speed Difference (change in sog between consecutive points)
    df['speed_change'] = df['sog'] - df['lag_1_sog']

    # Calculate Bearing between consecutive points
    def calculate_bearing(lat1, lon1, lat2, lon2):
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)

        x = np.sin(dlon_rad) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
        initial_bearing = np.degrees(np.arctan2(x, y))
        return (initial_bearing + 360) % 360  # Normalize to [0, 360] degrees

    df['bearing'] = calculate_bearing(df['lag_1_latitude'], df['lag_1_longitude'], df['latitude'], df['longitude'])

    return df

baseDataset = create_time_dependent_spatial_features(baseDataset)


def create_rolling_features(df, columns, windows):
    """
    Create rolling statistics (mean and standard deviation) for specified columns over given time windows.

    Parameters:
    - df (pd.DataFrame): The dataframe containing spatial and time-dependent data.
    - columns (list of str): List of columns to calculate rolling statistics for.
    - windows (list of int): List of window sizes for the rolling statistics.

    Returns:
    - df (pd.DataFrame): The dataframe with new rolling statistics features.
    """

    # Iterate over each column and each window size to create rolling features
    for col in columns:
        for window in windows:
            # Create rolling mean feature
            df[f'{col}_rolling_mean_{window}'] = (
                df.groupby('vesselId')[col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            # Create rolling std feature
            df[f'{col}_rolling_std_{window}'] = (
                df.groupby('vesselId')[col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )

    return df

baseDataset = create_rolling_features(baseDataset, columns=['sog', 'cog', 'heading'], windows=[3, 5])

def extract_time_features(df):
    """
    Extract hour, day, and day of the week from a datetime column 'time'.

    Parameters:
    - df (pd.DataFrame): The dataframe containing a datetime 'time' column.

    Returns:
    - df (pd.DataFrame): The dataframe with new time-based features.
    """
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['day_of_week'] = df['time'].dt.dayofweek  # Monday=0, Sunday=6

    return df

baseDataset = extract_time_features(baseDataset)

def calculate_port_frequency(df):
    """
    Calculate how frequently each vessel visits a specific port based on portId.

    Parameters:
    - df (pd.DataFrame): The dataframe containing vessel data with portId.

    Returns:
    - df (pd.DataFrame): The modified dataframe with a new 'port_frequency' feature.
    """
    # Step 1: Group by vesselId and portId, and calculate the frequency of each port visit
    port_frequency = df.groupby(['vesselId', 'portId']).size().reset_index(name='port_frequency')

    # Step 2: Merge the port frequency back to the original dataframe
    df = df.merge(port_frequency, on=['vesselId', 'portId'], how='left')

    return df

# Calculate port frequency
baseDataset = calculate_port_frequency(baseDataset)

def add_previous_port_features_optimized(df):
    # Sort by vesselId and time to ensure sequential order within each vessel
    df = df.sort_values(by=['vesselId', 'time']).reset_index(drop=True)

    # Shift port info by one row to get the "previous" port details
    df['previous_portId'] = df.groupby('vesselId')['portId'].shift(1)
    df['previous_port_lat'] = df.groupby('vesselId')['portLatitude'].shift(1)
    df['previous_port_lon'] = df.groupby('vesselId')['portLongitude'].shift(1)

    # Identify rows where the portId has changed (within each vessel)
    change_mask = df.groupby('vesselId')['portId'].apply(lambda x: x != x.shift(1))

    # Only retain previous port info where there was a change in portId
    df['previous_portId'] = np.where(change_mask, df['previous_portId'], np.nan)
    df['previous_port_lat'] = np.where(change_mask, df['previous_port_lat'], np.nan)
    df['previous_port_lon'] = np.where(change_mask, df['previous_port_lon'], np.nan)

    # Forward fill previous port details across rows to maintain last known previous port until a new one is found
    df[['previous_portId', 'previous_port_lat', 'previous_port_lon']] = df.groupby('vesselId')[['previous_portId', 'previous_port_lat', 'previous_port_lon']].ffill()

    return df


baseDataset = add_previous_port_features_optimized(baseDataset)

baseDataset.info()


def convert_dataframe_types(df: pd.DataFrame, encode_categorical: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts DataFrame columns to appropriate data types for XGBoost and CatBoost models.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with original data types.
    - encode_categorical (bool): If True, perform label encoding on categorical features.

    Returns:
    - df_converted (pd.DataFrame): DataFrame with converted data types.
    - categorical_features (list): List of column names that are categorical features (empty if encoded).
    """
    df_converted = df.copy()

    # 1. Convert datetime columns to numerical features
    datetime_cols = df_converted.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    for col in datetime_cols:
        # Convert datetime to int
        df_converted[f'{col}'] = df_converted[col].astype(np.int64) // 10**9  # Convert to seconds

    # 2. Convert object columns to category dtype
    object_cols = df_converted.select_dtypes(include=['object']).columns.tolist()
    for col in object_cols:
        df_converted[col] = df_converted[col].astype('category')

    # 3. Ensure existing 'category' columns are correctly typed
    category_cols = df_converted.select_dtypes(include=['category']).columns.tolist()

    # 4. Convert boolean columns to integers
    bool_cols = df_converted.select_dtypes(include=['bool']).columns.tolist()
    for col in bool_cols:
        df_converted[col] = df_converted[col].astype(int)

    # 5. Handle categorical features
    if encode_categorical:
        # Perform label encoding
        for col in category_cols:
            le = LabelEncoder()
            df_converted[col] = le.fit_transform(df_converted[col].astype(str))
        categorical_features = []  # No categorical features after encoding
    else:
        categorical_features = category_cols.copy()

    # 6. Verify data types
    acceptable_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df_converted.columns:
        if df_converted[col].dtype.name not in acceptable_dtypes and col not in categorical_features:
            print(f"Warning: Column {col} has dtype {df_converted[col].dtype}, which may not be accepted by XGBoost.")

    return df_converted, categorical_features

X_train_converted, categorical_features = convert_dataframe_types(baseDataset, encode_categorical=True)
X_train_converted.info()

def calculate_weighted_geodesic_distance(row) -> float:
    """
    Calculates the weighted geodesic distance between actual and predicted lat/long points.

    Parameters:
    - row: A row from the merged DataFrame containing 'latitude', 'longitude', 'latitude_predicted',
           'longitude_predicted', and 'scaling_factor'.

    Returns:
    - Weighted geodesic distance in meters.
    """
    if pd.isna(row['latitude']) or pd.isna(row['latitude_predicted']):
        return np.nan

    # Calculate the geodesic distance in meters between actual and predicted coordinates
    distance = geodesic((row['latitude'], row['longitude']),
                        (row['latitude_predicted'], row['longitude_predicted'])).meters
    # Multiply by the scaling factor to get the weighted distance
    weighted_distance = distance * row['scaling_factor']
    return weighted_distance

def compute_score(solution: pd.DataFrame, prediction: pd.DataFrame) -> float:
    """
    Computes the mean weighted geodesic distance for model tuning.

    Parameters:
    - solution: DataFrame with columns ['vesselId', 'time', 'latitude', 'longitude', 'scaling_factor'] (ground truth).
    - prediction: DataFrame with columns ['vesselId', 'time', 'latitude_predicted', 'longitude_predicted'] (model predictions).

    Returns:
    - Mean weighted geodesic distance in kilometers.
    """
    # Merge solution and prediction DataFrames on 'vesselId' and 'time'
    merged = pd.merge(solution, prediction, on=['vesselId', 'time'], how='inner')

    # Apply the weighted distance calculation row by row
    merged['weighted_distance'] = merged.apply(calculate_weighted_geodesic_distance, axis=1)

    # Return the mean weighted distance in kilometers
    return merged['weighted_distance'].mean() / 1000.0  # Convert meters to kilometers

def create_time_based_folds(df: pd.DataFrame, num_folds: int = 3, days_per_fold: int = 5) -> List[Dict[str, pd.DataFrame]]:
    """
    Splits the data into sequential time-based folds for time-based cross-validation.

    Parameters:
    - df: DataFrame containing the full dataset.
    - num_folds: Number of folds for cross-validation.
    - days_per_fold: Number of days per validation set.

    Returns:
    - List of dictionaries with 'train', 'val', 'val_start_time', and 'val_end_time' for each fold.
    """
    # Ensure data is sorted by integer 'time'
    df = df.sort_values(by='time')
    folds = []

    # Compute the range of time in integer format for fold construction
    max_time = df['time'].max()
    fold_duration = days_per_fold * 24 * 3600  # days_per_fold converted to seconds

    # Generate folds by subtracting `fold_duration` for each new fold's validation start
    for i in range(num_folds):
        val_end_time = max_time - i * fold_duration
        val_start_time = val_end_time - fold_duration

        val_data = df[(df['time'] < val_end_time) & (df['time'] >= val_start_time)]
        train_data = df[df['time'] < val_start_time]

        if not train_data.empty and not val_data.empty:
            folds.append({
                'train': train_data,
                'val': val_data,
                'val_start_time': val_start_time,
                'val_end_time': val_end_time
            })
        else:
            print(f"Fold {i + 1} skipped due to empty train or validation set.")

    return folds

def recursive_prediction(
    model,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    static_features: list,
    time_dependent_features: list,
    lag_features: dict
) -> pd.DataFrame:
    """
    Perform recursive predictions on validation data using the trained model,
    updating lag features and other necessary features.

    Parameters:
    - model: Trained model for predictions
    - train_data: Training data DataFrame
    - val_data: Validation data DataFrame with 'vesselId' and 'time' columns, other features set to NaN
    - static_features: List of static feature names
    - time_dependent_features: List of time-dependent feature names set to NaN in validation
    - lag_features: Dict with keys 'lat_lon' (latitude/longitude lags) and 'other' (e.g., `cog`, `sog`, `heading` lags)

    Returns:
    - predictions: DataFrame with predicted 'latitude' and 'longitude' for the validation data
    """
    predictions = []
    val_data = val_data.sort_values(['vesselId', 'time']).reset_index(drop=True)

    # Define the full feature set for model training and prediction
    model_features = list(
        static_features +
        time_dependent_features +
        lag_features['lat_lon'] +
        lag_features['other'] +
        ["hour", "day", "day_of_week", "time_diff"]
    )

    for vessel_id in val_data['vesselId'].unique():
        val_vessel = val_data[val_data['vesselId'] == vessel_id].copy().reset_index(drop=True)
        train_vessel = train_data[train_data['vesselId'] == vessel_id].copy()
        if train_vessel.empty:
            continue

        # Cache static features
        static_values = train_vessel.iloc[-1][static_features].to_dict()

        # Initialize features DataFrame
        features_df = val_vessel[['time']].copy()

        # Convert 'time' to datetime within the function for feature calculations
        features_df['time_datetime'] = pd.to_datetime(features_df['time'], unit='s')
        train_vessel['time_datetime'] = pd.to_datetime(train_vessel['time'], unit='s')

        # Compute time features
        features_df['hour'] = features_df['time_datetime'].dt.hour
        features_df['day'] = features_df['time_datetime'].dt.day
        features_df['day_of_week'] = features_df['time_datetime'].dt.dayofweek

        # Compute time_diff
        # For the initial time, we need the last time from train_vessel
        last_train_time = train_vessel['time_datetime'].iloc[-1]
        times = pd.concat([pd.Series([last_train_time]), features_df['time_datetime']], ignore_index=True)
        features_df['time_diff'] = times.diff().dt.total_seconds().iloc[1:].values

        # Add static features
        for feature in static_features:
            features_df[feature] = static_values.get(feature, np.nan)

        # Initialize lag features with NaNs
        for feature in lag_features['lat_lon'] + lag_features['other']:
            features_df[feature] = np.nan

        # Set time-dependent features to NaN
        for feature in time_dependent_features:
            features_df[feature] = np.nan

        # Initialize lag deques with last known values from training data
        lat_lon_rolling = {
            feature: deque(train_vessel[feature].dropna().tail(5), maxlen=5)
            for feature in lag_features['lat_lon']
        }
        other_lags_rolling = {
            feature: deque(train_vessel[feature].dropna().tail(2), maxlen=2)
            for feature in lag_features['other']
        }

        # Process time steps recursively
        for idx in features_df.index:
            # Update lag features
            for feature in lag_features['lat_lon']:
                if len(lat_lon_rolling[feature]) > 0:
                    features_df.at[idx, feature] = lat_lon_rolling[feature][-1]
                else:
                    features_df.at[idx, feature] = np.nan

            for feature in lag_features['other']:
                if len(other_lags_rolling[feature]) > 0:
                    features_df.at[idx, feature] = other_lags_rolling[feature][-1]
                else:
                    features_df.at[idx, feature] = np.nan

            # Prepare features for prediction
            X_pred = features_df.loc[[idx], model_features]
            # Remove 'time_datetime' from features if present
            if 'time_datetime' in X_pred.columns:
                X_pred = X_pred.drop(columns=['time_datetime'])

            # Ensure that all features are acceptable to the model (e.g., numeric types)
            # Convert any datetime columns back to integer timestamps if necessary
            # For example, if 'time' is included as a feature, ensure it's integer
            if 'time' in X_pred.columns:
                X_pred['time'] = features_df.loc[idx, 'time']

            # Predict
            pred = model.predict(X_pred)
            pred_latitude, pred_longitude = pred[0]

            # Append predictions
            predictions.append({
                'vesselId': vessel_id,
                'time': features_df.at[idx, 'time'],  # Keep 'time' as integer
                'latitude_predicted': pred_latitude,
                'longitude_predicted': pred_longitude
            })

            # Update lag deques with predicted values
            for feature in lag_features['lat_lon']:
                if 'latitude' in feature:
                    lat_lon_rolling[feature].append(pred_latitude)
                elif 'longitude' in feature:
                    lat_lon_rolling[feature].append(pred_longitude)

            # Update other lag features if applicable
            # For this example, we assume they remain NaN or are not updated

    return pd.DataFrame(predictions)

def get_model(model_name: str, model_params: dict):
    if model_name == 'xgboost':
        from xgboost import XGBRegressor
        base_model = XGBRegressor(**model_params)
    elif model_name == 'lightgbm':
        from lightgbm import LGBMRegressor
        base_model = LGBMRegressor(**model_params)
    elif model_name == 'catboost':
        from catboost import CatBoostRegressor
        base_model = CatBoostRegressor(**model_params, verbose=False)
    else:
        raise ValueError("Invalid model name.")

    # Wrap the model with MultiOutputRegressor
    model = MultiOutputRegressor(base_model)
    return model

def cross_validate_model(
    df: pd.DataFrame,
    num_folds: int,
    days_per_fold: int,
    static_features: list,
    dynamic_features: list,
    lag_features: dict,
    model_name: str,
    model_params: dict,
    verbosity: int = 1
):
    """
    Cross-validate the model using time-based folds and recursive prediction.

    Parameters:
    - df: Full dataset DataFrame
    - num_folds: Number of folds
    - days_per_fold: Number of days per validation set
    - static_features, dynamic_features, lag_features: Lists of feature names
    - model_name: Name of the model to use ('xgboost', 'lightgbm', 'catboost')
    - model_params: Dictionary of hyperparameters for the model
    - verbosity: Verbosity level

    Returns:
    - cv_scores: List of scores for each fold
    - models: List of trained models for each fold
    """
    # Set GPU parameters based on the model
    if model_name == 'xgboost':
        model_params['tree_method'] = 'gpu_hist'
    elif model_name == 'lightgbm':
        model_params['device'] = 'gpu'
    elif model_name == 'catboost':
        model_params['task_type'] = 'GPU'

    # Create time-based folds
    folds = create_time_based_folds(df, num_folds, days_per_fold)
    cv_scores = []
    models = []

    for fold_idx, fold in enumerate(folds):
        if verbosity > 0:
            print(f"Processing Fold {fold_idx + 1}/{num_folds}")

        train_data = fold['train'].copy()
        val_data = fold['val'].copy()
        val_start_time = fold['val_start_time']

        # 'time' remains as integer 'time' in 'train_data' and 'val_data'

        # Prepare training features and target
        lag_feature_names = lag_features['lat_lon'] + lag_features['other']
        model_features = static_features + dynamic_features + lag_feature_names + ["hour", "day", "day_of_week", "time_diff"]
        X_train = train_data[model_features]
        y_train = train_data[['latitude', 'longitude']]

        # Prepare the model
        model = get_model(model_name, model_params)

        # Fit the model with early stopping
        model.fit(X_train, y_train,)

        # Prepare validation data for recursive prediction
        val_data_recursive = val_data[['vesselId', 'time']].copy()
        val_data_recursive = val_data_recursive.reindex(columns=['vesselId', 'time'] + model_features)
        val_data_recursive[model_features] = np.nan

        # Perform recursive prediction
        predictions = recursive_prediction(
            model,
            train_data,
            val_data_recursive,
            static_features,
            dynamic_features,
            lag_features
        )

        # Compute scaling factors
        val_start_datetime = pd.to_datetime(val_start_time, unit='s')
        val_data['time_datetime'] = pd.to_datetime(val_data['time'], unit='s')
        val_data['day_number'] = (val_data['time_datetime'] - val_start_datetime).dt.days + 1

        scaling_factors = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        val_data['scaling_factor'] = val_data['day_number'].map(scaling_factors)

        # Prepare solution and prediction DataFrames
        solution = val_data[['vesselId', 'time', 'latitude', 'longitude', 'scaling_factor']].reset_index(drop=True)
        prediction = predictions[['vesselId', 'time', 'latitude_predicted', 'longitude_predicted']].copy()
        prediction = prediction.reset_index(drop=True)

        # Compute score using your custom metric
        score = compute_score(
            solution=solution,
            prediction=prediction
        )

        if verbosity > 0:
            print(f"Fold {fold_idx + 1} Score: {score:.4f}")
        cv_scores.append(score)
        models.append(model)

    return cv_scores, models

def objective(trial, df, num_folds, days_per_fold, static_features, dynamic_features, lag_features, model_name, verbosity=1):
    # Suggest hyperparameters based on the model
    if model_name == 'xgboost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 3),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 3),
            'random_state': 42,
            'tree_method': 'hist',  # Faster tree construction
            'verbosity': 0
        }
    elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'num_leaves': trial.suggest_int('num_leaves', 15, 31),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 20),
                'random_state': 42
            }

    elif model_name == 'catboost':
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3, 8),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.5, 1.5),
            'random_seed': 42,
            'verbose': 0
        }

    else:
        raise ValueError("Invalid model name.")

    # Perform cross-validation
    cv_scores, models = cross_validate_model(
        df=df,
        num_folds=num_folds,
        days_per_fold=days_per_fold,
        static_features=static_features,
        dynamic_features=dynamic_features,
        lag_features=lag_features,
        model_name=model_name,
        model_params=params,
        verbosity=verbosity
    )

    # Return the average score over the folds
    avg_score = sum(cv_scores) / len(cv_scores)

    return avg_score

def hyperparameter_tuning(df, num_folds, days_per_fold, static_features, dynamic_features, lag_features, model_name, n_trials=50, verbosity=1):
    study = optuna.create_study(direction='minimize')

    # Define the objective function with partial to fix the additional parameters
    objective_func = partial(
        objective,
        df=df,
        num_folds=num_folds,
        days_per_fold=days_per_fold,
        static_features=static_features,
        dynamic_features=dynamic_features,
        lag_features=lag_features,
        model_name=model_name,
        verbosity=verbosity
    )

    study.optimize(objective_func, n_trials=n_trials)

    if verbosity > 0:
        print('Best trial score:', study.best_value)
        print('Best hyperparameters:', study.best_params)

    return study

# Define static features (these do not change over time)
static_features = [
    'shippingLineId', 'CEU',
    'DWT', 'GT', 'vesselType', 'breadth', 'homePort', 'length', 'age'
]

# Define dynamic features (exclude lags)
dynamic_features = [
    # List dynamic features here (e.g., 'cog', 'sog', 'rot', 'heading')
    'cog', 'sog', 'rot', 'heading', 'portId', 'portLatitude', 'portLongitude',
    'large_gap_flag', 'vessel_status', 'near_land', 'distance_to_port',
    'cluster_label', 'distance_traveled', 'speed_change', 'bearing',
    'sog_rolling_mean_3', 'sog_rolling_std_3', 'sog_rolling_mean_5', 'sog_rolling_std_5',
    'cog_rolling_mean_3', 'cog_rolling_std_3', 'cog_rolling_mean_5', 'cog_rolling_std_5',
    'heading_rolling_mean_3', 'heading_rolling_std_3', 'heading_rolling_mean_5', 'heading_rolling_std_5',
    'port_frequency', 'previous_portId', 'previous_port_lat', 'previous_port_lon'
]

# Define lag features
lag_features = {
    'lat_lon': [
        f'lag_{i}_latitude' for i in range(1, 6)
    ] + [
        f'lag_{i}_longitude' for i in range(1, 6)
    ],
    'other': [
        f'lag_{i}_cog' for i in range(1, 3)
    ] + [
        f'lag_{i}_sog' for i in range(1, 3)
    ] + [
        f'lag_{i}_heading' for i in range(1, 3)
    ]
}

df = X_train_converted

# Perform hyperparameter tuning for the desired model
model_name = 'xgboost'  # or 'lightgbm' or 'catboost'

study = hyperparameter_tuning(
    df,
    num_folds=4,
    days_per_fold=2,
    static_features=static_features,
    dynamic_features=dynamic_features,
    lag_features=lag_features,
    model_name=model_name,
    n_trials=50,  # Adjust n_trials based on your computational resources
    verbosity=1
)

# Retrieve the best hyperparameters
best_params = study.best_params

# Prepare the full training data (use the entire dataset)
full_train_data = df.copy()
full_train_data['time'] = full_train_data['time'].astype(int)

# Prepare training features and target
lag_feature_names = lag_features['lat_lon'] + lag_features['other']
model_features = static_features + dynamic_features + lag_feature_names
X_full_train = full_train_data[model_features]
y_full_train = full_train_data[['latitude', 'longitude']]

# Create the final model
final_model = get_model(model_name, best_params)

# Fit the final model
print("Training the final model on the full dataset...")
final_model.fit(X_full_train, y_full_train)
print("Final model training complete.")

# Load the test set
test_df = ais_test_df

# Sort test_df based on 'vesselId' and 'time'
test_df = test_df.sort_values(['vesselId', 'time']).reset_index(drop=True)

# Store the IDs and remove the 'ID' column
test_ids = test_df[['ID']].copy()
test_df = test_df.drop(columns=['ID'])

# Add the model features and set them to NaN
test_df = test_df.reindex(columns=['vesselId', 'time'] + model_features + ['latitude', 'longitude', 'time_diff', 'hour', 'day', 'day_of_week'])
test_df[model_features] = np.nan

# Fix category varible vesselId
le = LabelEncoder()
test_df['vesselId'] = le.fit_transform(test_df['vesselId'].astype(str))

# Step 1: Convert 'time' column from object to datetime
test_df['time'] = pd.to_datetime(test_df['time'], errors='coerce')
# Step 2: Convert datetime to Unix timestamp in seconds
test_df['time'] = test_df['time'].view(np.int64) // 10**9  # Unix timestamp in seconds
test_df = test_df[full_train_data.columns]


# Perform recursive prediction
print("Performing recursive prediction on the test set...")
test_predictions = recursive_prediction(
    final_model,
    full_train_data,
    test_df,
    static_features,
    dynamic_features,
    lag_features
)
print("Recursive prediction complete.")

# Merge the predictions with the IDs
test_predictions = test_predictions.sort_values(['vesselId', 'time']).reset_index(drop=True)
submission = pd.concat([test_ids, test_predictions[['latitude_predicted', 'longitude_predicted']]], axis=1)

# Rename columns as per Kaggle submission requirements
submission.rename(columns={'latitude_predicted': 'latitude', 'longitude_predicted': 'longitude'}, inplace=True)

# Sort the submission DataFrame based on 'ID'
submission = submission.sort_values('ID').reset_index(drop=True)

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created.")

# Print the best hyperparameters
print("Best hyperparameters found by Optuna:")
for param, value in best_params.items():
    print(f"{param}: {value}")

import json

with open("best_params.json", "w") as file:
    json.dump(best_params, file)

print("Best hyperparameters saved to best_params.json")
