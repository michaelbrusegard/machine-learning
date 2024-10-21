import pandas as pd
import numpy as np
import logging
import os
import time

script_start_time = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

try:
    logging.info('Reading the AIS data from ais_train.csv...')
    ais_df = pd.read_csv(
        os.path.join(current_dir, '../../original_data/ais/ais_train.csv'),
        delimiter='|',
    )

    logging.info('Converting time columns to datetime...')
    ais_df['time'] = pd.to_datetime(ais_df['time'], format='%Y-%m-%d %H:%M:%S')
    ais_df['etaRaw'] = pd.to_datetime(ais_df['etaRaw'], errors='coerce', format='%m-%d %H:%M')

    logging.info('Replacing year in etaRaw...')
    ais_df['etaRaw'] = ais_df['etaRaw'].apply(
        lambda x: x.replace(year=2024) if pd.notna(x) else pd.NaT
    )

    logging.info('Converting columns to float...')
    for col in ['cog', 'sog', 'rot', 'heading', 'latitude', 'longitude']:
        ais_df[col] = ais_df[col].astype(float)

    logging.info('Replacing invalid values...')
    ais_df['cog'] = np.where((ais_df['cog'] == 360) | (ais_df['cog'] > 360), np.nan, ais_df['cog'])
    ais_df['sog'] = np.where((ais_df['sog'] == 1023), np.nan, ais_df['sog'])
    ais_df['rot'] = np.where((ais_df['rot'] == -128), np.nan, ais_df['rot'])
    ais_df['heading'] = np.where(
        (ais_df['heading'] > 360) | (ais_df['heading'] == 511), np.nan, ais_df['heading']
    )

    logging.info('Grouping navstat values...')

    def group_navstat(navstat):
        if navstat in [0, 8]:
            return 'under_way'
        elif navstat in [1, 5, 6]:
            return 'stopped'
        elif navstat in [2, 3, 4, 7]:
            return 'special_maneuver'
        elif navstat in [9, 10, 11, 12, 13]:
            return 'reserved'
        elif navstat == 14:
            return 'emergency'
        else:
            return 'undefined'

    ais_df['navstat'] = ais_df['navstat'].apply(group_navstat)
    ais_df = pd.get_dummies(ais_df, columns=['navstat'])
    for column in ais_df.columns:
        if 'navstat' in column:
            ais_df[column] = ais_df[column].fillna(0).astype(int)

    logging.info('Sorting values...')
    ais_df = ais_df.sort_values(by=['vesselId', 'time']).reset_index(drop=True)

    logging.info('Writing cleaned AIS data to ais_train_m.csv...')
    ais_df.to_csv(os.path.join(current_dir, '../../cleaned_data/ais/ais_train_m.csv'), index=False)

    logging.info('Cleaned AIS data shape: %s', ais_df.shape)
    logging.info('Cleaned AIS data columns: %s', ais_df.columns.tolist())
    logging.info('Number of missing values in cleaned AIS data: %s', ais_df.isnull().sum())
    logging.info('AIS data cleaning completed successfully.')
except Exception as e:
    logging.error('An error occurred during the AIS data cleaning process: %s', e)

try:
    logging.info('Reading the 2024 schedules data from schedules_to_may_2024.csv...')
    schedules_df = pd.read_csv(
        os.path.join(current_dir, '../../original_data/ais/schedules_to_may_2024.csv'),
        delimiter='|',
    )

    logging.info('Dropping duplicates...')
    schedules_df.drop_duplicates(inplace=True)

    logging.info('Converting date columns to datetime...')
    schedules_df['arrivalDate'] = pd.to_datetime(schedules_df['arrivalDate'], errors='coerce')
    schedules_df['sailingDate'] = pd.to_datetime(schedules_df['sailingDate'], errors='coerce')

    logging.info('Localizing sailingDate...')
    schedules_df['arrivalDate'] = schedules_df['arrivalDate'].dt.tz_localize(None)
    schedules_df['sailingDate'] = schedules_df['sailingDate'].dt.tz_localize(None)

    logging.info('Dropping rows with missing vesselId or portId...')
    schedules_df.dropna(subset=['vesselId', 'portId'], inplace=True)

    logging.info('Dropping unnecessary columns...')
    columns_to_drop = ['portName', 'shippingLineId', 'shippingLineName']
    schedules_df.drop(columns=columns_to_drop, inplace=True)

    logging.info('Keeping rows where sailingDate is after arrivalDate...')
    schedules_df = schedules_df[schedules_df['sailingDate'] >= schedules_df['arrivalDate']]

    logging.info('Dropping duplicates based on vesselId, arrivalDate, sailingDate, and portId...')
    schedules_df.drop_duplicates(
        subset=['vesselId', 'arrivalDate', 'sailingDate', 'portId'], inplace=True
    )

    logging.info('Sorting values...')
    schedules_df.sort_values(by=['vesselId', 'arrivalDate'], inplace=True)
    schedules_df.reset_index(drop=True, inplace=True)

    logging.info(
        'Preprocessing schedules to handle infeasible arrival dates and small movements at the same port...'
    )
    schedules_df['small_movement_flag'] = 0
    schedules_df['skipped_port_flag'] = 0

    rows_to_drop = set()

    for vessel_id in schedules_df['vesselId'].unique():
        vessel_indices = schedules_df[schedules_df['vesselId'] == vessel_id].index.tolist()
        i = 0
        while i < len(vessel_indices) - 1:
            current_idx = vessel_indices[i]
            next_idx = vessel_indices[i + 1]
            current_row = schedules_df.loc[current_idx]
            next_row = schedules_df.loc[next_idx]

            if current_row['portId'] == next_row['portId']:
                schedules_df.at[current_idx, 'sailingDate'] = next_row['sailingDate']
                schedules_df.at[current_idx, 'small_movement_flag'] = 1
                rows_to_drop.add(next_idx)
                vessel_indices.pop(i + 1)
                continue

            elif next_row['arrivalDate'] <= current_row['sailingDate']:
                rows_to_drop.add(next_idx)
                schedules_df.at[current_idx, 'skipped_port_flag'] = 1
                vessel_indices.pop(i + 1)
                continue

            else:
                i += 1

    logging.info('Removing %s rows marked for dropping...', len(rows_to_drop))
    schedules_df = schedules_df.drop(index=list(rows_to_drop)).reset_index(drop=True)

    logging.info('Writing cleaned schedules data to schedules_to_may_2024_m.csv...')
    schedules_df.to_csv(
        os.path.join(current_dir, '../../cleaned_data/ais/schedules_to_may_2024_m.csv'), index=False
    )

    logging.info('Cleaned schedules data shape: %s', schedules_df.shape)
    logging.info('Cleaned schedules data columns: %s', schedules_df.columns.tolist())
    logging.info(
        'Number of missing values in cleaned schedules data: %s', schedules_df.isnull().sum()
    )
    logging.info('2024 schedules data cleaning completed successfully.')
except Exception as e:
    logging.error('An error occurred during the schedules data cleaning process: %s', e)

try:
    logging.info('Reading the port data from ports.csv...')
    ports_df = pd.read_csv(
        os.path.join(current_dir, '../../original_data/ports/ports.csv'),
        delimiter='|',
    )

    logging.info('Dropping unnecessary columns...')
    columns_to_drop = ['name', 'portLocation', 'UN_LOCODE', 'countryName', 'ISO']
    ports_df = ports_df.drop(columns=columns_to_drop)

    logging.info('Checking for missing values...')
    if ports_df.isnull().values.any():
        logging.info('Dropping rows with missing values...')
        ports_df = ports_df.dropna()

    logging.info('Writing cleaned port data to ports_m.csv...')
    ports_df.to_csv(os.path.join(current_dir, '../../cleaned_data/ports/ports_m.csv'), index=False)

    logging.info('Cleaned ports data shape: %s', ports_df.shape)
    logging.info('Cleaned ports data columns: %s', ports_df.columns.tolist())
    logging.info('Number of missing values in cleaned ports data: %s', ports_df.isnull().sum())
    logging.info('Port data cleaning completed successfully.')
except Exception as e:
    logging.error('An error occurred during the ports data cleaning process: %s', e)

try:
    logging.info('Reading the vessel data from vessels.csv...')
    vessels_df = pd.read_csv(
        os.path.join(current_dir, '../../original_data/vessels/vessels.csv'),
        delimiter='|',
    )

    logging.info('Calculating age and dropping yearBuilt column...')
    current_year = 2024
    vessels_df['age'] = vessels_df['yearBuilt'].apply(
        lambda x: current_year - x if pd.notna(x) else np.nan
    )
    vessels_df = vessels_df.drop(columns=['yearBuilt'])

    logging.info('Dropping unnecessary columns...')
    columns_to_drop = [
        'NT',
        'depth',
        'draft',
        'freshWater',
        'fuel',
        'maxHeight',
        'maxWidth',
        'rampCapacity',
    ]
    vessels_df = vessels_df.drop(columns=columns_to_drop)

    logging.info('Writing cleaned vessel data to vessels_m.csv...')
    vessels_df.to_csv(
        os.path.join(current_dir, '../../cleaned_data/vessels/vessels_m.csv'), index=False
    )

    logging.info('Vessel data cleaning completed successfully.')
except Exception as e:
    logging.error('An error occurred during the vessels data cleaning process: %s', e)

elapsed_time = time.time() - script_start_time
logging.info(f'Total time elapsed: {elapsed_time:.2f} seconds')
