import pandas as pd
import logging
import os
import time

script_start_time = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

logging.info('Loading data...')
ais_df = pd.read_csv(os.path.join(current_dir, '../../cleaned_data/ais/ais_train_m.csv'))
schedules_df = pd.read_csv(
    os.path.join(current_dir, '../../cleaned_data/ais/schedules_to_may_2024_m.csv')
)
ports_df = pd.read_csv(os.path.join(current_dir, '../../cleaned_data/ports/ports_m.csv'))
vessels_df = pd.read_csv(os.path.join(current_dir, '../../cleaned_data/vessels/vessels_m.csv'))

logging.info('Merging schedules and ports data...')
schedules_df = schedules_df.merge(ports_df, on='portId', how='left')

merged_dfs = []

logging.info('Processing vessel data...')
for vessel_id in ais_df['vesselId'].unique():
    ais_vessel_df = (
        ais_df[ais_df['vesselId'] == vessel_id].sort_values('time').reset_index(drop=True)
    )
    schedules_vessel_df = (
        schedules_df[schedules_df['vesselId'] == vessel_id]
        .sort_values('arrivalDate')
        .reset_index(drop=True)
    )

    if 'portId' in ais_vessel_df.columns:
        ais_vessel_df.rename(columns={'portId': 'ais_portId'}, inplace=True)

    schedule_cols = [
        'schedule_arrivalDate',
        'schedule_sailingDate',
        'schedule_moored_portId',
        'schedule_moored_portLatitude',
        'schedule_moored_portLongitude',
        'schedule_voyage_end',
        'schedule_destination_portId',
        'schedule_destination_portLatitude',
        'schedule_destination_portLongtitude',
        'schedule_small_movement_flag',
        'schedule_skipped_port_flag',
    ]

    if schedules_vessel_df.empty:
        for col in schedule_cols:
            ais_vessel_df[col] = pd.NA
        merged_vessel_df = ais_vessel_df
    else:
        events = []

        for idx in range(len(schedules_vessel_df) - 1):
            arrivalDate = schedules_vessel_df.loc[idx, 'arrivalDate']
            voyage_start = schedules_vessel_df.loc[idx, 'sailingDate']
            voyage_end = schedules_vessel_df.loc[idx + 1, 'arrivalDate']
            voyage_start_portId = schedules_vessel_df.loc[idx, 'portId']
            mooredPortLongitude = schedules_vessel_df.loc[idx, 'portLongitude']
            mooredPortLatitude = schedules_vessel_df.loc[idx, 'portLatitude']
            voyage_end_portId = schedules_vessel_df.loc[idx + 1, 'portId']
            destinationLatitude = schedules_vessel_df.loc[idx + 1, 'portLatitude']
            destinationLongitude = schedules_vessel_df.loc[idx + 1, 'portLongitude']
            small_movement_flag = schedules_vessel_df.loc[idx, 'small_movement_flag']
            skipped_port_flag = schedules_vessel_df.loc[idx, 'skipped_port_flag']
            event = {
                'start_time': arrivalDate,
                'end_time': voyage_end,
                'schedule_arrivalDate': arrivalDate,
                'schedule_sailingDate': voyage_start,
                'schedule_moored_portId': voyage_start_portId,
                'schedule_moored_portLatitude': mooredPortLatitude,
                'schedule_moored_portLongitude': mooredPortLongitude,
                'schedule_voyage_end': voyage_end,
                'schedule_destination_portId': voyage_end_portId,
                'schedule_destination_portLatitude': destinationLatitude,
                'schedule_destination_portLongtitude': destinationLongitude,
                'schedule_small_movement_flag': small_movement_flag,
                'schedule_skipped_port_flag': skipped_port_flag,
            }
            events.append(event)

        # Handle the last voyage after the last port stay
        last_idx = len(schedules_vessel_df) - 1
        arrivalDate = schedules_vessel_df.loc[last_idx, 'arrivalDate']
        voyage_start = schedules_vessel_df.loc[last_idx, 'sailingDate']
        voyage_start_portId = schedules_vessel_df.loc[last_idx, 'portId']
        mooredPortLongitude = schedules_vessel_df.loc[last_idx, 'portLongitude']
        mooredPortLatitude = schedules_vessel_df.loc[last_idx, 'portLatitude']
        small_movement_flag = schedules_vessel_df.loc[last_idx, 'small_movement_flag']
        skipped_port_flag = schedules_vessel_df.loc[last_idx, 'skipped_port_flag']
        voyage_end = pd.NaT  # No known destination time
        voyage_end_portId = pd.NA  # No known destination port
        destinationLatitude = pd.NA
        destinationLongitude = pd.NA
        event = {
            'start_time': arrivalDate,
            'end_time': voyage_start,
            'schedule_arrivalDate': arrivalDate,
            'schedule_sailingDate': voyage_start,
            'schedule_moored_portId': voyage_start_portId,
            'schedule_moored_portLatitude': mooredPortLatitude,
            'schedule_moored_portLongitude': mooredPortLongitude,
            'schedule_voyage_end': voyage_end,
            'schedule_destination_portId': voyage_end_portId,
            'schedule_destination_portLatitude': destinationLatitude,
            'schedule_destination_portLongtitude': destinationLongitude,
            'schedule_small_movement_flag': small_movement_flag,
            'schedule_skipped_port_flag': skipped_port_flag,
        }
        events.append(event)

        events_df = pd.DataFrame(events)

        data_points = []

        for idx, event in events_df.iterrows():
            start_time = event['start_time']
            end_time = event['end_time']

            ais_event_df = ais_vessel_df[
                (ais_vessel_df['time'] >= start_time) & (ais_vessel_df['time'] <= end_time)
            ].copy()

            if not ais_event_df.empty:
                for col in schedule_cols:
                    ais_event_df[col] = event.get(col, pd.NA)
                data_points.append(ais_event_df)
            else:
                synthetic_point = {col: pd.NA for col in ais_vessel_df.columns if col != 'vesselId'}
                synthetic_point['vesselId'] = vessel_id  # Ensure vesselId is set
                synthetic_point['time'] = pd.NaT  # Set time to NaT

                for col in schedule_cols:
                    synthetic_point[col] = event.get(col, pd.NA)

                synthetic_df = pd.DataFrame([synthetic_point])
                data_points.append(synthetic_df)

        event_intervals = [
            (event['start_time'], event['end_time']) for _, event in events_df.iterrows()
        ]

        def is_in_event(time):
            return any((time >= start) and (time <= end) for (start, end) in event_intervals)

        ais_outside_events_df = ais_vessel_df[~ais_vessel_df['time'].apply(is_in_event)].copy()
        if not ais_outside_events_df.empty:
            for col in schedule_cols:
                ais_outside_events_df[col] = pd.NA
            data_points.append(ais_outside_events_df)

        merged_vessel_df = pd.concat(data_points, ignore_index=True)

        merged_vessel_df['sort_time'] = merged_vessel_df.apply(
            lambda row: row['time']
            if pd.notnull(row['time'])
            else (
                row['schedule_arrivalDate']
                if pd.notnull(row['schedule_arrivalDate'])
                else pd.Timestamp.max
            ),
            axis=1,
        )

        merged_vessel_df.sort_values('sort_time', inplace=True)
        merged_vessel_df.drop(columns=['sort_time'], inplace=True)
        merged_vessel_df.reset_index(drop=True, inplace=True)

    merged_vessel_df = merged_vessel_df.merge(vessels_df, on='vesselId', how='left')

    merged_dfs.append(merged_vessel_df)

logging.info('Concatenating merged data...')
merged_df = pd.concat(merged_dfs, ignore_index=True)

logging.info('Dropping values to predict that are missing...')
merged_df = merged_df.dropna(subset=['latitude', 'longitude'])

logging.info('Filling missing values...')
for col in ['schedule_small_movement_flag', 'schedule_skipped_port_flag']:
    merged_df[col] = merged_df[col].fillna(0).astype(int)

numerical_columns = merged_df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_columns:
    merged_df[col] = merged_df[col].fillna(merged_df[col].median())

date_columns_train = [
    'time',
    'etaRaw',
    'schedule_arrivalDate',
    'schedule_sailingDate',
    'schedule_voyage_end',
]

for col in date_columns_train:
    if col in merged_df.columns:
        merged_df[col] = pd.to_datetime(merged_df[col], errors='coerce')

categorical_columns = merged_df.select_dtypes(include=['object']).columns
merged_df[categorical_columns] = merged_df[categorical_columns].fillna('Unknown')

logging.info('Dropping duplicates...')
merged_df = merged_df.drop_duplicates()

output_path = os.path.join(current_dir, '../../cleaned_data/merged_m.csv')
merged_df.to_csv(output_path, index=False, encoding='utf-8')
logging.info('Merged data saved to %s', output_path)

elapsed_time = time.time() - script_start_time
logging.info(f'Total time elapsed: {elapsed_time:.2f} seconds')
