import pandas as pd
import pandas.api.types
from geopy.distance import geodesic
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
current_dir = os.path.dirname(os.path.abspath(__file__))


class ParticipantVisibleError(Exception):
    def __init__(self, message):
        super().__init__(message)
        logging.error(message)


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')

    for col in ['ID', 'longitude_predicted', 'latitude_predicted']:
        if col not in submission.columns:
            raise ParticipantVisibleError(f'Submission is missing column {col}')

    try:
        solution_submission = solution.merge(
            submission[[row_id_column_name, 'longitude_predicted', 'latitude_predicted']],
            on=row_id_column_name,
            how='left',
        )
        solution_submission['weighted_distance'] = solution_submission.apply(
            calculate_distance, axis=1
        )
        weighted_distance = solution_submission['weighted_distance'].mean() / 1000.0
        return weighted_distance
    except (KeyError, ValueError) as e:
        raise ParticipantVisibleError(f'Evaluation metric raised an unexpected error: {e}')


def calculate_distance(row):
    if pd.isna(row['latitude']) or pd.isna(row['latitude_predicted']):
        return np.nan
    distance = geodesic(
        (row['latitude'], row['longitude']), (row['latitude_predicted'], row['longitude_predicted'])
    ).meters
    weighted_distance = distance * row['scaling_factor']
    return weighted_distance


submission = pd.read_csv(os.path.join(current_dir, '../../output/submission_m_0.csv'))
solution = pd.read_csv(os.path.join(current_dir, '../../original_data/ais/ais_test.csv'))

try:
    # result = score(solution, submission, 'ID')
    logging.info(f'Score: {result}')
except ParticipantVisibleError as e:
    logging.error(f'Error: {e}')
