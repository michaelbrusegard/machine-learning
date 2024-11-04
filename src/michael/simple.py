import pandas as pd
import os

# Define the paths to the data files
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '../../original_data')
ais_test_path = os.path.join(data_dir, 'ais_test.csv')
ais_train_path = os.path.join(data_dir, 'ais_train.csv')

# Define the path for the submissions folder and create it if it doesn't exist
submissions_dir = os.path.join(current_dir, '../../submissions')
os.makedirs(submissions_dir, exist_ok=True)
submission_file_path = os.path.join(submissions_dir, 'submission.csv')

# Load the data
print('Loading data...')
ais_test = pd.read_csv(ais_test_path)
ais_train = pd.read_csv(ais_train_path, delimiter='|')

# Convert the 'time' columns to datetime
print('Converting time columns to datetime...')
ais_test['time'] = pd.to_datetime(ais_test['time'])
ais_train['time'] = pd.to_datetime(ais_train['time'])

# Initialize the result DataFrame
result = pd.DataFrame(columns=['ID', 'longitude_predicted', 'latitude_predicted'])

# Iterate over each row in ais_test
print('Processing ais_test data...')
for index, test_row in ais_test.iterrows():
    vessel_id = test_row['vesselId']

    # Filter ais_train for the same vesselId and get the most recent entry
    recent_entry = (
        ais_train[ais_train['vesselId'] == vessel_id]
        .sort_values(by='time', ascending=False)
        .head(1)
    )

    if not recent_entry.empty:
        longitude_predicted = recent_entry.iloc[0]['longitude']
        latitude_predicted = recent_entry.iloc[0]['latitude']
    else:
        longitude_predicted = 0.0
        latitude_predicted = 0.0

    # Append the result
    new_row = pd.DataFrame(
        [
            {
                'ID': test_row['ID'],
                'longitude_predicted': longitude_predicted,
                'latitude_predicted': latitude_predicted,
            }
        ]
    )

    if not new_row.isna().all(axis=None):
        result = pd.concat([result, new_row], ignore_index=True)

    print(f"Processed ID: {test_row['ID']}")

# Save the result to a CSV file
print('Saving results to submission.csv...')
result.to_csv(submission_file_path, index=False)
print('Done!')
