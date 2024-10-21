import logging
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time
import os

script_start_time = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info('Loading training data...')
train_data = pd.read_csv(os.path.join(current_dir, '../../cleaned_data/merged_m.csv'))

logging.info('Loading test data...')
test_data = pd.read_csv(os.path.join(current_dir, '../../original_data/ais/ais_test.csv'))

if 'ID' in test_data.columns:
    test_ids = test_data['ID']
else:
    raise KeyError("'ID' column not found in test_data.")

logging.info('Converting datetime columns to numbers...')
date_columns_train = [
    'time',
    'etaRaw',
    'schedule_arrivalDate',
    'schedule_sailingDate',
    'schedule_voyage_end',
]

for column in date_columns_train:
    train_data[column] = pd.to_datetime(train_data[column], errors='coerce')
    train_data[column] = pd.to_numeric(train_data[column], errors='coerce')

test_data['time'] = pd.to_datetime(test_data['time'])
test_data['time'] = pd.to_numeric(test_data['time'], errors='coerce')

logging.info('Converting id columns to codes...')
id_columns = [
    'ais_portId',
    'vesselId',
    'schedule_moored_portId',
    'schedule_destination_portId',
    'shippingLineId',
    'homePort',
]
for column in id_columns:
    train_data[column] = train_data[column].astype('category').cat.codes
    train_data[column] = train_data[column].astype('int64')

test_data['vesselId'] = test_data['vesselId'].astype('category').cat.codes
test_data['vesselId'] = test_data['vesselId'].astype('int64')

logging.info('Filling missing columns in test data with appropriate data types...')
missing_columns = set(train_data.columns) - set(test_data.columns)

for column in missing_columns:
    if train_data[column].dtype == 'float64':
        test_data[column] = 0.0
    elif train_data[column].dtype == 'int64':
        test_data[column] = 0

for column in train_data.columns:
    if column in test_data.columns:
        test_data[column] = test_data[column].astype(train_data[column].dtype)

test_data = test_data[train_data.columns]

logging.info('Dropping latitude and longitude from features...')
features = train_data.drop(['latitude', 'longitude'], axis=1)
labels = train_data[['latitude', 'longitude']]

logging.info('Scale appropriate columns...')
exclude_columns = [
    'navstat_emergency',
    'navstat_reserved',
    'navstat_special_maneuver',
    'navstat_stopped',
    'navstat_undefined',
    'navstat_under_way',
    'schedule_small_movement_flag',
    'schedule_skipped_port_flag',
]

for column in exclude_columns:
    features[column] = pd.to_numeric(features[column], errors='coerce')

scale_columns = [col for col in features.columns if col not in exclude_columns]
features_scale = features[scale_columns]

scaler = StandardScaler()
features_scale = scaler.fit_transform(features_scale)

features[scale_columns] = features_scale


class AISDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features.iloc[idx].values.astype(float), self.labels.iloc[
                idx
            ].values.astype(float)
        return self.features.iloc[idx].values.astype(float)


logging.info('Creating datasets...')
features = features.reset_index(drop=True)
labels = labels.reset_index(drop=True)

train_dataset = AISDataset(features, labels)
test_features = test_data.drop(['latitude', 'longitude'], axis=1)
test_dataset = AISDataset(test_features)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=16)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(features.shape[1], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.layer(x)


logging.info('Creating model...')
model = Model().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

early_stopping_patience = 10
best_loss = float('inf')
patience_counter = 0

logging.info('Training started')

for epoch in range(100):
    model.train()
    epoch_loss = 0
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    logging.info(f'Epoch {epoch+1} completed with loss: {epoch_loss}')

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            logging.info('Early stopping triggered')
            break

logging.info('Testing started')

model.eval()
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device).float()
        outputs = model(inputs)
        if i == 0:
            predictions = outputs
        else:
            predictions = torch.cat((predictions, outputs), 0)

predictions = predictions.cpu().numpy()
submission = pd.DataFrame(predictions, columns=['longitude_predicted', 'latitude_predicted'])

submission.insert(0, 'ID', test_ids.values)


def get_unique_filename(base_filename):
    counter = 0
    while True:
        filename = f'{base_filename}_{counter}.csv'
        if not os.path.isfile(filename):
            return filename
        counter += 1


base_filename = os.path.join(current_dir, '../../output/submission_m')
unique_filename = get_unique_filename(base_filename)

logging.info(f'Saving submission file to {unique_filename}')
submission.to_csv(unique_filename, index=False)

elapsed_time = time.time() - script_start_time
logging.info(f'Total time elapsed: {elapsed_time:.2f} seconds')
