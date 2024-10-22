import logging
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
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
    def __init__(self, features, labels=None, sequence_length=5):
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length + 1

    def __getitem__(self, idx):
        feature_seq = self.features.iloc[idx : idx + self.sequence_length].values.astype(float)
        if self.labels is not None:
            label_seq = self.labels.iloc[idx + self.sequence_length - 1].values.astype(float)
            return feature_seq, label_seq
        return feature_seq


logging.info('Creating datasets...')
features = features.reset_index(drop=True)
labels = labels.reset_index(drop=True)

train_dataset = AISDataset(features, labels)
test_features = test_data.drop(['latitude', 'longitude'], axis=1)
test_dataset = AISDataset(test_features)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=1024, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=1024, num_workers=16)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 128).to(device)
        c_0 = torch.zeros(2, x.size(0), 128).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out


logging.info('Creating model...')
model = Model(features.shape[1], 128, 2).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0

logging.info('Training started')

epoch = 0
while True:
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

    logging.info(f'Epoch {epoch+1} completed with training loss: {epoch_loss}')

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device).float()
            targets = targets.to(device).float()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    logging.info(f'Epoch {epoch+1} completed with validation loss: {val_loss}')
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(current_dir, './best_model.pth'))
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            logging.info('Early stopping triggered')
            break

    epoch += 1

model.load_state_dict(torch.load(os.path.join(current_dir, './best_model.pth')))

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

if len(predictions) != len(test_ids):
    logging.warning(
        f'Length mismatch: predictions ({len(predictions)}) vs test_ids ({len(test_ids)})'
    )
    min_length = min(len(predictions), len(test_ids))
    predictions = predictions[:min_length]
    test_ids = test_ids[:min_length]

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
