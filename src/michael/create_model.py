import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time
import os
import logging

script_start_time = time.time()
current_dir = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.info('Loading training data...')
train_data = pd.read_csv(os.path.join(current_dir, '../../original_data/ais/ais_train.csv'))

logging.info('Loading test data...')
test_data = pd.read_csv(os.path.join(current_dir, '../../original_data/ais/ais_test.csv'))

features = train_data.drop(['latitude', 'longitude'], axis=1)
labels = train_data[['latitude', 'longitude']]

scaler = StandardScaler()
features = scaler.fit_transform(features)


class AISDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


train_dataset = AISDataset(features, labels)
test_dataset = AISDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = nn.Sequential(nn.Linear(features.shape[1], 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        return self.layer(x)


model = Model().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

logging.info('Training started')

for epoch in range(100):  # 100 epochs
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device).float()
        targets = targets.to(device).float()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logging.info(f'Epoch {epoch+1} completed')

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
submission.insert(0, 'ID', test_data['ID'])


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
