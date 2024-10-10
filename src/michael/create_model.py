import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Starting script...')

print('Loading data...')
data = pd.read_csv('original_data/ais/ais_train.csv', delimiter='|')

print('Preprocessing data...')
data = data.fillna(data.mean())  # fill missing values with column mean

print('Splitting data into features and targets...')
X = data.drop(['longitude', 'latitude'], axis=1)
Y = data[['longitude', 'latitude']]

print('Normalizing features...')
scaler = StandardScaler()
X = scaler.fit_transform(X)

print('Converting to PyTorch tensors...')
X = torch.tensor(X, dtype=torch.float32).to(device)
Y = torch.tensor(Y.values, dtype=torch.float32).to(device)

print('Defining the model...')
model = nn.Linear(X.shape[1], 2).to(device)  # simple linear regression model

print('Defining the loss function and optimizer...')
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print('Training the model...')
for epoch in range(10):
    optimizer.zero_grad()
    predictions = model(X)
    loss = loss_fn(predictions, Y)
    loss.backward()
    optimizer.step()
    if epoch % 1 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

print('Making predictions on new data...')
new_data = pd.read_csv('original_data/ais/ais_test.csv', delimiter='|')
new_data = new_data.fillna(new_data.mean())
new_X = scaler.transform(new_data.drop(['longitude', 'latitude'], axis=1))
new_X = torch.tensor(new_X, dtype=torch.float32).to(device)
new_predictions = model(new_X)

print('Saving predictions...')
submission = pd.DataFrame(
    new_predictions.cpu().detach().numpy(), columns=['longitude_predicted', 'latitude_predicted']
)
submission.insert(0, 'ID', range(len(submission)))
submission.to_csv('submission.csv', index=False)

print('Script completed.')
