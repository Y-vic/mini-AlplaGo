import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from Network import Network
import pandas as pd


def train(model_save_path):
    data_path = './dataset'

    states = pd.read_csv(data_path + "/states.csv")
    values = pd.read_csv(data_path + "/values.csv")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    states = torch.tensor(data=states.values, device=device, dtype=torch.float32)
    values = torch.tensor(data=values.values, device=device, dtype=torch.float32)
    dataset = TensorDataset(states, values)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset=dataset, lengths=[train_size, test_size])

    network = Network(seed=326).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=network.parameters(), lr=0.01)

    batch_size = 24
    epochs = 10
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for _, (batch_states, batch_values) in enumerate(train_loader):
            prediction = network(batch_states)
            loss = criterion(prediction, batch_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

    torch.save(network.state_dict(), model_save_path)

    network.eval()
    loss_list = []

    with torch.no_grad():
        for batch_states, batch_values in test_loader:
            prediction = network(batch_states)
            loss = criterion(prediction, batch_values)
            loss_list.append(loss.item())
    print(f"accuracy:{100 - sum(loss_list) / len(loss_list)}")
