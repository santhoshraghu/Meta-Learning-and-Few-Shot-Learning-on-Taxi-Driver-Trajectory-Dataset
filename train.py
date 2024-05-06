import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import SiameseLSTM  # Assuming your model is defined in a file named model.py
import numpy as np
import pickle
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.pairs[idx]
        y = self.labels[idx]
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def load_data(filename='X_Y_train400_pairs.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    pairs = data['pairs']
    labels = data['labels']
    return pairs, labels

def train(model, optimizer, criterion, train_loader):
    """
    Function to handle the training of the model.
    Iterates over the training dataset and updates model parameters.
    """
    model.train()  # Set model to training mode
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate loss
        loss = criterion(outputs.squeeze(), targets)
        train_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Compute accuracy
        predicted = torch.round(torch.sigmoid(outputs)).squeeze()
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    train_loss /= len(train_loader)
    train_acc = correct / total

    return train_loss, train_acc

def evaluate(model, criterion, loader):
    """
    Function to evaluate the model performance on the validation set.
    Computes loss and accuracy without updating model parameters.
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs.squeeze(), targets)
            test_loss += loss.item()

            # Compute accuracy
            predicted = torch.round(torch.sigmoid(outputs)).squeeze()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    test_loss /= len(loader)
    test_acc = correct / total

    return test_loss, test_acc

def train_model():
    """
    Main function to initiate the model training process.
    Includes loading data, setting up the model, optimizer, and criterion,
    and executing the training and validation loops.
    """
    # Loading data
    pairs, labels = load_data()
    X_train, X_val, y_train, y_val = train_test_split(pairs, labels, test_size=0.2, random_state=42)

    # DataLoader for training and validation
    train_dataset = SiameseDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = SiameseDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64)

    input_dim = X_train.shape[-1]
    hidden_dim = 128 
    dropout_rate = 0.2 
    model = SiameseLSTM(input_dim, hidden_dim, dropout=dropout_rate).to(device)

    # optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Train the model
        train_loss, train_acc = train(model, optimizer, criterion, train_loader)
        
        # Evaluation
        val_loss, val_acc = evaluate(model, criterion, val_loader)

        # Printing epoch-wise training and validation results
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save trained model
    model_path = "model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved at: {os.path.abspath(model_path)}")

if __name__ == "__main__":
    train_model()
