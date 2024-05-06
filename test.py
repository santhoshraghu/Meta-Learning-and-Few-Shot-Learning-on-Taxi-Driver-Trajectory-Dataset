import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SiameseLSTM
from train import load_data, evaluate, SiameseDataset

def test_model():
    """
    Main function to initiate the model testing process.
    Includes loading test data, setting up the model and test loader,
    and executing the testing function.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test data
    pairs, labels = load_data('X_Y_test_pairs.pkl')

    # Create DataLoader for testing
    test_dataset = SiameseDataset(pairs, labels)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Set up model
    input_dim = pairs.shape[-1]
    hidden_dim = 64  # Adjust as needed
    model = SiameseLSTM(input_dim, hidden_dim).to(device)

    # Load trained model weights
    model.load_state_dict(torch.load('model.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Define criterion
    criterion = nn.BCEWithLogitsLoss()

    # Evaluate the model on the test set
    test_loss, test_acc = evaluate(model, criterion, test_loader)

    # Print test results
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

if __name__ == "__main__":
    test_model()
