# main.py

import sys
from train import train_model
from test import test_model

def main():
    # Check if the user has provided an argument (train or test)
    if len(sys.argv) != 2:
        print("Usage: python main.py <mode>")
        print("<mode> should be 'train' or 'test'")
        sys.exit(1)  # Exit the program if no argument is provided

    # Fetch the mode from the command line argument
    mode = sys.argv[1]

    # Decide what to do based on the mode
    if mode == 'train':
        print("Starting training process...")
        train_model()  # Call the function to start training
    elif mode == 'test':
        print("Starting testing process...")
        test_model()  # Call the function to start testing
    else:
        print("Invalid mode. Choose 'train' or 'test'.")
        sys.exit(1)  # Exit if an invalid mode is provided

if __name__ == '__main__':
    main()
