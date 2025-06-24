import os
from utils import generate_kfolds_index, load_data, pain_labels

# Directory where .npz files are saved
npz_dir = 'npz directory'  # Change X to pain level of interest

# Number of folds for cross-validation
k_folds = 87

# Generate K-folds index
kfolds_index = generate_kfolds_index(npz_dir, k_folds)

# Use the first fold for demonstration
train_set, test_set = kfolds_index[0]

# DataLoader parameters
batch_size = 32
num_workers = 4  # Adjust based on your system's capacity

# Create DataLoader objects
train_loader, valid_loader, dist = load_data(train_set, test_set, pain_labels, batch_size, num_workers)

# Print out the distribution of classes
print("Distribution of classes in the dataset:", dist)
print("Number of training batches:", len(train_loader))
print("Number of validation batches:", len(valid_loader))
