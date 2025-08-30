import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from CNN.CNNModel import CNNModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


# Function to move data to CUDA if available
def to_cuda(tensor):
    return tensor.cuda() if torch.cuda.is_available() else tensor


# Function to calculate accuracy
def calculate_accuracy(y_true, y_pred):
    if y_true.dim() > 1 and y_true.size(1) > 1:
        y_true = torch.argmax(y_true, dim=1)

    predicted_classes = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_classes == y_true).float()
    return correct_predictions.sum() / len(correct_predictions)


# Function to plot accuracy graph
def plot_accuracy_graph(train_accuracies, val_accuracies, epochs):
    plt.plot(range(1, epochs + 1), train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(range(1, epochs + 1), val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Function to plot loss graph
def plot_loss_graph(train_losses, val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Load and preprocess data
data = pd.read_excel("../featureExtraction/alphabet_data.xlsx", header=0)
data.pop("CHARACTER")
group_value, coordinates = data.pop("GROUPVALUE"), data.copy()
coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
coordinates = torch.from_numpy(coordinates).float()
group_value = torch.from_numpy(group_value.to_numpy()).long()

k_folds = 4
epochs = 70

# Initialize variables to store fold-wise results
fold_train_losses, fold_val_losses = [], []
fold_train_accuracies, fold_val_accuracies = [], []
fold_train_precision, fold_val_precision = [], []
fold_train_recall, fold_val_recall = [], []
fold_train_f1, fold_val_f1 = [], []

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# K-fold cross-validation loop
for fold, (train_idx, val_idx) in enumerate(kf.split(coordinates)):
    print(f"Training on fold {fold + 1}/{k_folds}")

    # Split data for the current fold
    training, group_value_training = coordinates[train_idx], group_value[train_idx]
    validation, group_value_validation = coordinates[val_idx], group_value[val_idx]

    # Model and optimizer setup
    model = CNNModel()
    model = to_cuda(model)
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = CrossEntropyLoss()

    # Initialize lists for losses, accuracies, and metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_precisions, train_recalls, train_f1s = [], [], []
    val_precisions, val_recalls, val_f1s = [], [], []

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Move data to CUDA
        training, validation = to_cuda(training), to_cuda(validation)
        group_value_training, group_value_validation = to_cuda(group_value_training), to_cuda(group_value_validation)

        # Forward pass
        output_train = model(training)
        output_val = model(validation)

        # Compute loss
        loss_train = criterion(output_train, group_value_training)
        train_losses.append(loss_train.item())
        loss_val = criterion(output_val, group_value_validation)
        val_losses.append(loss_val.item())

        # Backpropagation
        loss_train.backward()
        optimizer.step()

        # Switch to evaluation mode for accuracy calculation
        with torch.no_grad():
            train_output = model(training).cpu()
            train_accuracy = calculate_accuracy(group_value_training, train_output)
            train_accuracies.append(train_accuracy.item())

            output_valid = model(validation).cpu()
            val_accuracy = calculate_accuracy(group_value_validation, output_valid)
            val_accuracies.append(val_accuracy.item())

        # Calculate metrics
        train_prec = precision_score(group_value_training.cpu(), torch.argmax(train_output, dim=1), average='weighted', zero_division=0)
        train_rec = recall_score(group_value_training.cpu(), torch.argmax(train_output, dim=1), average='weighted', zero_division=0)
        train_f1 = f1_score(group_value_training.cpu(), torch.argmax(train_output, dim=1), average='weighted', zero_division=0)

        val_prec = precision_score(group_value_validation.cpu(), torch.argmax(output_valid, dim=1), average='weighted', zero_division=0)
        val_rec = recall_score(group_value_validation.cpu(), torch.argmax(output_valid, dim=1), average='weighted', zero_division=0)
        val_f1 = f1_score(group_value_validation.cpu(), torch.argmax(output_valid, dim=1), average='weighted', zero_division=0)

        # Store metrics
        train_precisions.append(train_prec)
        train_recalls.append(train_rec)
        train_f1s.append(train_f1)
        val_precisions.append(val_prec)
        val_recalls.append(val_rec)
        val_f1s.append(val_f1)

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f'Fold {fold + 1}, Epoch {epoch}, Train Loss: {loss_train.item()}, Val Loss: {loss_val.item()}')

    # Store fold-wise results
    fold_train_losses.append(train_losses)
    fold_val_losses.append(val_losses)
    fold_train_accuracies.append(train_accuracies)
    fold_val_accuracies.append(val_accuracies)
    fold_train_precision.append(train_precisions)
    fold_train_recall.append(train_recalls)
    fold_train_f1.append(train_f1s)
    fold_val_precision.append(val_precisions)
    fold_val_recall.append(val_recalls)
    fold_val_f1.append(val_f1s)

# Average results across folds
avg_train_loss = np.mean(fold_train_losses, axis=0)
avg_val_loss = np.mean(fold_val_losses, axis=0)
avg_train_accuracy = np.mean(fold_train_accuracies, axis=0)
avg_val_accuracy = np.mean(fold_val_accuracies, axis=0)
avg_train_prec = np.mean(fold_train_precision, axis=0)
avg_train_rec = np.mean(fold_train_recall, axis=0)
avg_train_f1 = np.mean(fold_train_f1, axis=0)
avg_val_prec = np.mean(fold_val_precision, axis=0)
avg_val_rec = np.mean(fold_val_recall, axis=0)
avg_val_f1 = np.mean(fold_val_f1, axis=0)

# Plotting
plot_loss_graph(avg_train_loss, avg_val_loss)
plot_accuracy_graph(avg_train_accuracy, avg_val_accuracy, epochs)

# Final average performance
print(f"Final Average Training Accuracy: {avg_train_accuracy[-1] * 100:.2f}%")
print(f"Final Average Validation Accuracy: {avg_val_accuracy[-1] * 100:.2f}%")
print(f"Final Average Training Precision: {avg_train_prec[-1] * 100:.2f}%")
print(f"Final Average Validation Precision: {avg_val_prec[-1] * 100:.2f}%")
print(f"Final Average Training Recall: {avg_train_rec[-1] * 100:.2f}%")
print(f"Final Average Validation Recall: {avg_val_rec[-1] * 100:.2f}%")
print(f"Final Average Training F1-Score: {avg_train_f1[-1] * 100:.2f}%")
print(f"Final Average Validation F1-Score: {avg_val_f1[-1] * 100:.2f}%")

# Save the model
model_path = "CNN_model_alphabet_SIBI.pth"
torch.save(model.state_dict(), model_path)
