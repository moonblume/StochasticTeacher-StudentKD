import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def extract_metrics_from_log(log_file):
    epochs = []
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    try:
        with open(log_file, 'r') as file:
            for line in file:
                epoch_match = re.search(r'Epoch\s+(\d+)/\d+', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    epochs.append(epoch)
                
                train_loss_match = re.search(r'Training Loss:\s+([\d\.]+)', line)
                if train_loss_match:
                    train_loss = float(train_loss_match.group(1))
                    train_losses.append(train_loss)
                
                train_accuracy_match = re.search(r'Training Accuracy:\s+([\d\.]+)', line)
                if train_accuracy_match:
                    train_accuracy = float(train_accuracy_match.group(1))
                    train_accuracies.append(train_accuracy)

                val_loss_match = re.search(r'Validation Loss:\s+([\d\.]+)', line)
                if val_loss_match:
                    val_loss = float(val_loss_match.group(1))
                    val_losses.append(val_loss)
                
                val_accuracy_match = re.search(r'Validation Accuracy:\s+([\d\.]+)', line)
                if val_accuracy_match:
                    val_accuracy = float(val_accuracy_match.group(1))
                    val_accuracies.append(val_accuracy)
    except Exception as e:
        print(f"Error processing log file {log_file}: {e}")
        return [], [], [], [], []

    # Ensure all lists have the same length
    min_length = min(len(epochs), len(train_losses), len(train_accuracies), len(val_losses), len(val_accuracies))
    if min_length == 0:
        print(f"No complete data found in log file {log_file}")
        return [], [], [], [], []

    return epochs[:min_length], train_losses[:min_length], train_accuracies[:min_length], val_losses[:min_length], val_accuracies[:min_length]

def plot_metrics(epochs, train_accuracies, val_accuracies, train_losses, val_losses, fold_index, output_directory):
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_accuracies, label='Training Accuracy')
        plt.plot(epochs, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Training and Validation Accuracy over Epochs - Fold {fold_index}')
        plt.legend()
        plt.grid(True)
        output_path_acc = os.path.join(output_directory, f'accuracy_fold_{fold_index}.png')
        plt.savefig(output_path_acc)
        plt.close()
        print(f'Saved accuracy plot for fold {fold_index} at {output_path_acc}')

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, label='Training Loss')
        plt.plot(epochs, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss over Epochs - Fold {fold_index}')
        plt.legend()
        plt.grid(True)
        output_path_loss = os.path.join(output_directory, f'loss_fold_{fold_index}.png')
        plt.savefig(output_path_loss)
        plt.close()
        print(f'Saved loss plot for fold {fold_index} at {output_path_loss}')
    except Exception as e:
        print(f"Error plotting metrics for fold {fold_index}: {e}")

def collect_and_plot_fold_data(log_dir, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f'Created directory {output_directory}')
    else:
        print(f'Using existing directory {output_directory}')

    fold_index = 1
    best_metrics = []

    try:
        for root, dirs, files in os.walk(log_dir):
            print(f"Scanning directory {root} for log files...")
            for file_name in files:
                if file_name.startswith('fold_') and file_name.endswith('_training.log'):
                    log_file = os.path.join(root, file_name)
                    print(f'Processing log file: {log_file}')
                    epochs, train_losses, train_accuracies, val_losses, val_accuracies = extract_metrics_from_log(log_file)
                    if not epochs or not train_losses or not train_accuracies or not val_losses or not val_accuracies:
                        print(f"Incomplete data in log file {log_file}. Skipping...")
                        continue

                    best_validation_accuracy = np.max(val_accuracies)
                    best_training_accuracy = np.max(train_accuracies)

                    best_metrics.append({
                        'Fold': fold_index,
                        'Best Validation Accuracy': best_validation_accuracy,
                        'Best Training Accuracy': best_training_accuracy
                    })

                    plot_metrics(epochs, train_accuracies, val_accuracies, train_losses, val_losses, fold_index, output_directory)
                    fold_index += 1
    except Exception as e:
        print(f"Error during processing: {e}")

    if best_metrics:
        try:
            df_best_metrics = pd.DataFrame(best_metrics)
            df_best_metrics.loc['Average'] = df_best_metrics.mean(numeric_only=True)
            csv_path = os.path.join(output_directory, 'best_metrics_per_fold.csv')
            excel_path = os.path.join(output_directory, 'best_metrics_per_fold.xlsx')
            df_best_metrics.to_csv(csv_path, index=False)
            df_best_metrics.to_excel(excel_path, index=False)
            print(f'Saved best metrics to CSV at {csv_path}')
            print(f'Saved best metrics to Excel at {excel_path}')
            print(df_best_metrics)
            print(f"Average Best Validation Accuracy: {df_best_metrics['Best Validation Accuracy'].mean()}")
            print(f"Average Best Training Accuracy: {df_best_metrics['Best Training Accuracy'].mean()}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
    else:
        print("No valid metrics collected. Check your log files for data completeness.")

log_directory = 'path to saved log files for each run'
output_directory = 'path to output directory for final plots and metrics' 
collect_and_plot_fold_data(log_directory, output_directory)
