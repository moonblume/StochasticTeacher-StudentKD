import os
import numpy as np
import pandas as pd

# Define paths
data_dir = 'path to dataset' # Change X to pain level of interest
output_dir = 'path to output' # Change X to pain level of interest
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file in the data directory
for csv_file in os.listdir(data_dir):
    # Check if it's a CSV file
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(data_dir, csv_file)
        
        # Read the CSV file
        df = pd.read_csv(csv_path, sep='\t')
                
        # Extract GSR values
        if 'gsr' in df.columns:
            gsr_signal = df['gsr'].values
            
            # Assuming pain_level is part of the filename or a constant value, update accordingly
            # Here, I'm assuming pain_level is constant. Update this if you have a way to extract it from filename.
            pain_level = 0  # Replace with actual pain level extraction logic if needed
            
            # Create an output file name
            base_name = os.path.basename(csv_file).replace('.csv', '.npz')
            output_file = os.path.join(output_dir, base_name)
            
            # Save as .npz file
            np.savez(output_file, x=gsr_signal, y=pain_level)
            print(f"Saved {output_file}")
        else:
            print(f"'gsr' column not found in {csv_file}. Skipping file.")

print("All data samples have been saved as .npz files.")

