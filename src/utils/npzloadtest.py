import numpy as np

# Load the .npz file
file_path = '~/091809_w_43-BL1-097_bio.npz'
data = np.load(file_path)

# Check the keys in the .npz file
print("Keys in the .npz file:", data.keys())

# Extract and print the GSR signal and label
gsr_signal = data['x']
label = data['y']

print("GSR Signal:", gsr_signal)
print("Label:", label)
