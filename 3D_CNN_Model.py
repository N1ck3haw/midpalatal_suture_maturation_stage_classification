import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed

nii_path = "./nii_data"  # Directory containing .nii files
numpy_save_path = "./numpy_file_modified"  # Directory to save .npy files
os.makedirs(numpy_save_path, exist_ok=True)  # Create directory if it doesn't exist
target_size = (128, 128, 128)

# Define the processing function
def process_nii_file(file_path, save_path, target_size):
    try:
        # Read the NIfTI file and convert it to an array
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        
        # Normalize to [0, 1]
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
        
        # Resize to the target size
        image_resized = resize(image_array, target_size, mode='constant', preserve_range=True)
        
        # Add channel dimension
        image_resized = np.expand_dims(image_resized, axis=-1)
        
        # Save as a NumPy file
        np.save(save_path, image_resized)
        print(f"Saved {save_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Get all .nii files
nii_files = []
for root, _, files in os.walk(nii_path):
    for file_name in files:
        if file_name.endswith(".nii"):
            nii_files.append(os.path.join(root, file_name))

# Use ProcessPoolExecutor for parallel processing
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        futures = []
        for file_path in nii_files:
            file_name = os.path.basename(file_path)
            ct_number = file_name.split(".")[0]
            save_path = os.path.join(numpy_save_path, f"{ct_number}.npy")
            futures.append(executor.submit(process_nii_file, file_path, save_path, target_size))
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

print("All NIfTI files have been processed and saved as .npy.")

# Load tabular data
csv_path = "./data.csv"
df = pd.read_csv(csv_path)
df["CT"] = df["CT"].astype(int).astype(str)

# Encode labels
le = LabelEncoder()
df["Stage"] = le.fit_transform(df["Stage"])

# Load 3D image data
numpy_path = "./numpy_file_modified"
ct_to_image = {}

for file_name in os.listdir(numpy_path):
    if file_name.endswith(".npy"):
        ct_number = file_name.split(".")[0]
        file_path = os.path.join(numpy_path, file_name)
        ct_to_image[ct_number] = np.load(file_path)

# Match CT numbers and generate image data
images = []
valid_cts = []
for ct_number in df["CT"]:
    if ct_number in ct_to_image:
        images.append(ct_to_image[ct_number])
        valid_cts.append(ct_number)

print(f"Number of valid CTs: {len(valid_cts)}")
# Convert to NumPy array
images = np.array(images)

# Reshape image data to (num_samples, 1, 128, 128, 128)
images = images.squeeze(-1)
print(f"Images shape: {images.shape}")

# Filter tabular data to only include valid CT numbers
df_filtered = df[df["CT"].isin(valid_cts)]
labels = df_filtered["Stage"].values
table_features = df_filtered.drop(columns=["Name", "CT", "Stage"]).values
labels_filtered = df_filtered["Stage"].values

# Normalize tabular features
scaler = StandardScaler()
table_features = scaler.fit_transform(table_features)

# Split the dataset
X_image_train, X_image_test, X_table_train, X_table_test, y_train, y_test = train_test_split(
    images, table_features, labels_filtered, test_size=0.2, random_state=42
)

print(f"Image training shape: {X_image_train.shape}")
print(f"Table training shape: {X_table_train.shape}")

#--------output-----------
'''
Number of valid CTs: 600
Images shape: (600, 128, 128, 128)
Image training shape: (480, 128, 128, 128)
Table training shape: (480, 10)
'''

import torch
import torch.nn as nn
import torch.optim as optim

# Define a 3D CNN model
class CNN3DModel(nn.Module):
    def __init__(self):
        super(CNN3DModel, self).__init__()
        
        # 3D CNN for image data
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        
        # Fully connected layers for tabular data
        self.fc_table = nn.Sequential(
            nn.Linear(X_table_train.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fully connected layers for combined data
        self.fc_combined = nn.Sequential(
            nn.Linear(32 * 16 * 16 * 16 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, len(le.classes_))
        )
    
    def forward(self, image_input, table_input):
        # 3D CNN part
        x = self.pool1(torch.relu(self.conv1(image_input)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        
        # Tabular data part
        table_out = self.fc_table(table_input)
        
        # Concatenate the two parts
        combined = torch.cat((x, table_out), dim=1)
        output = self.fc_combined(combined)
        return output

# Initialize the model
model = CNN3DModel()


import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import DataParallel

# Ensure all GPU operations are completed
torch.cuda.synchronize()

# Free unused memory
torch.cuda.empty_cache()

# Convert data to Tensor
X_image_train_tensor = torch.tensor(X_image_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
X_image_test_tensor = torch.tensor(X_image_test, dtype=torch.float32).unsqueeze(1)
X_table_train_tensor = torch.tensor(X_table_train, dtype=torch.float32)
X_table_test_tensor = torch.tensor(X_table_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

import torch, gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DataParallel(model)
# Move the model to the device
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training process
num_epochs = 300
best_model_wts = model.state_dict()
best_acc = 0.0

# Define the model save path
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_image_train_tensor.to(device), X_table_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    _, preds = torch.max(outputs, 1)
    correct = (preds == y_train_tensor.to(device)).sum().item()
    accuracy = correct / len(y_train_tensor)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {accuracy}")

    # Save the best model
    if accuracy > best_acc:
        best_acc = accuracy
        best_model_wts = model.state_dict()
        torch.save(model.state_dict(), best_model_path)  # Save the best model

    # Stop training if accuracy reaches or exceeds 0.95
    if accuracy >= 0.95:
        print(f"Stopping training as accuracy has reached {accuracy:.4f} which is >= 0.95")
        break

# Load the best model
model.load_state_dict(torch.load(best_model_path))
print(f"Best model loaded with accuracy: {best_acc}")
#-------------output----------------
'''
Epoch 1/300, Loss: 1.6013580560684204, Accuracy: 0.17291666666666666
Epoch 2/300, Loss: 1.7547962665557861, Accuracy: 0.2625
Epoch 3/300, Loss: 1.5974678993225098, Accuracy: 0.2708333333333333
Epoch 4/300, Loss: 1.5400118827819824, Accuracy: 0.26458333333333334
Epoch 5/300, Loss: 1.5244977474212646, Accuracy: 0.28541666666666665
Epoch 6/300, Loss: 1.528157353401184, Accuracy: 0.26458333333333334
Epoch 7/300, Loss: 1.4987454414367676, Accuracy: 0.3145833333333333
Epoch 8/300, Loss: 1.5135542154312134, Accuracy: 0.31666666666666665
Epoch 9/300, Loss: 1.507057547569275, Accuracy: 0.2916666666666667
Epoch 10/300, Loss: 1.495538353919983, Accuracy: 0.33125
Epoch 11/300, Loss: 1.4742934703826904, Accuracy: 0.3229166666666667
Epoch 12/300, Loss: 1.4586741924285889, Accuracy: 0.3145833333333333
Epoch 13/300, Loss: 1.4363629817962646, Accuracy: 0.3375
Epoch 14/300, Loss: 1.4452825784683228, Accuracy: 0.3104166666666667
Epoch 15/300, Loss: 1.4165207147598267, Accuracy: 0.36666666666666664
Epoch 16/300, Loss: 1.3998959064483643, Accuracy: 0.34791666666666665
Epoch 17/300, Loss: 1.3787983655929565, Accuracy: 0.38958333333333334
Epoch 18/300, Loss: 1.3662188053131104, Accuracy: 0.3958333333333333
Epoch 19/300, Loss: 1.3420166969299316, Accuracy: 0.4270833333333333
Epoch 20/300, Loss: 1.3335487842559814, Accuracy: 0.4166666666666667
Epoch 21/300, Loss: 1.3259822130203247, Accuracy: 0.39791666666666664
Epoch 22/300, Loss: 1.3109616041183472, Accuracy: 0.39791666666666664
Epoch 23/300, Loss: 1.2905049324035645, Accuracy: 0.43125
Epoch 24/300, Loss: 1.2721184492111206, Accuracy: 0.40625
Epoch 25/300, Loss: 1.2967482805252075, Accuracy: 0.41041666666666665
...
Epoch 245/300, Loss: 0.16512267291545868, Accuracy: 0.93125
Epoch 246/300, Loss: 0.1437671184539795, Accuracy: 0.9541666666666667
Stopping training as accuracy has reached 0.9542 which is >= 0.95
Best model loaded with accuracy: 0.9541666666666667
'''

