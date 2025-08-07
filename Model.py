import torch
import torch.nn as nn
import torch.optim as optim

# Define the 3D CNN model
class CNN3DModel(nn.Module):
    def __init__(self):
        super(CNN3DModel, self).__init__()
        
        # 3D CNN for Image Data
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        
        # Table Data
        self.fc_table = nn.Sequential(
            nn.Linear(X_table_train.shape[1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Concatenation
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
        
        # Table data part
        table_out = self.fc_table(table_input)
        
        # Combine both parts
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

# Free unused GPU memory
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

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training process
num_epochs = 300
best_model_wts = model.state_dict()
best_acc = 0.0

# Define model save path
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward propagation
    outputs = model(X_image_train_tensor.to(device), X_table_train_tensor.to(device))
    loss = criterion(outputs, y_train_tensor.to(device))

    # Backward propagation and optimization
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

    # Stop training if accuracy reaches 0.95
    if accuracy >= 0.95:
        print(f"Stopping training as accuracy has reached {accuracy:.4f} which is >= 0.95")
        break

# Load the best model
model.load_state_dict(torch.load(best_model_path))
print(f"Best model loaded with accuracy: {best_acc}")