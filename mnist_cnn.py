# mnist_cnn.py
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# pip install scikit-learn matplotlib

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# -----------------------
# 0) Reproducibility
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -----------------------
# 1) Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# 2) Data (MNIST)
# -----------------------
# MNIST images are grayscale 28x28.
# We convert to tensor and normalize (mean/std are common MNIST values).
transform = transforms.Compose([
    transforms.ToTensor(),                 # (1,28,28), values in [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # normalize
])

train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

# -----------------------
# 3) Model: CNN
# -----------------------
class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (B, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # -> (B,32,28,28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> (B,64,14,14) after pool
        self.pool = nn.MaxPool2d(2, 2)                            # halves H,W
        self.dropout = nn.Dropout(0.25)

        # After: conv1->pool => (B,32,14,14)
        # Then: conv2->pool => (B,64,7,7)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (0-9)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # (B,32,28,28)
        x = self.pool(x)            # (B,32,14,14)

        x = F.relu(self.conv2(x))   # (B,64,14,14)
        x = self.pool(x)            # (B,64,7,7)

        x = self.dropout(x)
        x = torch.flatten(x, 1)     # (B, 64*7*7)

        x = F.relu(self.fc1(x))     # (B,128)
        x = self.dropout(x)

        logits = self.fc2(x)        # (B,10) raw scores (logits)
        return logits

model = MNISTCNN().to(device)

# -----------------------
# 4) Loss + Optimizer
# -----------------------
# For multi-class classification in PyTorch:
# Use CrossEntropyLoss with raw logits (NO softmax needed in forward)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------
# 5) Train + Evaluate
# -----------------------
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_true = []
    all_pred = []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        logits = model(Xb)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        all_pred.extend(preds)
        all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)
    return acc, cm, all_true, all_pred

EPOCHS = 5  # MNIST gets strong accuracy fast; raise to 10 for more

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader)
    test_acc, _, _, _ = evaluate(model, test_loader)
    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.4f}")

# Final evaluation report
test_acc, cm, y_true, y_pred = evaluate(model, test_loader)
print("\n--- Final Test Results ---")
print("Accuracy:", f"{test_acc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

# Save model
torch.save(model.state_dict(), "mnist_cnn.pth")
print("\nSaved model to mnist_cnn.pth")

# RESULT
# Using device: cpu
# 100.0%
# 100.0%
# 100.0%
# 100.0%
# Epoch 01 | Train Loss: 0.1757 | Test Acc: 0.9817
# Epoch 02 | Train Loss: 0.0618 | Test Acc: 0.9886
# Epoch 03 | Train Loss: 0.0461 | Test Acc: 0.9902
# Epoch 04 | Train Loss: 0.0392 | Test Acc: 0.9894
# Epoch 05 | Train Loss: 0.0315 | Test Acc: 0.9906

# --- Final Test Results ---
# Accuracy: 0.9906

# Confusion Matrix:
#  [[ 976    0    0    0    1    0    2    1    0    0]
#  [   0 1134    0    0    1    0    0    0    0    0]
#  [   2    2 1019    0    2    0    0    6    1    0]
#  [   0    0    0 1000    0    6    0    1    2    1]
#  [   0    0    0    0  981    0    0    0    0    1]
#  [   2    1    0    4    0  880    4    0    0    1]
#  [   4    2    0    0    1    1  950    0    0    0]
#  [   0    1    3    0    0    0    0 1020    1    3]
#  [   4    1    1    0    3    0    1    2  958    4]
#  [   0    0    0    0   13    3    0    4    1  988]]

# Classification Report:
#                precision    recall  f1-score   support

#            0     0.9879    0.9959    0.9919       980
#            1     0.9939    0.9991    0.9965      1135
#            2     0.9961    0.9874    0.9917      1032
#            3     0.9960    0.9901    0.9930      1010
#            4     0.9790    0.9990    0.9889       982
#            5     0.9888    0.9865    0.9877       892
#            6     0.9927    0.9916    0.9922       958
#            7     0.9865    0.9922    0.9893      1028
#            8     0.9948    0.9836    0.9892       974
#            9     0.9900    0.9792    0.9846      1009

#     accuracy                         0.9906     10000
#    macro avg     0.9906    0.9905    0.9905     10000
# weighted avg     0.9906    0.9906    0.9906     10000


# Saved model to mnist_cnn.pth
# (.venv) PS C:\Users\zubai\Desktop\python_workshop4>
