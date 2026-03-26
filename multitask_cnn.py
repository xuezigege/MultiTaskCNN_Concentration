import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, r2_score
import itertools
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib as mpl

# Define root directory
base_dir = r'./data'


# 1. Directly load pre-split dataset
def load_dataset_from_folder(folder_name):
    """Load dataset from the specified folder."""
    folder_path = os.path.join(base_dir, folder_name)
    csv_path = os.path.join(folder_path, f"{folder_name}.csv")

    # Read CSV file
    df = pd.read_csv(csv_path, encoding='gbk')

    # 1. Handle missing values
    df = df.dropna(subset=['image_path'])
    df['image_path'] = df['image_path'].astype(str)

    # 2. directly use full path from CSV
    image_paths = [path.strip() for path in df['image_path'].values]

    # 3. Check if image files exist
    valid_mask = [os.path.exists(p) for p in image_paths]
    df = df[valid_mask]
    image_paths = [p for p, valid in zip(image_paths, valid_mask) if valid]

    # 4. Extract concentration data (for classification)
    concentrations = df[['x', 'y', 'z']].values.astype(np.float32)

    return image_paths, concentrations


# Load training, validation, and test sets
train_paths, train_concs = load_dataset_from_folder('train')
val_paths, val_concs = load_dataset_from_folder('val')
test_paths, test_concs = load_dataset_from_folder('test')

print(f"Training set size: {len(train_paths)}")
print(f"Validation set size: {len(val_paths)}")
print(f"Test set size: {len(test_paths)}")
print(f"Concentration range: X[{train_concs[:, 0].min()}-{train_concs[:, 0].max()}], " +
      f"Y[{train_concs[:, 1].min()}-{train_concs[:, 1].max()}], " +
      f"Z[{train_concs[:, 2].min()}-{train_concs[:, 2].max()}]")

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Validation and test sets do not require data augmentation
val_test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define custom dataset
class ConcentrationDataset(Dataset):
    def __init__(self, image_paths, concentrations, transform=None):
        self.image_paths = image_paths
        self.concentrations = concentrations
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Unable to load image {img_path}: {e}")
            # Return a placeholder empty image
            image = Image.new('RGB', (128, 128), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

            # Use floating point concentration values (no truncation)
            reg_x = self.concentrations[idx, 0]
            reg_y = self.concentrations[idx, 1]
            reg_z = self.concentrations[idx, 2]

            # Classification target: presence based on threshold
            conc_x = 1 if reg_x > 0.1 else 0
            conc_y = 1 if reg_y > 0.1 else 0
            conc_z = 1 if reg_z > 0.1 else 0

            return image, (conc_x, conc_y, conc_z, reg_x, reg_y, reg_z)


# Create datasets and data loaders
train_dataset = ConcentrationDataset(train_paths, train_concs, transform=transform)
val_dataset = ConcentrationDataset(val_paths, val_concs, transform=val_test_transform)
test_dataset = ConcentrationDataset(test_paths, test_concs, transform=val_test_transform)

# Use num_workers=0 to avoid multiprocessing issues
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)


# Enhanced CNN model (multi-task classification + regression)
class EnhancedMultiTaskCNN(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # Shared base feature extraction layers
        self.base_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )

        # Residual block to be added to the base feature extraction
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU()

            def forward(self, x):
                residual = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += residual
                return self.relu(out)

        # Substance-specific feature extraction layers
        self.x_features = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.y_features = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.z_features = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # X substance classification head (enhanced)
        self.head_x_class = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))

        # Y substance classification head
        self.head_y_class = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))

        # Z substance classification head
        self.head_z_class = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes))

        # Three regression heads (for continuous prediction of X, Y, Z)
        self.head_x_reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),  # add an extra layer
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # add another layer
            nn.ReLU(),
            nn.Linear(128, 1)     # final output
        )

        self.head_y_reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.head_z_reg = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        base_feat = self.base_features(x)

        # Substance-specific features
        x_feat = self.x_features(base_feat)
        y_feat = self.y_features(base_feat)
        z_feat = self.z_features(base_feat)

        # Classification outputs
        class_x = self.head_x_class(x_feat)
        class_y = self.head_y_class(y_feat)
        class_z = self.head_z_class(z_feat)

        # Regression outputs
        reg_x = self.head_x_reg(x_feat)
        reg_y = self.head_y_reg(y_feat)
        reg_z = self.head_z_reg(z_feat)

        return class_x, class_y, class_z, reg_x, reg_y, reg_z


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = EnhancedMultiTaskCNN().to(device)

# Define all possible combination types
COMBINATION_TYPES = ['0', 'X', 'Y', 'Z', 'X+Y', 'X+Z', 'Y+Z', 'X+Y+Z']


# Define combination label function
def get_combination_label(x, y, z):
    """Generate combination label based on X, Y, Z concentrations."""
    # Convert to integers
    x = int(x)
    y = int(y)
    z = int(z)

    # Determine combination type
    if x == 0 and y == 0 and z == 0:
        return '0'
    elif x > 0 and y == 0 and z == 0:
        return 'X'
    elif x == 0 and y > 0 and z == 0:
        return 'Y'
    elif x == 0 and y == 0 and z > 0:
        return 'Z'
    elif x > 0 and y > 0 and z == 0:
        return 'X+Y'
    elif x > 0 and y == 0 and z > 0:
        return 'X+Z'
    elif x == 0 and y > 0 and z > 0:
        return 'Y+Z'
    elif x > 0 and y > 0 and z > 0:
        return 'X+Y+Z'
    else:
        # Handle unusual cases
        return f'X{x}Y{y}Z{z}'


# Enhanced evaluation function (classification + regression)
def evaluate_model(model, loader, save_mistakes=False, save_all_results=False):
    model.eval()
    # Collect classification data (including probabilities)
    all_probs_x, all_probs_y, all_probs_z = [], [], []
    all_preds_x, all_labels_x = [], []
    all_preds_y, all_labels_y = [], []
    all_preds_z, all_labels_z = [], []
    all_preds_comb, all_labels_comb = [], []

    # Collect regression data
    all_reg_preds_x, all_reg_labels_x = [], []
    all_reg_preds_y, all_reg_labels_y = [], []
    all_reg_preds_z, all_reg_labels_z = [], []

    # Collect image paths and detailed information
    image_paths = []
    results = []  # Store detailed information for each image

    # Loss calculation
    class_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.SmoothL1Loss()
    total_class_loss = 0.0
    total_reg_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            conc_x, conc_y, conc_z, reg_x, reg_y, reg_z = labels  # Receive six target values
            conc_x, conc_y, conc_z = conc_x.to(device).long(), conc_y.to(device).long(), conc_z.to(device).long()
            reg_x, reg_y, reg_z = reg_x.to(device).float(), reg_y.to(device).float(), reg_z.to(device).float()

            # Model outputs
            out_x_class, out_y_class, out_z_class, out_x_reg, out_y_reg, out_z_reg = model(images)

            # Obtain classification probabilities (using softmax)
            probs_x = F.softmax(out_x_class, dim=1)
            probs_y = F.softmax(out_y_class, dim=1)
            probs_z = F.softmax(out_z_class, dim=1)

            # Collect positive class probabilities (probability of class 1)
            all_probs_x.extend(probs_x[:, 1].cpu().numpy())  # Only positive class probability
            all_probs_y.extend(probs_y[:, 1].cpu().numpy())
            all_probs_z.extend(probs_z[:, 1].cpu().numpy())

            # Calculate losses
            class_loss_x = class_criterion(out_x_class, conc_x)
            class_loss_y = class_criterion(out_y_class, conc_y)
            class_loss_z = class_criterion(out_z_class, conc_z)
            class_loss = class_loss_x + class_loss_y + class_loss_z

            reg_loss_x = reg_criterion(out_x_reg.squeeze(), reg_x)
            reg_loss_y = reg_criterion(out_y_reg.squeeze(), reg_y)
            reg_loss_z = reg_criterion(out_z_reg.squeeze(), reg_z)
            reg_loss = reg_loss_x + reg_loss_y + reg_loss_z

            total_class_loss += class_loss.item() * images.size(0)
            total_reg_loss += reg_loss.item() * images.size(0)
            total_samples += images.size(0)

            # Get predicted classes
            preds_x = torch.argmax(out_x_class, dim=1)
            preds_y = torch.argmax(out_y_class, dim=1)
            preds_z = torch.argmax(out_z_class, dim=1)

            # Collect classification results
            all_preds_x.extend(preds_x.cpu().numpy())
            all_labels_x.extend(conc_x.cpu().numpy())
            all_preds_y.extend(preds_y.cpu().numpy())
            all_labels_y.extend(conc_y.cpu().numpy())
            all_preds_z.extend(preds_z.cpu().numpy())
            all_labels_z.extend(conc_z.cpu().numpy())

            # Collect regression results
            all_reg_preds_x.extend(out_x_reg.squeeze().cpu().numpy())
            all_reg_labels_x.extend(reg_x.cpu().numpy())
            all_reg_preds_y.extend(out_y_reg.squeeze().cpu().numpy())
            all_reg_labels_y.extend(reg_y.cpu().numpy())
            all_reg_preds_z.extend(out_z_reg.squeeze().cpu().numpy())
            all_reg_labels_z.extend(reg_z.cpu().numpy())

            # Collect combination results
            for i in range(len(conc_x)):
                # Create combination labels
                label_comb = get_combination_label(conc_x[i].item(), conc_y[i].item(), conc_z[i].item())
                pred_comb = get_combination_label(preds_x[i].item(), preds_y[i].item(), preds_z[i].item())

                all_labels_comb.append(label_comb)
                all_preds_comb.append(pred_comb)

                # Collect image paths
                image_paths.append(loader.dataset.image_paths[i])
                # Collect detailed information
                results.append({
                    'image_path': loader.dataset.image_paths[i],
                    'true_combination': label_comb,
                    'pred_combination': pred_comb,
                    'true_x_class': conc_x[i].item(),
                    'pred_x_class': preds_x[i].item(),
                    'true_y_class': conc_y[i].item(),
                    'pred_y_class': preds_y[i].item(),
                    'true_z_class': conc_z[i].item(),
                    'pred_z_class': preds_z[i].item(),
                    'true_x_reg': reg_x[i].item(),
                    'pred_x_reg': out_x_reg[i].squeeze().item(),
                    'true_y_reg': reg_y[i].item(),
                    'pred_y_reg': out_y_reg[i].squeeze().item(),
                    'true_z_reg': reg_z[i].item(),
                    'pred_z_reg': out_z_reg[i].squeeze().item(),
                    'is_correct': int(label_comb == pred_comb)
                })

            # Save misclassified image information to CSV
            if save_mistakes:
                mistakes = [r for r in results if r['is_correct'] == 0]
                if mistakes:
                    df_mistakes = pd.DataFrame(mistakes)
                    df_mistakes.to_csv('classification_mistakes.csv', index=False)
                    print(f"Saved {len(mistakes)} misclassified samples to classification_mistakes.csv")

            # Save all results to CSV
            if save_all_results:
                df_all = pd.DataFrame(results)
                df_all.to_csv('all_test_results.csv', index=False)
                print(f"Saved all test results to all_test_results.csv")

    # Calculate average losses
    avg_class_loss = total_class_loss / total_samples
    avg_reg_loss = total_reg_loss / total_samples

    # Compute evaluation metrics for each substance
    metrics = {}

    # Classification metrics
    for conc_type, preds, labels in zip(['x', 'y', 'z'],
                                        [all_preds_x, all_preds_y, all_preds_z],
                                        [all_labels_x, all_labels_y, all_labels_z]):
        acc = accuracy_score(labels, preds)
        metrics[f'{conc_type}_acc'] = acc

        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        metrics[f'{conc_type}_report'] = report

    # Regression metrics
    for conc_type, preds, labels in zip(['x', 'y', 'z'],
                                        [all_reg_preds_x, all_reg_preds_y, all_reg_preds_z],
                                        [all_reg_labels_x, all_reg_labels_y, all_reg_labels_z]):
        mae = mean_absolute_error(labels, preds)
        r2 = r2_score(labels, preds)
        metrics[f'{conc_type}_reg_mae'] = mae
        metrics[f'{conc_type}_reg_r2'] = r2

    # Full match accuracy (all three substances correctly predicted)
    full_match = np.mean(
        (np.array(all_preds_x) == np.array(all_labels_x)) &
        (np.array(all_preds_y) == np.array(all_labels_y)) &
        (np.array(all_preds_z) == np.array(all_labels_z))
    )
    metrics['full_match_acc'] = full_match

    # Combination accuracy
    comb_acc = accuracy_score(all_labels_comb, all_preds_comb)
    metrics['combination_acc'] = comb_acc

    # Confusion matrices for each substance
    cm_x = confusion_matrix(all_labels_x, all_preds_x)
    cm_y = confusion_matrix(all_labels_y, all_preds_y)
    cm_z = confusion_matrix(all_labels_z, all_preds_z)

    # Combination confusion matrix (using predefined combination types order)
    combination_cm = confusion_matrix(
        all_labels_comb,
        all_preds_comb,
        labels=COMBINATION_TYPES
    )

    # Add to metrics
    metrics['x_cm'] = cm_x
    metrics['y_cm'] = cm_y
    metrics['z_cm'] = cm_z
    metrics['combination_cm'] = combination_cm
    metrics['combination_labels'] = COMBINATION_TYPES

    # Add losses
    metrics['class_loss'] = avg_class_loss
    metrics['reg_loss'] = avg_reg_loss
    metrics['total_loss'] = avg_class_loss + avg_reg_loss

    # Add regression predictions
    metrics['reg_preds'] = {
        'x': (all_reg_preds_x, all_reg_labels_x),
        'y': (all_reg_preds_y, all_reg_labels_y),
        'z': (all_reg_preds_z, all_reg_labels_z)
    }
    # Add probabilities
    metrics['probs'] = {
        'x': (all_probs_x, all_labels_x),
        'y': (all_probs_y, all_labels_y),
        'z': (all_probs_z, all_labels_z)
    }

    return metrics


# Create model saving directory
model_dir = './models'
os.makedirs(model_dir, exist_ok=True)

# Training
num_epochs = 35
best_val_acc = 0.0
class_criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
reg_criterion = nn.SmoothL1Loss()       # Smooth L1 loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode='min',  # Monitor MAE of X substance
                                                 factor=0.5,
                                                 patience=3,
                                                 min_lr=1e-6,
                                                 threshold=0.01,
                                                 threshold_mode='abs'
                                                 )

# Store training history
history = {
    'train_class_loss': [],
    'train_reg_loss': [],
    'train_total_loss': [],
    'train_full_acc': [],
    'train_comb_acc': [],
    'val_class_loss': [],
    'val_reg_loss': [],
    'val_total_loss': [],
    'val_full_acc': [],
    'val_comb_acc': [],
    'val_mae_x': [],
    'val_mae_y': [],
    'val_mae_z': [],
    'val_reg_mae_x': [],
    'val_reg_mae_y': [],
    'val_reg_mae_z': [],
}

# Early stopping
early_stop_counter = 0
patience = 10
best_val_loss = float('inf')

class_criterion = nn.CrossEntropyLoss()
reg_criterion = nn.SmoothL1Loss()  # Using Huber loss as L1 loss

for epoch in range(num_epochs):
    model.train()
    running_class_loss = 0.0
    running_reg_loss = 0.0
    running_total_loss = 0.0

    # Temporary storage for training set evaluation
    train_preds_x, train_labels_x = [], []
    train_preds_y, train_labels_y = [], []
    train_preds_z, train_labels_z = [], []
    train_preds_comb, train_labels_comb = [], []

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        conc_x, conc_y, conc_z, reg_x, reg_y, reg_z = labels  # Receive six target values

        # Explicit device conversion and data type
        conc_x = conc_x.to(device).long()
        conc_y = conc_y.to(device).long()
        conc_z = conc_z.to(device).long()
        reg_x = reg_x.to(device).float()
        reg_y = reg_y.to(device).float()
        reg_z = reg_z.to(device).float()

        optimizer.zero_grad()
        out_x_class, out_y_class, out_z_class, out_x_reg, out_y_reg, out_z_reg = model(images)

        # Calculate classification loss for three tasks
        loss_x_class = class_criterion(out_x_class, conc_x)
        loss_y_class = class_criterion(out_y_class, conc_y)
        loss_z_class = class_criterion(out_z_class, conc_z)
        class_loss = loss_x_class + loss_y_class + loss_z_class

        # Calculate regression loss for three tasks
        loss_x_reg = reg_criterion(out_x_reg.squeeze(), reg_x)
        loss_y_reg = reg_criterion(out_y_reg.squeeze(), reg_y)
        loss_z_reg = reg_criterion(out_z_reg.squeeze(), reg_z)
        reg_loss = loss_x_reg + loss_y_reg + loss_z_reg

        # Dynamic loss weighting (based on task difficulty)
        total_class_loss = loss_x_class + loss_y_class + loss_z_class
        total_reg_loss = loss_x_reg + loss_y_reg + loss_z_reg

        # Auto adjust weights based on task loss ratio
        class_weight = 1.0
        reg_weight = total_class_loss.item() / (total_reg_loss.item() + 1e-8)
        reg_weight = min(max(reg_weight, 0.5), 2.0)  # Limit to reasonable range

        total_loss = class_weight * total_class_loss + reg_weight * total_reg_loss

        total_loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_class_loss += class_loss.item()
        running_reg_loss += reg_loss.item()
        running_total_loss += total_loss.item()

        # Collect training set predictions (to compute training accuracy)
        preds_x = torch.argmax(out_x_class, dim=1)
        preds_y = torch.argmax(out_y_class, dim=1)
        preds_z = torch.argmax(out_z_class, dim=1)

        train_preds_x.extend(preds_x.cpu().numpy())
        train_labels_x.extend(conc_x.cpu().numpy())
        train_preds_y.extend(preds_y.cpu().numpy())
        train_labels_y.extend(conc_y.cpu().numpy())
        train_preds_z.extend(preds_z.cpu().numpy())
        train_labels_z.extend(conc_z.cpu().numpy())

        # Collect combination results
        for i in range(len(conc_x)):
            label_comb = get_combination_label(conc_x[i].item(), conc_y[i].item(), conc_z[i].item())
            pred_comb = get_combination_label(preds_x[i].item(), preds_y[i].item(), preds_z[i].item())
            train_labels_comb.append(label_comb)
            train_preds_comb.append(pred_comb)

        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                  f'Class Loss: {class_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}')

    # Compute training set accuracy
    train_full_acc = np.mean(
        (np.array(train_preds_x) == np.array(train_labels_x)) &
        (np.array(train_preds_y) == np.array(train_labels_y)) &
        (np.array(train_preds_z) == np.array(train_labels_z))
    )

    train_comb_acc = accuracy_score(train_labels_comb, train_preds_comb)

    # Record training losses and accuracies
    avg_class_loss = running_class_loss / len(train_loader)
    avg_reg_loss = running_reg_loss / len(train_loader)
    avg_total_loss = running_total_loss / len(train_loader)

    history['train_class_loss'].append(avg_class_loss)
    history['train_reg_loss'].append(avg_reg_loss)
    history['train_total_loss'].append(avg_total_loss)
    history['train_full_acc'].append(train_full_acc)
    history['train_comb_acc'].append(train_comb_acc)

    # Validation set evaluation
    val_metrics = evaluate_model(model, val_loader)

    # Record validation metrics
    history['val_class_loss'].append(val_metrics['class_loss'])
    history['val_reg_loss'].append(val_metrics['reg_loss'])
    history['val_total_loss'].append(val_metrics['total_loss'])
    history['val_full_acc'].append(val_metrics['full_match_acc'])
    history['val_comb_acc'].append(val_metrics['combination_acc'])
    history['val_reg_mae_x'].append(val_metrics['x_reg_mae'])
    history['val_reg_mae_y'].append(val_metrics['y_reg_mae'])
    history['val_reg_mae_z'].append(val_metrics['z_reg_mae'])

    print(f'\nEpoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_total_loss:.4f} (Class: {avg_class_loss:.4f}, Reg: {avg_reg_loss:.4f}), '
          f'Train Full Acc: {train_full_acc:.4f}, Train Comb Acc: {train_comb_acc:.4f}')
    print(
        f'Val Loss: {val_metrics["total_loss"]:.4f} (Class: {val_metrics["class_loss"]:.4f}, Reg: {val_metrics["reg_loss"]:.4f}), '
        f'Val Full Acc: {val_metrics["full_match_acc"]:.4f}, Val Comb Acc: {val_metrics["combination_acc"]:.4f}')
    print(
        f'Val Reg MAE: X={val_metrics["x_reg_mae"]:.4f}, Y={val_metrics["y_reg_mae"]:.4f}, Z={val_metrics["z_reg_mae"]:.4f}')

    # Save best model (based on validation full match accuracy)
    if val_metrics['full_match_acc'] > best_val_acc:
        best_val_acc = val_metrics['full_match_acc']
        torch.save(model.state_dict(), os.path.join(model_dir, 'best_concentration_model.pth'))
        print(f"Saved new best model with full match accuracy: {best_val_acc:.4f}")

    # Early stopping
    current_val_loss = val_metrics["total_loss"]
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Validation loss did not improve for {patience} epochs, stopping early.")
            break

    # Update learning rate
    scheduler.step(val_metrics['full_match_acc'])
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current learning rate: {current_lr}\n")

# Save final model
torch.save(model.state_dict(), os.path.join(model_dir, 'final_concentration_model.pth'))
print("Training completed, model saved.")

# ================ Visualize training process ================
plt.figure(figsize=(20, 15))

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Preferred SimHei, fallback DejaVu
plt.rcParams['axes.unicode_minus'] = False


# 2. Add ROC curve plotting function
def plot_roc_curves(metrics):
    """Plot ROC curves for X, Y, Z substances."""
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red']
    substances = ['X', 'Y', 'Z']

    # Plot ROC curve for each substance
    for i, substance in enumerate(['x', 'y', 'z']):
        probs, labels = metrics['probs'][substance]
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[i],
                 label=f'{substances[i]} (AUC = {roc_auc:.2f})',
                 lw=2, alpha=0.8)

    # Add diagonal line
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set chart properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save chart
    plt.tight_layout()
    plt.savefig('roc_curves.pdf', bbox_inches='tight', dpi=300)
    plt.show()

    # Return AUC values
    return {
        'x_auc': auc(*roc_curve(metrics['probs']['x'][1], metrics['probs']['x'][0])[:2]),
        'y_auc': auc(*roc_curve(metrics['probs']['y'][1], metrics['probs']['y'][0])[:2]),
        'z_auc': auc(*roc_curve(metrics['probs']['z'][1], metrics['probs']['z'][0])[:2])
    }


# Loss curves
plt.subplot(2, 2, 1)
plt.plot(history['train_total_loss'], label='Train Total Loss')
plt.plot(history['val_total_loss'], label='Validation Total Loss')
plt.plot(history['train_class_loss'], '--', label='Train Class Loss')
plt.plot(history['val_class_loss'], '--', label='Validation Class Loss')
plt.plot(history['train_reg_loss'], '-.', label='Train Reg Loss')
plt.plot(history['val_reg_loss'], '-.', label='Validation Reg Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Accuracy curves
plt.subplot(2, 2, 2)
plt.plot(history['train_full_acc'], label='Train Full Match Accuracy')
plt.plot(history['val_full_acc'], label='Validation Full Match Accuracy')
plt.plot(history['train_comb_acc'], '--', label='Train Combination Accuracy')
plt.plot(history['val_comb_acc'], '--', label='Validation Combination Accuracy')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# MAE curves
plt.subplot(2, 2, 3)
plt.plot(history['val_reg_mae_x'], label='X MAE')
plt.plot(history['val_reg_mae_y'], label='Y MAE')
plt.plot(history['val_reg_mae_z'], label='Z MAE')
plt.title('Validation Mean Absolute Error (MAE)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

# Training vs validation loss
plt.subplot(2, 2, 4)
plt.plot(history['train_total_loss'], label='Train Loss')
plt.plot(history['val_total_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.pdf', bbox_inches='tight', dpi=300)
plt.show()

# ================ Load best model for final evaluation ================
model.load_state_dict(torch.load(os.path.join(model_dir, 'best_concentration_model.pth')))
final_metrics = evaluate_model(model, test_loader, save_mistakes=True, save_all_results=True)  # Use test set for evaluation

# Print final evaluation results
print("\nFinal Evaluation Results:")
print(f"Full Match Accuracy: {final_metrics['full_match_acc']:.4f}")
print(f"Combination Accuracy: {final_metrics['combination_acc']:.4f}")
print(f"Classification Loss: {final_metrics['class_loss']:.4f}, Regression Loss: {final_metrics['reg_loss']:.4f}")
print(
    f"X Substance Accuracy: {final_metrics['x_acc']:.4f}, Reg MAE: {final_metrics['x_reg_mae']:.4f}, R²: {final_metrics['x_reg_r2']:.4f}")
print(
    f"Y Substance Accuracy: {final_metrics['y_acc']:.4f}, Reg MAE: {final_metrics['y_reg_mae']:.4f}, R²: {final_metrics['y_reg_r2']:.4f}")
print(
    f"Z Substance Accuracy: {final_metrics['z_acc']:.4f}, Reg MAE: {final_metrics['z_reg_mae']:.4f}, R²: {final_metrics['z_reg_r2']:.4f}")

# Add ROC curve analysis
print("\nROC Curve Analysis:")
roc_aucs = plot_roc_curves(final_metrics)
print(f"X Substance AUC: {roc_aucs['x_auc']:.4f}")
print(f"Y Substance AUC: {roc_aucs['y_auc']:.4f}")
print(f"Z Substance AUC: {roc_aucs['z_auc']:.4f}")


# ================ Combination classification confusion matrix heatmap ================
def plot_combination_confusion_matrix(cm, classes):
    """Plot heatmap for combination classification confusion matrix."""
    # Save raw confusion matrix (un-normalized)
    raw_cm = cm.copy()
    # Normalize confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Create DataFrame for raw counts
    df_raw = pd.DataFrame(raw_cm, index=classes, columns=classes)
    df_raw.index.name = 'True Label'
    df_raw.columns.name = 'Predicted Label'

    # Create DataFrame for normalized percentages
    df_norm = pd.DataFrame(np.round(cm * 100, 2), index=classes, columns=classes)
    df_norm.index.name = 'True Label'
    df_norm.columns.name = 'Predicted Label'

    # Save CSV files
    df_raw.to_csv('combination_confusion_matrix_raw.csv')
    df_norm.to_csv('combination_confusion_matrix_normalized.csv')

    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                cbar=False, annot_kws={"size": 14})

    plt.title('Combination Classification Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Combination', fontsize=14)
    plt.ylabel('True Combination', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig('combination_confusion_matrix.pdf', bbox_inches='tight', dpi=300)
    plt.show()


# Plot combination confusion matrix
plot_combination_confusion_matrix(final_metrics['combination_cm'], COMBINATION_TYPES)


# ================ Save concentration prediction results to CSV ================
def save_regression_results(loader, model, save_path):
    """Save regression prediction results to CSV."""
    model.eval()
    results = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            conc_x, conc_y, conc_z, reg_x, reg_y, reg_z = labels

            # Model outputs
            _, _, _, out_x_reg, out_y_reg, out_z_reg = model(images)

            # Collect results
            for i in range(len(images)):
                results.append({
                    'true_x': reg_x[i].item(),
                    'true_y': reg_y[i].item(),
                    'true_z': reg_z[i].item(),
                    'pred_x': out_x_reg[i].squeeze().item(),
                    'pred_y': out_y_reg[i].squeeze().item(),
                    'pred_z': out_z_reg[i].squeeze().item()
                })

    # Create DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    return df


def calculate_concentration_stats(df, substance):
    """Calculate concentration statistics (mean and standard deviation)."""
    # Group by true concentration and compute prediction statistics
    grouped = df.groupby(f'true_{substance}').agg({
        f'pred_{substance}': ['mean', 'std', 'count']
    }).reset_index()

    # Rename columns
    grouped.columns = [
        f'{substance}_concentration',
        f'mean_pred_{substance}',
        f'std_pred_{substance}',
        'sample_count'
    ]

    # Calculate error bars (95% confidence interval)
    grouped[f'ci_{substance}'] = 1.96 * grouped[f'std_pred_{substance}'] / np.sqrt(grouped['sample_count'])

    return grouped


# Save test set results
test_results_df = save_regression_results(test_loader, model, 'test_regression_results.csv')

# Save validation set results
val_results_df = save_regression_results(val_loader, model, 'val_regression_results.csv')

# Save training set results
train_results_df = save_regression_results(train_loader, model, 'train_regression_results.csv')

# Calculate and save concentration statistics
for substance in ['x', 'y', 'z']:
    # Test set statistics
    test_stats = calculate_concentration_stats(test_results_df, substance)
    test_stats.to_csv(f'test_{substance}_concentration_stats.csv', index=False)

    # Validation set statistics
    val_stats = calculate_concentration_stats(val_results_df, substance)
    val_stats.to_csv(f'val_{substance}_concentration_stats.csv', index=False)

    # Training set statistics
    train_stats = calculate_concentration_stats(train_results_df, substance)
    train_stats.to_csv(f'train_{substance}_concentration_stats.csv', index=False)


# Visualize regression prediction results
def plot_regression_results(reg_preds, reg_labels, substance):
    plt.figure(figsize=(10, 8))
    plt.scatter(reg_labels, reg_preds, alpha=0.6)

    # Add ideal prediction line
    min_val = min(min(reg_labels), min(reg_preds))
    max_val = max(max(reg_labels), max(reg_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Calculate metrics
    mae = mean_absolute_error(reg_labels, reg_preds)
    r2 = r2_score(reg_labels, reg_preds)

    plt.title(f'{substance} Concentration Prediction\nMAE: {mae:.3f}, R²: {r2:.3f}')
    plt.xlabel('True Concentration')
    plt.ylabel('Predicted Concentration')
    plt.grid(True)
    plt.savefig(f'{substance}_regression_results.pdf', bbox_inches='tight', dpi=300)
    plt.show()


# Plot regression results
for substance, data in final_metrics['reg_preds'].items():
    plot_regression_results(data[0], data[1], substance.upper())


# ================ Unknown image prediction ================
def predict_unknown_image(image_path):
    """Predict concentration (continuous and classification) for an unknown image."""
    model.eval()

    # Load and preprocess image - use validation set transform
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out_x_class, out_y_class, out_z_class, out_x_reg, out_y_reg, out_z_reg = model(image_tensor)

        # Classification predictions
        class_pred_x = torch.argmax(out_x_class, dim=1).item()
        class_pred_y = torch.argmax(out_y_class, dim=1).item()
        class_pred_z = torch.argmax(out_z_class, dim=1).item()

        # Regression predictions (continuous values)
        reg_pred_x = out_x_reg.squeeze().item()
        reg_pred_y = out_y_reg.squeeze().item()
        reg_pred_z = out_z_reg.squeeze().item()

        # Combination prediction
        comb_pred = get_combination_label(class_pred_x, class_pred_y, class_pred_z)

        # Classification probabilities
        prob_x = torch.softmax(out_x_class, dim=1).squeeze().cpu().numpy()
        prob_y = torch.softmax(out_y_class, dim=1).squeeze().cpu().numpy()
        prob_z = torch.softmax(out_z_class, dim=1).squeeze().cpu().numpy()

    # Create result dictionary
    result = {
        'image_path': image_path,
        'classification': {
            'x': class_pred_x,
            'y': class_pred_y,
            'z': class_pred_z,
            'combination': comb_pred,
            'probabilities': {
                'x': prob_x,
                'y': prob_y,
                'z': prob_z
            }
        },
        'regression': {
            'x': reg_pred_x,
            'y': reg_pred_y,
            'z': reg_pred_z
        }
    }

    return result


# Example: predict unknown image
unknown_image_path = r'path/to/your/image.jpg'
if os.path.exists(unknown_image_path):
    prediction = predict_unknown_image(unknown_image_path)

    print("\nUnknown Image Prediction Results:")
    print(f"Image path: {prediction['image_path']}")
    print("\nClassification Prediction:")
    print(f"X concentration: {prediction['classification']['x']}")
    print(f"Y concentration: {prediction['classification']['y']}")
    print(f"Z concentration: {prediction['classification']['z']}")
    print(f"Combination type: {prediction['classification']['combination']}")

    print("\nRegression Prediction (continuous):")
    print(f"X concentration: {prediction['regression']['x']:.2f}")
    print(f"Y concentration: {prediction['regression']['y']:.2f}")
    print(f"Z concentration: {prediction['regression']['z']:.2f}")

    # Visualize probability distribution
    plt.figure(figsize=(15, 5))
    substances = ['X', 'Y', 'Z']
    for i, substance in enumerate(substances):
        plt.subplot(1, 3, i + 1)
        probs = prediction['classification']['probabilities'][substance.lower()]
        plt.bar(range(len(probs)), probs)
        plt.title(f'{substance} Concentration Probabilities')
        plt.xlabel('Concentration Level')
        plt.ylabel('Probability')
        plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('unknown_image_probabilities.pdf', bbox_inches='tight', dpi=300)
    plt.show()
else:
    print(f"Unknown image does not exist: {unknown_image_path}")