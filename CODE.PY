BASE_DIR = "./"
# STEP1: GENERATING SYNTHETIC IMAGES FROM PROMPTS USING STABLE DIFFUSION MODEL 
# ALREADY DONE AND UPLOADED IN IMAGE1000 FOLDER FOR THREE CLASSES
################################ STEP2: TRAINING THE MODEL ON TRAIN SET OF SYNTHETIC IMAGES
# IMPORTS
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision.transforms import RandAugment
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.autograd import Function
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image

# MODEL TRAIN HYPERPARAMETERS (PRETRAINED RESNET)
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
IMG_SIZE = 96
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.3   

# DATA AUGMENTATION
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DATASET, DATALOADER
DATA_DIR = BASE_DIR + "/image1000/synthetic_dataset"
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# MODEL ARCHITECTURE BASED ON RESNET18
class CustomResNet(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base.maxpool = nn.Identity()  
        
        self.base.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.base.fc.in_features, num_classes))
        
    def forward(self, x):
        return self.base(x)
    
    def extract_features(self, x):
        """
        Extract features from the backbone (512-dim vector)
        following the forward pass up to the average pooling layer.
        """
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        return x

# MODEL INITIALIZATION, LOSS, OPTIMIZER, SCHEDULER
torch.manual_seed(0)
model = CustomResNet(num_classes=3, dropout_rate=DROPOUT_RATE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)  
 
# LOSSES & ACCURACY
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []
best_val_acc = 0.0

# TRAIN LOOP
for epoch in range(EPOCHS):
    # TRAIN PHASE
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    # VAL PHASE
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    
    # UPDATE LR 
    scheduler.step()
    # BEST MODEL
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    
    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)
    
    # PRINT EPOCH SUMMARY
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
    print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")  # Log learning rate
    print("-" * 50)

# PLOTS LOSSES & ACCURACY
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train')
plt.plot(val_loss_history, label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc_history, label='Train')
plt.plot(val_acc_history, label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('training_curves.png')
plt.show()
################################## STEP3: TESTING MODEL ON REAL DATSET and COMPARE WITH VAL SET PERFORMANCE
# DIRECTORY
REAL_TEST_DIR = BASE_DIR + "/stl10_png/labeled_test"    
SYNTH_VAL_DIR = BASE_DIR + "/image1000/synthetic_dataset/val"  
MODEL_PATH = "best_model.pth"

# LOAD TRAINED MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomResNet(num_classes=3, dropout_rate=DROPOUT_RATE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

# TEST TIME AUGMENTATION
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.FiveCrop(IMG_SIZE),
    transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops]),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])(crop) for crop in crops]))
])

def five_crop_prediction(batch):
    # INPUT: BATCH OF 5 CROPS
    # OUTPUT: AVERAGED PREDICTIONS
    bs, ncrops, c, h, w = batch.size()
    outputs = model(batch.view(-1, c, h, w))
    return outputs.view(bs, ncrops, -1).mean(1)

# DATASETS, DATALOADER
synth_val_dataset = datasets.ImageFolder(SYNTH_VAL_DIR, transform = val_transform)
real_test_dataset = datasets.ImageFolder(REAL_TEST_DIR, transform = test_transform)

synth_val_loader = DataLoader(synth_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
real_test_loader = DataLoader(real_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# EVALUATE
def evaluate_model(model, loader, name="Dataset", use_tta=False):
    model.eval() 
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if use_tta:  
                outputs = five_crop_prediction(inputs)
            else:
                outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item() * inputs.size(0)
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f"{name} Results:")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
    return accuracy

print("="*50)
synth_acc = evaluate_model(model, synth_val_loader, "Synthetic Validation")
print("="*50)
real_acc = evaluate_model(model, real_test_loader, "Real Test", use_tta=True)
print("="*50)

# Calculate performance gap
print(f"\nPerformance Gap Analysis:")
print(f"Synthetic Validation Accuracy: {synth_acc:.2f}%")
print(f"Real Test Accuracy: {real_acc:.2f}%")
print(f"Absolute Gap: {abs(synth_acc - real_acc):.2f}%")
print(f"Relative Gap: {abs(synth_acc - real_acc)/synth_acc*100:.2f}%")

################################### STEP4: DOMAIN ADAPTATION WITH CDAN, SHARPENED PSEUDO LABELS,
# CONSISTENCY REGULARIZATION, SPECTRAL NORMALIZATION, AND RANDAUGMENT
# DIRECTORIES
UNLABELED_DIR = BASE_DIR + "/stl10_png/unlabeled" 

# HYPERPARAMETERS
CONF_THRESH = 0.99        # Confidence threshold for pseudo-labeling
EPOCHS_ADAPT = 20        # Domain adaptation epochs
LR_ADAPT = 1e-4          # Lower learning rate for adaptation
WEIGHT_DECAY_LR = 1e-5
ADV_LAMBDA = 1           # Weight for the adversarial loss
TEMP = 0.3               # Temperature for sharpening pseudo-labels
CONSISTENCY_WEIGHT = 0.5 # Weight for the consistency loss

# UNLABELED DATSET FOR PSEUDO-LABELING (WEAK AUGMENTATION)
class UnlabeledDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)

unlabeled_dataset = UnlabeledDataset(UNLABELED_DIR)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False)

# PSEUDO LABELING WITH SHARPENED PREDICTION (TEMP SACLING)
pseudo_data = []
model.eval()
with torch.no_grad():
    for images in unlabeled_loader:
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs / TEMP, dim=1)
        max_probs, preds = torch.max(probs, dim=1)
        mask = max_probs > CONF_THRESH 
        pseudo_data.extend(zip(images[mask].cpu(), preds[mask].cpu()))
print(f"Retained {len(pseudo_data)}/{len(unlabeled_dataset)} samples after filtering")

# DATASET WRAPPERS WITH DOMAIN LABELS
# Synthetic dataset wrapper (domain label 0)
class DomainSyntheticDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        domain = 0  
        return img, label, domain

# Pseudo dataset wrapper (domain label 1)
class PseudoDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        domain = 1  
        return img, int(label), domain

# Wrap and combine the labeled datasets
domain_synthetic = DomainSyntheticDataset(train_dataset)
domain_pseudo = PseudoDataset(pseudo_data)
combined_dataset = ConcatDataset([domain_synthetic, domain_pseudo])
train_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

# UNLABELD DATASET FOR CONSISTENCY REGULARIZATION
# (Returns two views: one weak and one strong using RandAugment)
# Define weak and strong transforms for consistency
weak_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
strong_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    RandAugment(), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class UnlabeledConsistencyDataset(Dataset):
    def __init__(self, root_dir, weak_transform, strong_transform):
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.weak_transform(img), self.strong_transform(img)

consistency_dataset = UnlabeledConsistencyDataset(UNLABELED_DIR, weak_transform, strong_transform)
consistency_loader = DataLoader(consistency_dataset, batch_size=BATCH_SIZE, shuffle=True)

# GRADIENT REVERSAL LAYER (GRL)
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

# CDAN Domain Classifier with Spectral Normalization
class CDANDomainClassifier(nn.Module):
    def __init__(self, feature_dim=512, num_classes=3):
        super(CDANDomainClassifier, self).__init__()
        self.input_dim = feature_dim * num_classes
        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(self.input_dim, 1024)),
            nn.ReLU(),
            nn.Dropout(0.5),
            spectral_norm(nn.Linear(1024, 1024)),
            nn.ReLU(),
            nn.Dropout(0.5),
            spectral_norm(nn.Linear(1024, 1))
        )
    
    def forward(self, features, softmax_output):
        op_out = torch.bmm(softmax_output.unsqueeze(2), features.unsqueeze(1))
        op_out = op_out.view(features.size(0), -1) 
        return self.fc(op_out)

domain_clf = CDANDomainClassifier(feature_dim=512, num_classes=3).to(device)

# DOMAIN ADAPTATION TRAINING WITH ADVERSARIAL LOSS (CDAN) AND CONSISTENCY REGULARIZATION
# Unfreeze the entire model for adversarial adaptation.
for param in model.parameters():
    param.requires_grad = True

# Optimizers for main model and domain classifier
optimizer_model = optim.AdamW(model.parameters(), lr=LR_ADAPT, weight_decay=WEIGHT_DECAY_LR)
optimizer_domain = optim.Adam(domain_clf.parameters(), lr=LR_ADAPT)
scheduler = CosineAnnealingLR(optimizer_model, T_max=EPOCHS_ADAPT)

# Loss functions
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

train_losses = []
train_accuracies = []
consistency_losses = [] 

for epoch in range(EPOCHS_ADAPT):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # ---- Combined Labeled Training (Synthetic + Pseudo) with Domain Adaptation ----
    for inputs, labels, domain_labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        domain_labels = torch.as_tensor(domain_labels, dtype=torch.float32, device=device).unsqueeze(1)
        
        # Step 1: Update main model (classification + adversarial loss)
        optimizer_model.zero_grad()
        optimizer_domain.zero_grad()  # Zero domain optimizer for combined step
        
        outputs = model(inputs)
        class_loss = class_criterion(outputs, labels)
        
        softmax_out = F.softmax(outputs, dim=1)
        features = model.extract_features(inputs)
        features_reversed = grad_reverse(features, ADV_LAMBDA)
        domain_preds_for_model = domain_clf(features_reversed, softmax_out)
        domain_loss_model = domain_criterion(domain_preds_for_model, domain_labels)
        
        total_loss = class_loss + domain_loss_model
        total_loss.backward()
        optimizer_model.step()
        
        # Step 2: Update domain classifier only
        optimizer_domain.zero_grad()
        features_detached = features.detach()
        domain_preds = domain_clf(features_detached, softmax_out.detach())
        domain_loss = domain_criterion(domain_preds, domain_labels)
        domain_loss.backward()
        optimizer_domain.step()
        
        running_loss += total_loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    scheduler.step()
    
    print(f"Adaptation Epoch {epoch+1}/{EPOCHS_ADAPT} - Labeled Training")
    print(f"Train Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%")
    print("-" * 50)
    
    # ---- Consistency Regularization on Unlabeled Data ----
    model.train()
    epoch_consistency_loss = 0.0
    consistency_batches = 0
    for weak_imgs, strong_imgs in consistency_loader:
        weak_imgs, strong_imgs = weak_imgs.to(device), strong_imgs.to(device)
        # Forward pass for both views
        weak_logits = model(weak_imgs)
        strong_logits = model(strong_imgs)
        weak_prob = F.softmax(weak_logits, dim=1)
        strong_prob = F.softmax(strong_logits, dim=1)
        # Symmetric KL divergence
        kl_loss = (F.kl_div(strong_prob.log(), weak_prob, reduction='batchmean') + 
                   F.kl_div(weak_prob.log(), strong_prob, reduction='batchmean')) / 2.0
        
        optimizer_model.zero_grad()
        (CONSISTENCY_WEIGHT * kl_loss).backward()
        optimizer_model.step()
        
        epoch_consistency_loss += kl_loss.item()
        consistency_batches += 1
    avg_consistency_loss = epoch_consistency_loss / consistency_batches if consistency_batches > 0 else 0.0
    consistency_losses.append(avg_consistency_loss)
    
    print(f"Adaptation Epoch {epoch+1}/{EPOCHS_ADAPT} - Consistency Loss: {avg_consistency_loss:.4f}")
    print("=" * 50)

# ---- Plot Training Curves ----
epochs_range = range(1, EPOCHS_ADAPT+1)
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(epochs_range, train_losses, marker='o', color=color, label='Domain Adaptation Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True)
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('Accuracy (%)', color=color)
ax2.plot(epochs_range, train_accuracies, marker='s', color=color, label='Train Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.title('Domain Adaptation (CDAN + Sharpened Pseudo Labels + Consistency Regularization)')
plt.savefig('domain_adaptation_cdan_consistency.png')
plt.show()

# RUN DOMAIN ADAPTATION EVALUATION
print("="*50)
real_acc = evaluate_model(model, real_test_loader, "Real Test", use_tta=True)
print("="*50)
print(f"\nFINAL Performance Gap Analysis:")
print(f"Synthetic Validation Accuracy: {synth_acc:.2f}%")
print(f"Real Test Accuracy: {real_acc:.2f}%")
print(f"Absolute Gap: {abs(synth_acc - real_acc):.2f}%")
print(f"Relative Gap: {abs(synth_acc - real_acc)/synth_acc*100:.2f}%")