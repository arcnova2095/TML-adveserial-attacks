import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, resnet34, resnet50

# ==========================================
# 1. Adversarial Attack Definitions
# ==========================================

def fgsm_attack(model, images, labels, criterion, epsilon):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    
    data_grad = images.grad.data
    perturbed_image = images + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image.detach()

def bim_attack(model, images, labels, criterion, epsilon, alpha, iters):
    perturbed_images = images.clone().detach()
    for _ in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()
            eta = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
            perturbed_images = torch.clamp(images + eta, 0, 1)
    return perturbed_images.detach()

def pgd_attack(model, images, labels, criterion, epsilon, alpha, iters):
    perturbed_images = images.clone().detach()
    # PGD initializes with random uniform noise within the epsilon sphere
    perturbed_images = perturbed_images + torch.empty_like(perturbed_images).uniform_(-epsilon, epsilon)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    for _ in range(iters):
        perturbed_images.requires_grad = True
        outputs = model(perturbed_images)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            perturbed_images = perturbed_images + alpha * perturbed_images.grad.sign()
            eta = torch.clamp(perturbed_images - images, min=-epsilon, max=epsilon)
            perturbed_images = torch.clamp(images + eta, 0, 1)
    return perturbed_images.detach()

# ==========================================
# 2. Data Loading (From Template)
# ==========================================

print("Loading dataset...")
data = np.load("tml26_task3\\train.npz")
# Images normalized to [0, 1] as required by the attack clamping functions
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

dataset = TensorDataset(images, labels)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

print("Dataset size:", len(dataset))
print("Image shape:", images.shape)
print("Label range:", labels.min().item(), "to", labels.max().item())

# ==========================================
# 3. Model & Hardware Setup
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

NUM_CLASSES = 9


model = resnet34(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Sanity check -- output shape must be (1, 9) [cite: 57] and input 3x32x32 [cite: 58]
model.eval()
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    out = model(dummy_input)
print("Output shape verification:", out.shape)

# ==========================================
# 4. Training Loop
# ==========================================


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 30
EPSILON = 8/255
ALPHA = 2/255
ITERS = 10

print("Starting Multi-Attack Adversarial Training...")
model.train()

# for epoch in range(EPOCHS):
#     running_loss = 0.0
#     correct_clean = 0
#     total_clean = 0
    
#     for batch_idx, (batch_images, batch_labels) in enumerate(loader):
#         batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
#         # Determine the size of each quarter (handles uneven final batches)
#         chunk_size = batch_images.size(0) // 4
        
#         # If the batch is too small to split 4 ways (e.g., end of dataset), skip or handle
#         if chunk_size == 0:
#             continue
            
#         # -----------------------------------------
#         # Step A: Split the batch and generate attacks
#         # -----------------------------------------
        
#         # 1. Clean inputs (1st Quarter)
#         clean_inputs = batch_images[:chunk_size]
#         clean_targets = batch_labels[:chunk_size]
        
#         # 2. FGSM inputs (2nd Quarter)
#         fgsm_base = batch_images[chunk_size:2*chunk_size]
#         fgsm_targets = batch_labels[chunk_size:2*chunk_size]
#         fgsm_inputs = fgsm_attack(model, fgsm_base, fgsm_targets, criterion, EPSILON)
        
#         # 3. BIM inputs (3rd Quarter)
#         # bim_base = batch_images[2*chunk_size:3*chunk_size]
#         # bim_targets = batch_labels[2*chunk_size:3*chunk_size]
#         # bim_inputs = bim_attack(model, bim_base, bim_targets, criterion, EPSILON, ALPHA, ITERS)
        
#         # 4. PGD inputs (4th Quarter)
#         pgd_base = batch_images[3*chunk_size:]
#         pgd_targets = batch_labels[3*chunk_size:]
#         pgd_inputs = pgd_attack(model, pgd_base, pgd_targets, criterion, EPSILON, ALPHA, ITERS)
        
#         # -----------------------------------------
#         # Step B: Recombine and Optimize
#         # -----------------------------------------
        
#         # Concatenate into one unified training batch
#         train_images = torch.cat([clean_inputs, fgsm_inputs, pgd_inputs])
#         train_labels = torch.cat([clean_targets, fgsm_targets, pgd_targets])
        
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(train_images)
#         loss = criterion(outputs, train_labels)
        
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
        
#         # Track training accuracy purely on the clean quarter
#         with torch.no_grad():
#             clean_outputs = outputs[:chunk_size]
#             _, predicted = torch.max(clean_outputs.data, 1)
#             total_clean += clean_targets.size(0)
#             correct_clean += (predicted == clean_targets).sum().item()
            
#     epoch_loss = running_loss / len(loader)
#     clean_acc_str = f"{(100 * correct_clean / total_clean):.2f}%" if total_clean > 0 else "N/A"
    
#     print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Batch Clean Acc: {clean_acc_str}")

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# EPOCHS = 30
# EPSILON = 8/255
# ALPHA = 2/255
# ITERS = 7

# print("Starting Adversarial Training...")
# model.train()

# for epoch in range(EPOCHS):
#     running_loss = 0.0
#     correct_clean = 0
#     total_clean = 0
    
#     for batch_idx, (batch_images, batch_labels) in enumerate(loader):
#         batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
#         # 50% chance for clean batch, 50% chance divided among attacks
#         attack_type = random.choices(
#             ['clean', 'fgsm','pgd'], 
#             weights=[0.5,  0.25, 0.25], 
#             k=1
#         )[0]
        
        
#         train_images = batch_images
#         if attack_type == 'fgsm':
#             train_images = fgsm_attack(model, batch_images, batch_labels, criterion, EPSILON)
#         elif attack_type == 'pgd':
#             train_images = pgd_attack(model, batch_images, batch_labels, criterion, EPSILON, ALPHA, ITERS)
            
#         # Forward pass
#         optimizer.zero_grad()
#         outputs = model(train_images)
#         loss = criterion(outputs, batch_labels)
        
#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
        
#         # Track training accuracy on clean batches to ensure we stay above 50% 
#         if attack_type == 'clean':
#             _, predicted = torch.max(outputs.data, 1)
#             total_clean += batch_labels.size(0)
#             correct_clean += (predicted == batch_labels).sum().item()
            
#     epoch_loss = running_loss / len(loader)
#     clean_acc_str = f"{(100 * correct_clean / total_clean):.2f}%" if total_clean > 0 else "N/A"
    
#     print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {epoch_loss:.4f} | Batch Clean Acc: {clean_acc_str}")


for epoch in range(EPOCHS):
    running_loss = 0.0
    correct_clean = 0
    total_clean = 0
    
    for batch_idx, (batch_images, batch_labels) in enumerate(loader):
        batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
        
        # 1. Randomly select an attack type (Clean is always included now)
        attack_type = random.choice([ 'pgd'])
        
        # 2. Generate the adversarial batch
        # if attack_type == 'fgsm':
        #     adv_images = fgsm_attack(model, batch_images, batch_labels, criterion, EPSILON)
        if attack_type == 'pgd':
            adv_images = pgd_attack(model, batch_images, batch_labels, criterion, EPSILON, ALPHA, ITERS)
            
        # 3. Concatenate clean images and adversarial images 
        #    (dim=0 stacks them along the batch dimension)
        train_images = torch.cat((batch_images, adv_images), dim=0)
        train_labels = torch.cat((batch_labels, batch_labels), dim=0)
        
        # Forward pass on the combined batch
        optimizer.zero_grad()
        outputs = model(train_images)
        loss = criterion(outputs, train_labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Note: running_loss now reflects the combined batch (clean + adv)
        running_loss += loss.item()
        
        # 4. Track training accuracy on clean images only
        # The clean images are the first half of the concatenated batch
        clean_outputs = outputs[:batch_images.size(0)]
        _, predicted = torch.max(clean_outputs.data, 1)
        total_clean += batch_labels.size(0)
        correct_clean += (predicted == batch_labels).sum().item()
            
    # Calculate averages
    epoch_loss = running_loss / len(loader)
    clean_acc_str = f"{(100 * correct_clean / total_clean):.2f}%" if total_clean > 0 else "N/A"
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | Combined Loss: {epoch_loss:.4f} | Batch Clean Acc: {clean_acc_str}")

# ==========================================
# 5. Save Submission
# ==========================================

print("Training complete. Saving model state dictionary...")
# Save only the state dict, not the full model instance [cite: 51, 59]
torch.save(model.state_dict(), "model.pt")
print("Saved to model.pt")