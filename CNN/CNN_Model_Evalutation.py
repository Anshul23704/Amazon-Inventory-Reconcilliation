import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
MODEL_PATH = "C:/Users/harry/snapshots/resnet34_1_best.pth.tar"
IMAGES_DIR = "D:/Sem5_Subjects/ML/MINIpORJ/Image-Inventory-Reconciliation-with-SVM-and-CNN-master/data/Allimages"
METADATA_DIR = "D:/Sem5_Subjects/ML/MINIpORJ/Image-Inventory-Reconciliation-with-SVM-and-CNN-master/data/metadata"  
NUM_CLASSES = 6
BATCH_SIZE = 32
OUTPUT_DIR = "C:/Users/harry/evaluation_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---
## 1. Parse JSON Files and Match with Images

class LabeledImageDataset(Dataset):
    """Dataset that loads images with labels from JSON metadata"""
    
    def __init__(self, images_dir, metadata_dir, transform=None):
        self.images_dir = images_dir
        self.metadata_dir = metadata_dir
        self.transform = transform
        self.samples = []
        
        print(f"\nScanning metadata directory: {metadata_dir}")
        
        # Check if metadata directory exists
        if not os.path.exists(metadata_dir):
            print(f"ERROR: Metadata directory not found!")
            return
        
        # Get all JSON files
        json_files = [f for f in os.listdir(metadata_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files")
        
        # Parse each JSON and match with image
        matched = 0
        skipped = 0
        
        for json_file in json_files:
            json_path = os.path.join(metadata_dir, json_file)
            
            # Try to find corresponding image (same name but different extension)
            base_name = os.path.splitext(json_file)[0]
            
            # Try common image extensions
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = os.path.join(images_dir, base_name + ext)
                if os.path.exists(potential_path):
                    image_path = potential_path
                    break
            
            if image_path is None:
                skipped += 1
                continue
            
            # Read JSON to get label
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                
                # Try to extract label - adjust these keys based on your JSON structure
                label = None
                
                # Common label key names - try these
                for key in ['label', 'class', 'category', 'class_id', 'label_id', 
                           'classification', 'ground_truth', 'true_label']:
                    if key in metadata:
                        label = metadata[key]
                        break
                
                # If label is still None, print the JSON structure for debugging
                if label is None and matched == 0:
                    print(f"\nSample JSON structure from {json_file}:")
                    print(json.dumps(metadata, indent=2)[:500])
                    print("\nPlease check the JSON structure and update the label extraction code.")
                    print("Common keys to look for: 'label', 'class', 'category', 'class_id'\n")
                    continue
                
                # Convert label to integer if needed
                if isinstance(label, str):
                    try:
                        label = int(label)
                    except ValueError:
                        # If it's a string class name, you might need a mapping
                        print(f"Warning: String label '{label}' found. Skipping.")
                        skipped += 1
                        continue
                
                # Validate label is in valid range
                if label is not None and 0 <= label < NUM_CLASSES:
                    self.samples.append({
                        'image_path': image_path,
                        'label': label,
                        'json_path': json_path
                    })
                    matched += 1
                else:
                    skipped += 1
                    
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                skipped += 1
                continue
        
        print(f"\nMatched {matched} images with labels")
        print(f"Skipped {skipped} files (no image match or invalid label)")
        
        if matched == 0:
            print("\nERROR: No valid image-label pairs found!")
            print("Please check:")
            print("1. JSON files are in the metadata directory")
            print("2. Corresponding images exist in the images directory")
            print("3. JSON files contain a 'label' or 'class' field")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, sample['label']
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Return a black image as fallback
            if self.transform:
                return self.transform(Image.new('RGB', (224, 224))), sample['label']
            return Image.new('RGB', (224, 224)), sample['label']

# ---
## 2. Load Model

print("\nLoading model...")
model = models.resnet34(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("Model loaded successfully!")

# ---
## 3. Prepare Data

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

eval_dataset = LabeledImageDataset(
    images_dir=IMAGES_DIR,
    metadata_dir=METADATA_DIR,
    transform=eval_transforms
)

if len(eval_dataset) == 0:
    print("\nNo labeled data found. Cannot evaluate.")
    print(f"Please check that:")
    print(f"  1. Metadata directory exists: {METADATA_DIR}")
    print(f"  2. JSON files contain label information")
    print(f"  3. Corresponding images exist in: {IMAGES_DIR}")
    exit()

eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---
## 4. Evaluate Model

print(f"\nEvaluating on {len(eval_dataset)} labeled images...")

all_predictions = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        outputs = model(images)
        
        probabilities = torch.softmax(outputs, dim=1)
        confidences, predictions = torch.max(probabilities, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confidences.extend(confidences.cpu().numpy())
        
        if (batch_idx + 1) % 5 == 0:
            print(f"Processed {(batch_idx + 1) * BATCH_SIZE} / {len(eval_dataset)} images")

# ---
## 5. Calculate Metrics

print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Overall Accuracy
accuracy = accuracy_score(all_labels, all_predictions)
print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}%")

# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    all_labels, all_predictions, average=None, zero_division=0
)

print(f"\n📈 Per-Class Metrics:")
print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 60)
for i in range(NUM_CLASSES):
    print(f"{i:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}")

# Macro and weighted averages
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    all_labels, all_predictions, average='macro', zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    all_labels, all_predictions, average='weighted', zero_division=0
)

print(f"\n📊 Average Metrics:")
print(f"  Macro Average    - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
print(f"  Weighted Average - Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")

# Confidence statistics
print(f"\n🎯 Prediction Confidence:")
print(f"  Mean: {np.mean(all_confidences):.4f}")
print(f"  Median: {np.median(all_confidences):.4f}")
print(f"  Std Dev: {np.std(all_confidences):.4f}")

# ---
## 6. Confusion Matrix

cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(NUM_CLASSES),
            yticklabels=range(NUM_CLASSES))
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()

cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"\n💾 Confusion matrix saved to: {cm_path}")
plt.close()

# ---
## 7. Save Detailed Report

report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.txt')
with open(report_path, 'w') as f:
    f.write("="*60 + "\n")
    f.write("MODEL EVALUATION REPORT\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Images: {len(eval_dataset)}\n")
    f.write(f"Number of Classes: {NUM_CLASSES}\n\n")
    
    f.write(f"Overall Accuracy: {accuracy*100:.2f}%\n\n")
    
    f.write("Per-Class Metrics:\n")
    f.write("-" * 60 + "\n")
    f.write(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
    f.write("-" * 60 + "\n")
    for i in range(NUM_CLASSES):
        f.write(f"{i:<8} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10.0f}\n")
    
    f.write(f"\nConfusion Matrix:\n")
    f.write(str(cm))

print(f"💾 Detailed report saved to: {report_path}")
print("\n" + "="*60)