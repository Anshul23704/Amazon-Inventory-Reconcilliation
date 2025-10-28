import numpy as np
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
import os
import random
import PIL
import cv2
import seaborn as sns
import datetime
import sys
import gc

# Configuration
random.seed(229)
np.random.seed(229)

# Paths
img_dir = "D:/Sem5_Subjects/ML/MINIpORJ/Image-Inventory-Reconciliation-with-SVM-and-CNN-master/data/Allimages/"
meta_dir = "D:/Sem5_Subjects/ML/MINIpORJ/Image-Inventory-Reconciliation-with-SVM-and-CNN-master/data/Metadata/"
output_dir = "D:/Sem5_Subjects/ML/MINIpORJ/Image-Inventory-Reconciliation-with-SVM-and-CNN-master/hog_results/"
os.makedirs(output_dir, exist_ok=True)

# Logging
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(output_dir, f"hog_log_{timestamp}.txt")
log_file = open(log_filename, "w")
sys.stdout = log_file
sys.stderr = log_file

print("="*80)
print("HOG-based SVM for Image Counting")
print("="*80)

# Parameters
NUM_SAMPLES = 5000  # Try to get up to 5000 samples
MAX_QUANTITY = 5    # Only use images with ≤5 items
IMG_SIZE = 224      # Resize to 224x224
PPC = 16           # Pixels per cell for HOG
HOG_ORIENTATIONS = 8
HOG_CELLS_PER_BLOCK = (4, 4)

# NEW: Memory optimization parameters
USE_PCA = True  # Reduce dimensionality with PCA
PCA_COMPONENTS = 500  # Reduce from 15488 to 500 features
USE_FLOAT32 = True  # Use float32 instead of float64 (half the memory)

def extract_hog_features(img_path, use_edges=True):
    """
    Extract HOG features from an image
    
    Args:
        img_path: Path to image
        use_edges: If True, apply Laplacian edge detection before HOG
    
    Returns:
        HOG feature vector
    """
    try:
        # Load and resize image
        img = PIL.Image.open(img_path)
        resized_img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(resized_img)
        
        # Convert to grayscale
        data_gray = color.rgb2gray(arr)
        
        # Apply edge detection if requested
        if use_edges:
            data_gray = cv2.Laplacian(data_gray, cv2.CV_64F)
            data_gray = np.abs(data_gray)  # Take absolute value
            data_gray = data_gray / data_gray.max()  # Normalize to [0,1]
        
        # Extract HOG features
        fd, hog_image = hog(
            data_gray, 
            orientations=HOG_ORIENTATIONS, 
            pixels_per_cell=(PPC, PPC),
            cells_per_block=HOG_CELLS_PER_BLOCK,
            block_norm='L2',
            visualize=True
        )
        
        # Convert to float32 to save memory
        if USE_FLOAT32:
            fd = fd.astype(np.float32)
        
        return fd, True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None, False

def load_dataset():
    """
    Load dataset from counting_train.json and extract HOG features
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label vector (n_samples,)
        valid_files: List of valid image filenames
    """
    print("\n" + "="*80)
    print("Loading Dataset and Extracting HOG Features")
    print("="*80)
    
    # Load metadata
    train_json = "D:/Sem5_Subjects/ML/MINIpORJ/Image-Inventory-Reconciliation-with-SVM-and-CNN-master/counting_train.json"
    
    with open(train_json) as f:
        metadata = json.load(f)
    
    print(f"Total metadata entries: {len(metadata)}")
    
    # Sample random indices
    N = min(len(metadata), NUM_SAMPLES)
    dataset_indices = random.sample(range(len(metadata)), N)
    
    X_list = []
    y_list = []
    valid_files = []
    skipped = 0
    
    for idx in dataset_indices:
        img_id, quantity = metadata[idx]
        
        # Filter by quantity
        if quantity > MAX_QUANTITY:
            skipped += 1
            continue
        
        # Check if file exists
        file_name = f'{img_id+1:05d}.jpg'
        img_path = os.path.join(img_dir, file_name)
        
        if not os.path.isfile(img_path):
            skipped += 1
            continue
        
        # Extract HOG features
        features, success = extract_hog_features(img_path, use_edges=True)
        
        if success and features is not None:
            X_list.append(features)
            y_list.append(quantity)
            valid_files.append(file_name)
            
            if len(X_list) % 100 == 0:
                print(f"Processed {len(X_list)} images...")
    
    X = np.array(X_list, dtype=np.float32 if USE_FLOAT32 else np.float64)
    y = np.array(y_list)
    
    print(f"\nDataset Summary:")
    print(f"  Valid samples: {len(y)}")
    print(f"  Skipped: {skipped}")
    print(f"  Feature dimension: {X.shape[1]}")
    print(f"  Memory usage: {X.nbytes / (1024**2):.2f} MB")
    print(f"  Class distribution: {np.bincount(y)}")
    
    # Force garbage collection
    gc.collect()
    
    return X, y, valid_files

def plot_confusion_matrix(cm, classes, accuracy, save_path, normalize=False):
    """
    Plot and save confusion matrix
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'{title}\nAccuracy: {accuracy:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")

def train_and_evaluate():
    """
    Train SVM with grid search and evaluate
    """
    # Load data
    X, y, valid_files = load_dataset()
    
    if len(X) < 100:
        print("\nERROR: Not enough samples! Need at least 100.")
        return
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=229, stratify=y
    )
    
    print(f"\nTrain/Test Split:")
    print(f"  Training samples: {len(y_train)}")
    print(f"  Test samples: {len(y_test)}")
    print(f"  Training class distribution: {np.bincount(y_train)}")
    print(f"  Test class distribution: {np.bincount(y_test)}")
    
    # Apply PCA for dimensionality reduction
    if USE_PCA:
        print("\n" + "="*80)
        print("Applying PCA for Dimensionality Reduction")
        print("="*80)
        
        pca = PCA(n_components=PCA_COMPONENTS, random_state=229)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        explained_var = np.sum(pca.explained_variance_ratio_)
        print(f"  Reduced from {X.shape[1]} to {PCA_COMPONENTS} features")
        print(f"  Explained variance: {explained_var:.4f}")
        print(f"  New memory usage (train): {X_train.nbytes / (1024**2):.2f} MB")
        
        gc.collect()
    
    # Grid search for best parameters
    print("\n" + "="*80)
    print("Grid Search for Hyperparameters")
    print("="*80)
    
    # Reduced parameter grid to search fewer combinations
    param_grid = {
        'C': [0.1, 1, 10, 100],  # Reduced from 6 to 4 values
        'gamma': ['scale', 'auto', 0.001, 0.01]  # Reduced from 6 to 4 values
    }
    
    # Try both RBF and Poly kernels
    for kernel in ['rbf', 'poly']:
        print(f"\n--- Testing {kernel.upper()} kernel ---")
        
        grid_search = GridSearchCV(
            svm.SVC(kernel=kernel, cache_size=500),  # Increased cache size
            param_grid,
            cv=3,
            scoring='accuracy',
            verbose=2,
            n_jobs=2,  # CHANGED: Use only 2 parallel jobs instead of -1
            pre_dispatch='2*n_jobs'  # Limit pre-dispatched jobs
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters for {kernel}: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate on test set
        y_pred = grid_search.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{kernel.upper()} Kernel Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrices
        classes = sorted(np.unique(y))
        
        cm_path = os.path.join(output_dir, f'cm_{kernel}_{test_accuracy:.4f}.png')
        plot_confusion_matrix(cm, classes, test_accuracy, cm_path, normalize=False)
        
        cm_norm_path = os.path.join(output_dir, f'cm_{kernel}_normalized_{test_accuracy:.4f}.png')
        plot_confusion_matrix(cm, classes, test_accuracy, cm_norm_path, normalize=True)
        
        print("\n" + "-"*80)
        
        # Force garbage collection between kernels
        gc.collect()

# Run the pipeline
if __name__ == "__main__":
    try:
        train_and_evaluate()
        print("\n" + "="*80)
        print("DONE! Check results in:", output_dir)
        print("="*80)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log_file.close()