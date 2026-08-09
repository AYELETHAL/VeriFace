import os
import cv2
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# Settings
DATADIR = r"C:\Users\97258\engineering_try_2\Engineering-Project"
CATEGORIES = [
    "No_Manipulation",
    "output_faces_change_lip_color_no_padding",
    "output_faces_change_eye_color_no_padding",
    "output_faces_change_nose_no_padding",
    "output_faces_change_eyebrows_no_padding"
]

IMG_SIZE = 128

# Extract HOG features from image
def get_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    
    return features


# Load data
def load_data():
    data = []
    labels = []
    
    print("Loading images and extracting HOG features...")
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        
        if not os.path.exists(path):
            print(f"Warning: Directory {path} not found.")
            continue
        
        count = 0
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                # Resize image
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                
                # Extract HOG features
                features = get_hog_features(image)
                
                data.append(features)
                labels.append(class_num)
                count += 1
                
            except Exception as e:
                pass
        
        print(f"Loaded {count} images from {category}")
    
    return np.array(data), np.array(labels)


# Main execution
print("=" * 60)
print("SVM with HOG Features (No Padding)")
print("=" * 60)

X, y = load_data()

print(f"\nTotal samples: {len(X)}")
print(f"Feature vector size: {X.shape[1]}")

if len(X) > 0:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create and train SVM model
    # 1. Define the Pipeline
    # We scale FIRST, then apply PCA, then train SVM
    pipe = Pipeline([
        ('pca', PCA()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale'))
    ])

    # 2. Define the range of PCA components to test
    # Let's test from 10 components up to a reasonable fraction of your features
    n_features = X_train.shape[1]
    param_grid = {
        'pca__n_components': [10, 50, 100, 200, 300, 500] 
    }

    # 3. Run GridSearch
    print("\nOptimizing PCA components...")
    grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid.fit(X_train, y_train)

    # 4. Extract results for plotting
    results = grid.cv_results_
    pca_components = param_grid['pca__n_components']
    mean_scores = results['mean_test_score']

    # 5. Plot Accuracy vs. PCA Components
    plt.figure(figsize=(10, 6))
    plt.plot(pca_components, mean_scores, marker='o', linestyle='-', color='b')
    plt.title('SVM Accuracy vs. Number of PCA Components')
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Mean CV Accuracy')
    plt.grid(True)
    plt.show()

    print(f"\nBest number of components: {grid.best_params_['pca__n_components']}")
    print(f"Best CV Accuracy: {grid.best_score_ * 100:.2f}%")

    # Final Model Evaluation
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    print(f"Final Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # 1. Define your short labels
    short_labels = ["Original", "Lip", "Eye", "Nose", "Eyebrows"]

    plt.subplot(1, 2, 2)
    cm = confusion_matrix(y_test, y_pred)

    # 2. Use short_labels for the ticklabels
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=short_labels, 
                yticklabels=short_labels)

    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.tight_layout()
    plt.show()

    import joblib

    joblib.dump(best_model, "svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(grid.best_estimator_.named_steps['pca'], "pca.pkl")


    print("Model saved!")
else:
    print("No images found!")