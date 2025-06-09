import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import matplotlib.pyplot as plt

# Constants
DATA_DIR = "C:/Users/oktay/AILab/Project/gesture_data"
LABELS = ["up", "down", "left", "right"]
NUM_CLASSES = len(LABELS)
label_map = {label: idx for idx, label in enumerate(LABELS)}

# Mediapipe setup
mp_hands = mp.solutions.hands

def extract_landmarks(image):
    if image is None:
        print("❌ Input image is None")
        return None

    print(f"Image shape: {image.shape}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Try with lower detection confidence
    with mp_hands.Hands(static_image_mode=True, 
                       max_num_hands=1,
                       min_detection_confidence=0.3,  # Lowered from 0.5
                       min_tracking_confidence=0.3) as hands:  # Lowered from 0.5
        result = hands.process(image_rgb)
        
        if not result:
            print("❌ No result from hand detection")
            return None
            
        if not result.multi_hand_landmarks:

            # Save the image for debugging
            debug_dir = os.path.join(DATA_DIR, "debug_images")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "no_landmarks.jpg")
            cv2.imwrite(debug_path, image)
            print(f"  Saved image to {debug_path} for debugging")
            return None
        
        if result.multi_hand_landmarks:
            print("✅ Hand detected!")
            landmarks = result.multi_hand_landmarks[0]
            wrist = landmarks.landmark[0]
            index_tip = landmarks.landmark[8]
            index_mcp = landmarks.landmark[5]
            middle_tip = landmarks.landmark[12]
            ring_tip = landmarks.landmark[16]
            pinky_tip = landmarks.landmark[20]

            # Print landmark positions for debugging
            print(f"Wrist position: x={wrist.x:.3f}, y={wrist.y:.3f}, z={wrist.z:.3f}")
            print(f"Index tip position: x={index_tip.x:.3f}, y={index_tip.y:.3f}, z={index_tip.z:.3f}")

            index_dx = index_tip.x - index_mcp.x
            index_dy = index_tip.y - index_mcp.y
            index_angle = np.arctan2(index_dy, index_dx)

            horizontal_component = np.cos(index_angle)
            vertical_component = np.sin(index_angle)

            normalized_landmarks = []
            for lm in landmarks.landmark:
                x = lm.x - wrist.x
                y = lm.y - wrist.y
                z = lm.z - wrist.z
                normalized_landmarks.extend([x, y, z])

            normalized_landmarks.extend([
                index_dx,
                index_dy,
                index_angle,
                horizontal_component,
                vertical_component,
                index_tip.x - wrist.x,
                index_tip.y - wrist.y,
                (index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 4 - wrist.x,
                (index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 4 - wrist.y
            ])

            return np.array(normalized_landmarks)

    return None

# Load dataset
X, y = [], []

print("\nLoading data from directories:")
for label in LABELS:
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        print(f"❌ Directory not found: {folder}")
        continue

    files = os.listdir(folder)

    print(f"\n{label} directory:")
    print(f"Found {len(files)} files")

    for file in files:
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            print(f"❌ Failed to read image: {file}")
            continue

        print(f"\nProcessing {file}:")
        features = extract_landmarks(img)
        
        if features is not None:
            X.append(features)
            y.append(label_map[label])  # ✅ Use folder name as ground truth

            # Augmentations
            X.append(features + np.random.normal(0, 0.01, features.shape))
            y.append(label_map[label])

            X.append(features + np.random.normal(0, 0.02, features.shape))
            y.append(label_map[label])
                  
            X.append(features * np.random.uniform(0.95, 1.05))
            features * np.random.uniform(0.95, 1.05)
            y.append(label_map[label])

        else:
            print(f"❌ No landmarks found in {file}, deleting it")
            try:
                os.remove(path)
            except:
                print(f"  Failed to delete {file}")

if not X:
    print("❌ No valid samples found in any directory!")
    exit()

X = np.array(X)
y = to_categorical(np.array(y), num_classes=NUM_CLASSES)
class_indices = np.argmax(y, axis=1)
class_counts = Counter(class_indices)

print("\nInitial samples per class (after augmentation):")
for label, idx in label_map.items():
    print(f"{label}: {class_counts.get(idx, 0)} samples")

# Calculate class weights
total_samples = len(class_indices)
class_weights = {i: total_samples / (len(LABELS) * count) for i, count in class_counts.items()}
print("\nClass weights for imbalanced data:")
for label, idx in label_map.items():
    print(f"{label}: {class_weights.get(idx, 0):.2f}")

# Filter out classes with too few samples
valid_classes = [cls for cls, count in class_counts.items() if count >= 2]
if len(valid_classes) < 2:
    print("\n❌ Not enough valid classes with sufficient samples to train.")
    exit()

mask = np.isin(class_indices, valid_classes)
X = X[mask]
y = y[mask]
class_indices = np.argmax(y, axis=1)

print("\nFiltered samples per class:")
for label, idx in label_map.items():
    print(f"{label}: {Counter(class_indices).get(idx, 0)} samples")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=class_indices, random_state=42)

model = Sequential([
    Dense(1024, activation='relu', input_shape=(72,)),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.4),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASSES, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint("C:/Users/oktay/AILab/Project/gesture_landmark_model_best.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

print("\nStarting training...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
                    callbacks=[early_stopping, reduce_lr, checkpoint], class_weight=class_weights, verbose=1)

model.save("C:/Users/oktay/AILab/Project/gesture_landmark_model_v4.h5")
print("✅ Model saved as 'gesture_landmark_model_v4.h5'")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('C:/Users/oktay/AILab/Project/training_history.png')
plt.close()

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=LABELS))
