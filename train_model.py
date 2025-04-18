import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, Dense, TimeDistributed, Input
from tensorflow.keras.utils import to_categorical

# --- CONFIG ---
IMG_SIZE = 64
SEQ_LENGTH = 10
CLASSES = ['A', 'D', 'M', 'N', 'O', 'S', 'T', 'U', 'V', 'Y', 'blank']

BASE_PATH = r"C:\Users\mrrah\Pictures\Pictures\capstone"
DATASET_PATH = os.path.join(BASE_PATH, "SignLaguageMNST")
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")
MODEL_OUTPUT_PATH = os.path.join(BASE_PATH, "sign_language_model.h5")

# --- LOAD ---
def load_images_from_folder(folder):
    image_files = glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.jpeg"))
    images = []
    for filename in sorted(image_files):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
    return np.array(images, dtype=np.uint8)

def load_dataset(base_path):
    data, labels = [], []
    for idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(base_path, class_name)
        if os.path.isdir(class_path):
            images = load_images_from_folder(class_path)
            data.extend(images)
            labels.extend([idx] * len(images))
    return np.array(data), np.array(labels)

def create_sequences(data, labels, seq_length):
    sequences, seq_labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length].astype(np.float32) / 255.0
        sequences.append(seq)
        seq_labels.append(labels[i + seq_length - 1])
    return np.array(sequences, dtype=np.float32), np.array(seq_labels)

print("🔄 Loading data...")
X_train, y_train = load_dataset(TRAIN_PATH)
X_test, y_test = load_dataset(VAL_PATH)

print(f"✅ Loaded {len(X_train)} train and {len(X_test)} test samples.")
print(f"📚 Classes: {CLASSES}")

X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

y_train = to_categorical(y_train, num_classes=len(CLASSES))
y_test = to_categorical(y_test, num_classes=len(CLASSES))

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LENGTH)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, SEQ_LENGTH)

print("📦 Train Sequence Shape:", X_train_seq.shape)
print("📦 Test Sequence Shape:", X_test_seq.shape)

# --- MODEL ---
print("🧠 Building model...")
model = Sequential([
    Input(shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 1)),
    TimeDistributed(Conv2D(32, (3, 3), activation='relu')),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("🚀 Training...")
model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_test_seq, y_test_seq),
    epochs=10,
    batch_size=32
)

model.save(MODEL_OUTPUT_PATH)
print(f"✅ Model saved to {MODEL_OUTPUT_PATH}")
