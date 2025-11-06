import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================
# Configuration
# ===============================
DATA_DIR = "data_split"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20

# ===============================
# Dataset check
# ===============================
if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError("❌ Training data not found! Run prepare_data.py first.")

NUM_CLASSES = len(os.listdir(TRAIN_DIR))
print(f"✅ Number of Classes Detected: {NUM_CLASSES}")

# ===============================
# Data Augmentation
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_flow = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_flow = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ===============================
# Model Definition (CNN)
# ===============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================
# Callbacks (save best model + stop early)
# ===============================
os.makedirs("models", exist_ok=True)

checkpoint = ModelCheckpoint(
    "models/asl_cnn_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ===============================
# Train the Model
# ===============================
print("\n🚀 Starting Training...\n")

history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print("\n✅ Training Completed Successfully!")

# ===============================
# Save the Model (both formats)
# ===============================
print("\n💾 Saving the model...")
model.save("models/asl_cnn.h5")      # Legacy format
model.save("models/asl_cnn.keras")   # New format
print("✅ Model saved successfully as both .h5 and .keras formats!")

# ===============================
# Evaluate Model (optional)
# ===============================
val_loss, val_acc = model.evaluate(val_flow)
print(f"\n📊 Validation Accuracy: {val_acc*100:.2f}%")
print(f"📉 Validation Loss: {val_loss:.4f}")

print("\n🎯 All done! You can now run: python src\\real_time.py")
