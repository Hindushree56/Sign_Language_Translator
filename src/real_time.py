import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import string

# ================================
# Configuration
# ================================
MODEL_PATH = "models/asl_cnn.keras"  # or "models/asl_cnn.keras"
IMG_SIZE = 64
LETTERS = list(string.ascii_uppercase) + ['del', 'nothing', 'space']

# ================================
# Load the trained model
# ================================
print("🚀 Starting Real-Time ASL Translator...")
print("📦 Loading model...")

try:
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading model:", e)
    exit()

# ================================
# Initialize webcam
# ================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible! Close other camera apps and try again.")
    exit()
print("🎥 Webcam started. Press 'q' to quit, 'c' to clear text.\n")

# ================================
# Word formation & stability buffer
# ================================
prediction_buffer = deque(maxlen=20)
predicted_text = ""

# ================================
# Main Loop
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame not captured. Exiting...")
        break

    # Flip horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Define Region of Interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi_normalized = roi_resized.astype("float32") / 255.0
    roi_expanded = np.expand_dims(roi_normalized, axis=0)

    # Predict gesture
    prediction = model.predict(roi_expanded)
    label_index = np.argmax(prediction)
    confidence = np.max(prediction)
    label = LETTERS[label_index]

    # Smooth predictions using a buffer
    prediction_buffer.append(label)
    smoothed_label = max(set(prediction_buffer), key=prediction_buffer.count)

    # Build the word
    if smoothed_label == 'space':
        predicted_text += ' '
    elif smoothed_label == 'del':
        predicted_text = predicted_text[:-1]
    elif smoothed_label != 'nothing':
        predicted_text += smoothed_label

    # Display predictions
    cv2.putText(frame, f"Prediction: {label} ({confidence*100:.1f}%)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(frame, f"Detected: {smoothed_label}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Word: {predicted_text}",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("🤖 Real-Time ASL Translator", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n👋 Exiting Translator. Goodbye!")
        break
    elif key == ord('c'):
        predicted_text = ""
        print("🧹 Cleared text.")

cap.release()
cv2.destroyAllWindows()
