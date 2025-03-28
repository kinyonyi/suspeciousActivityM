import cv2
import numpy as np
import tensorflow as tf 

# Load the .h5 model
model = tf.keras.models.load_model("suspicious_activity_model.h5")
labels = ['normal', 'fighting', 'pointing_gun', 'assault', 'robbery', 'explosion']

cap = cv2.VideoCapture(0)
target_size = (224, 224)  # Match for model's input shape

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess activity
    img = cv2.resize(frame, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict activity
    predictions = model.predict(img)
    pred_index = np.argmax(predictions)
    label = labels[pred_index]
    confidence = np.max(predictions)

    # Display
    color = (0, 255, 0) if label == "normal" else (0, 0, 255)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()