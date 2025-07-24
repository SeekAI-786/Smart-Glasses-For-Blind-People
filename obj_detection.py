import cv2
import numpy as np
import pyttsx3
import time
import tflite_runtime.interpreter as tflite

# Load labels
with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Voice
engine = pyttsx3.init()
last_label = ""
frame_count = 0

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (width, height))
    input_data = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    label_id = np.argmax(output_data)
    confidence = output_data[label_id]
    label = labels[label_id]

    # Display
    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Teachable Pi", frame)

    # Speak if confidence is high and new label
    if frame_count % 10 == 0 and confidence > 0.7 and label != last_label:
        engine.say(f"{label} detected")
        engine.runAndWait()
        last_label = label

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
