import cv2
import numpy as np
import requests
import time

# If using the official TensorFlow installation (which includes TF Lite):
#   import tensorflow as tf
#   interpreter = tf.lite.Interpreter(model_path='model.tflite')
# If using tflite_runtime (a smaller package), do:
#   import tflite_runtime.interpreter as tflite
#   interpreter = tflite.Interpreter(model_path='model.tflite')

import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path='model_test.tflite')
interpreter.allocate_tensors()

# Get model input / output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Replace with your ESP32's IP address
ESP32_IP = "192.168.100.147"  # e.g., 192.168.1.100


# A small helper to send commands to the ESP32
def send_command(direction):
    """
    direction should be "forward" or "backward"
    """
    url = f"http://{ESP32_IP}/move"
    params = {"direction": direction}
    try:
        r = requests.get(url, params=params, timeout=2)
        print(f"Sent {direction}, response: {r.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending command: {e}")


# Suppose your model has two classes: 0 -> forward, 1 -> backward
labels = ["forward", "backward","nothing"]  # Adjust to your model's actual labels

# Capture from default webcam (index=0). Change if you have multiple cameras.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Preprocess the frame for your model
        # Example: resize to 224x224, normalize to [0,1], etc.
        # Adapt this exactly to how your model was trained!
        resized_frame = cv2.resize(frame, (224, 224))
        # Convert to float32 if your model expects float
        input_data = np.array(resized_frame, dtype=np.float32)
        # Normalize if needed (example)
        input_data = input_data / 255.0
        # Add batch dimension [1, height, width, channels]
        input_data = np.expand_dims(input_data, axis=0)

        # Set the tensor to the input data
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Depending on your model, output_data could be:
        # - A single integer class index
        # - A probability distribution over classes
        # For demonstration, let's assume it's a probability distribution:
        # output_data.shape might be [1, 2] for 2 classes
        predicted_class = np.argmax(output_data[0])

        label = labels[predicted_class]
        print(f"Predicted label: {label}")

        # Send command to the ESP32 if you want to move
        if label == "forward":
            send_command("forward")
        elif label == "backward":
            send_command("backward")
        elif label == "nothing":
            send_command("nothing")

        # Show the webcam feed (optional)
        cv2.imshow('frame', frame)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Sleep a bit so as not to spam the ESP32 with requests on every frame
        time.sleep(1.0)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()

