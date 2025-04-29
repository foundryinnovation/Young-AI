import cv2
import numpy as np
import tensorflow as tf  # This includes the TFLite interpreter
from djitellopy import Tello
from PIL import Image, ImageDraw, ImageFont
import time

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model_unquanterik.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define input size (based on your model's input shape)
IMG_SIZE = (224, 224)  # Adjust this if necessary

# Load labels from the provided file
labels = ['Nothing', 'TakeOff', 'Turn1', 'Land', 'Turn2']  # These are your provided labels

# Initialize the Tello drone
tello = Tello()
tello.connect()

# Start video stream from the Tello
tello.streamon()

def function():
    print("hi")

# Function to make predictions using the TFLite model
def predict(frame):
    # Convert the frame to RGB and resize to model input size
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize(IMG_SIZE)

    # Preprocess the image (normalize, reshape)
    image_array = np.asarray(image).astype('float32')
    image_array = image_array / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get the output tensor and find the predicted class
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)

    return labels[predicted_class[0]]


# Function to send commands to the Tello drone based on predictions
def execute_command(prediction):
    if prediction == 'Takeoff':
        if not tello.is_flying:
            tello.takeoff()
            print("Drone taking off")
    elif prediction == 'Land':
        if tello.is_flying:
            tello.land()
            print("Drone landing")
    elif prediction == 'Turn1':
        if tello.is_flying:
            #tello.move_up(30)
            tello.rotate_clockwise(90)  # Adjust the degree of rotation as needed
            print("Drone turning")
    elif prediction == 'Turn2':
        if tello.is_flying:
                # tello.move_up(30)
            tello.rotate_clockwise(-90)  # Adjust the degree of rotation as needed
            print("Drone turning (2)")
    else:
        print("No action (Nothing)")


# Main loop for processing video and sending commands
while True:
    # Capture frame from the Tello drone's video feed
    frame_read = tello.get_frame_read()
    frame = frame_read.frame

    # Resize the frame and make prediction
    prediction = predict(frame)

    # Execute the corresponding command based on prediction
    execute_command(prediction)

    # Overlay the prediction text on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Prediction: {prediction}', (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame with the overlay
    cv2.imshow('Tello Video Feed', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
tello.streamoff()
tello.end()
