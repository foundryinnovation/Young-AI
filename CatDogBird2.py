#the one
# WOW MOMENT: Importing the Magic Tools
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Load the CSV file containing labels
csv_file = "/content/doodle_labels.csv"
data = pd.read_csv(csv_file)

# Step 2: Define a function to load and process the images
def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert image to grayscale
        img = img.resize((64, 64))  # Resize to 64x64 pixels
        return np.array(img).flatten()  # Flatten the image to a 1D array

# Step 3: Prepare the dataset by loading images and creating the feature matrix
X = []  # Feature matrix (images)
y = []  # Labels

# Iterate through the CSV and load images
for index, row in data.iterrows():
    image_path = f"/content/doodles/{row['label']}/{row['image_name']}"
    if os.path.exists(image_path):
        X.append(load_image(image_path))
        y.append(row['label'])
    else:
        print(f"Warning: {image_path} does not exist.")

# Convert to numpy arrays for use in scikit-learn
X = np.array(X)
y = np.array(y)

# Step 4: Split the data into training and testing sets
# Using random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)

# WOW MOMENT: Visualizing Training Images
# Let's display some training images to see what the computer is learning from
num_train_samples = 5
fig, axes = plt.subplots(1, num_train_samples, figsize=(15, 3))
for i in range(num_train_samples):
    img = X_train[i].reshape(64, 64)
    label = y_train[i]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Train Label: {label}")
    axes[i].axis('off')
plt.show()

# Step 5: Train a Random Forest Classifier
# WOW MOMENT: The Computer Learns with a Forest of Decision Trees!
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Step 6: Evaluate the model
accuracy = classifier.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Display multiple predictions vs. actuals with prediction probabilities
# WOW MOMENT: See How Sure the Computer Is About Its Guesses!

num_samples = 5  # Number of samples to display
fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))

for i in range(num_samples):
    # Get the test image and reshape it
    img = X_test[i].reshape(64, 64)
    actual_label = y_test[i]
    predicted_label = classifier.predict([X_test[i]])[0]
    # Get prediction probabilities
    probabilities = classifier.predict_proba([X_test[i]])[0]
    # Map probabilities to class labels
    class_probabilities = dict(zip(classifier.classes_, probabilities))
    # Sort classes by probability
    sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
    # Prepare the probability string
    prob_text = '\n'.join([f"{cls}: {prob:.2f}" for cls, prob in sorted_probs])

    # Display the image
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
    # Display probabilities below the image
    axes[i].set_xlabel(prob_text, fontsize=10)
    axes[i].set_xticks([])
    axes[i].set_yticks([])

plt.tight_layout()
plt.show()

# WOW MOMENT: Visualizing Testing Images
# Let's display some testing images to see what the computer is being tested on
num_test_samples = 5
fig, axes = plt.subplots(1, num_test_samples, figsize=(15, 3))
for i in range(num_test_samples):
    img = X_test[i].reshape(64, 64)
    label = y_test[i]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Test Label: {label}")
    axes[i].axis('off')
plt.show()

# Step 8: Capture a New Drawing Using the Camera
# WOW MOMENT: Show Your Drawing to the Computer Live!
from IPython.display import display, Javascript
from google.colab.output import eval_js
import cv2
import base64

def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture Photo';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to the video size
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture button to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = base64.b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

try:
    # Capture photo
    photo_filename = take_photo()
    print('Photo saved to {}'.format(photo_filename))

    # Load and preprocess the captured image
    new_image_raw = load_image(photo_filename)
    new_image = scaler.transform([new_image_raw])  # Scale the image

    # Use the classifier to predict the label
    new_prediction = classifier.predict(new_image)[0]
    # Get prediction probabilities
    probabilities = classifier.predict_proba(new_image)[0]
    class_probabilities = dict(zip(classifier.classes_, probabilities))
    sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
    prob_text = '\n'.join([f"{cls}: {prob:.2f}" for cls, prob in sorted_probs])

    print(f"The AI thinks your drawing is a: {new_prediction}")

    # Display the image
    img = new_image_raw.reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {new_prediction}")
    plt.xlabel(prob_text, fontsize=10)
    plt.axis('off')
    plt.show()

except Exception as e:
    print("An error occurred while capturing the photo: ", e)


# Step 8: Upload a New Image File and Predict
# WOW MOMENT: Upload Your Drawing and Let the Computer Guess
from google.colab import files

# Upload a new image file
uploaded = files.upload()

for filename in uploaded.keys():
    # Load and preprocess the new image
    new_image = load_image(filename)

    # Use the classifier to predict the label
    new_prediction = classifier.predict([new_image])[0]
    # Get prediction probabilities
    probabilities = classifier.predict_proba([new_image])[0]
    class_probabilities = dict(zip(classifier.classes_, probabilities))
    sorted_probs = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
    prob_text = '\n'.join([f"{cls}: {prob:.2f}" for cls, prob in sorted_probs])

    print(f"The AI thinks your uploaded drawing '{filename}' is a: {new_prediction}")

    # Display the image
    img = new_image.reshape(64, 64)
    plt.imshow(img, cmap='gray')
    plt.title(f"Prediction: {new_prediction}")
    plt.xlabel(prob_text, fontsize=10)
    plt.axis('off')
    plt.show()
