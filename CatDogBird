# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn import svm
from PIL import Image
import matplotlib.pyplot as plt

# Step 1: Load the CSV file containing labels
csv_file = "/content/doodle_labels.csv"
data = pd.read_csv(csv_file)

# Preview the first few rows of the CSV
print("Data Preview:")
print(data.head())

# Step 2: Define a function to load and process the images
def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert image to grayscale
        img = img.resize((64, 64))  # Resize to 64x64 pixels for consistency
        return np.array(img).flatten()  # Flatten to a single array for the classifier

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train a Support Vector Machine (SVM) classifier
classifier = svm.SVC(gamma='auto')
classifier.fit(X_train, y_train)

# Step 6: Evaluate the model
accuracy = classifier.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Step 7: Test the model with a new doodle (optional)
# Let's pick a sample from the test set and visualize it
sample_index = 0  # You can change this index to see other images
sample_image = X_test[sample_index].reshape(64, 64)  # Reshape back to 64x64 image
sample_label = y_test[sample_index]

# Predict using the classifier
predicted_label = classifier.predict([X_test[sample_index]])

# Display the test image and prediction
plt.imshow(sample_image, cmap='gray')
plt.title(f"True Label: {sample_label}, Predicted: {predicted_label[0]}")
plt.axis('off')
plt.show()

# Step 8: Predict on a New Image (Optional)
# Upload a new doodle to /content/new_doodle.png and test it
new_image_path = "/content/new_doodle.png"  # Change this to the new image's path if testing
if os.path.exists(new_image_path):
    new_image = load_image(new_image_path)  # Load and preprocess the image
    new_prediction = classifier.predict([new_image])  # Predict the label
    print(f"The AI thinks your new doodle is a: {new_prediction[0]}")
else:
    print(f"New image '{new_image_path}' not found for testing.")
