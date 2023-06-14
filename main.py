import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import imgaug.augmenters as iaa
import joblib

# Path to the directory containing the flag images
flag_dir = 'flag_images'

# Path to the directory containing the non-flag images
non_flag_dir = 'noflag_images'

# Create empty lists to store the preprocessed images and labels
images = []
labels = []

# Define the image augmentation sequence
augmentation_seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip images horizontally with a probability of 0.5
    iaa.Affine(rotate=(-10, 10)),  # Rotate images by -10 to 10 degrees
    iaa.GaussianBlur(sigma=(0, 1.0))  # Apply Gaussian blur with a sigma between 0 and 1.0
])

# Function to preprocess a batch of images
def preprocess_batch(image_paths, batch_labels):
    preprocessed_images = []
    for image_path in image_paths:
        if not os.path.isfile(image_path):
            print(f"Invalid image file: {image_path}")
            continue
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read image: {image_path}")
            continue
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preprocessed_images.append(image)
    augmented_images = augmentation_seq(images=preprocessed_images)
    images.extend(augmented_images)
    labels.extend([batch_labels] * len(augmented_images))  # Extend with a list of batch_labels


# Process the flag images
for filename in os.listdir(flag_dir):
    image_path = os.path.join(flag_dir, filename)
    preprocess_batch([image_path], 1)  # Label 1 represents flag

# Process the non-flag images
for filename in os.listdir(non_flag_dir):
    image_path = os.path.join(non_flag_dir, filename)
    preprocess_batch([image_path], 0)  # Label 0 represents non-flag

# Convert the lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Shuffle the data to avoid any order bias
random_indices = np.random.permutation(len(images))
images = images[random_indices]
labels = labels[random_indices]

# Split the dataset into training and testing sets (80% for training, 20% for testing)
split_index = int(0.8 * len(images))
train_images = images[:split_index]
train_labels = labels[:split_index]
test_images = images[split_index:]
test_labels = labels[split_index:]

# Flatten the images (convert from 3D to 2D array)
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# Create an SVM classifier
svm_classifier = SVC()

# Train the SVM classifier on the training set
svm_classifier.fit(train_images, train_labels)

# Predict labels for the test set
predictions = svm_classifier.predict(test_images)

# Calculate accuracy of the model
accuracy = accuracy_score(test_labels, predictions)
print("Accuracy:", accuracy)

# Save the trained SVM classifier to a file using joblib
model_filename = 'svm_model.pkl'
joblib.dump(svm_classifier, model_filename)

# Now you can deploy the trained SVM classifier for flag identification in your application.
# Load the SVM classifier from the file
loaded_svm_classifier = joblib.load(model_filename)

# Print the loaded SVM classifier
print(loaded_svm_classifier)
