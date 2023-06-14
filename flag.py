import cv2
import os
from joblib import load

# Load the trained SVM classifier from the file
model_filename = 'svm_model.pkl'
svm_classifier = load(model_filename)

# Preprocess a single image
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Classify an image as flag or non-flag
def classify_image(image):
    preprocessed_image = preprocess_image(image)
    flattened_image = preprocessed_image.reshape(1, -1)
    prediction = svm_classifier.predict(flattened_image)
    return prediction[0]  # Return the predicted label (0 for non-flag, 1 for flag)

# Directory containing the images to classify
image_dir = 'testflag'

# Iterate over all images in the directory
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)
    if not os.path.isfile(image_path):
        print(f"Invalid image file: {image_path}")
        continue
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        continue
    prediction = classify_image(image)
    print(f"Image: {filename}, Prediction: {prediction}")
    if prediction == 1:
        print("The image is a flag.")
    else:
        print("The image is not a flag.")
    print()
