# Flag Identification

This project demonstrates a machine learning approach for flag identification using a Support Vector Machine (SVM) classifier. The SVM classifier is trained on a dataset of flag and non-flag images, and it can predict whether a given image is a flag or not.

## Description

The project consists of two main scripts: `main.py` and `flag.py`. The `main` script preprocesses the images, augments the dataset, splits it into training and testing sets, trains the SVM classifier, and saves the trained model to a file. The `flag` script loads the trained SVM classifier, preprocesses the test images, and classifies them as flags or non-flags.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/Samerhajj/flag_identifier.git
   cd flag_identifier
   ```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Prepare the dataset

- Place your flag images in the flag_images directory.
- Place your non-flag images in the noflag_images directory.

4. Train the SVM classifier
```bash
    python main.py
 ```

 > This script will preprocess the images, augment the dataset, split it into training and testing sets, train the SVM classifier, and save the trained model to a file (svm_model.pkl).

 5. Classify images:

- Place the images you want to classify in the testflag directory.
- Run the following command:
```bash
python flag.py
```
- The script will load the trained SVM classifier, preprocess the test images, and classify them as flags or non-flags. The predictions will be displayed on the console.

# Dependencies
The project relies on the following dependencies:

- opencv-python
- numpy
- scikit-learn
- imgaug
- joblib

You can install these dependencies using the pip package manager by running the following command:
```bash
pip install -r requirements.txt
```
Make sure you have Python 3.x installed on your system.