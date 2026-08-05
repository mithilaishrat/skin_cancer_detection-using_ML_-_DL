# skin_cancer_detection-using_ML_-_DL
# 🩺 Skin Cancer Detection Using Machine Learning and Deep Learning

## 📌 Project Overview

This project presents an AI-based **Skin Cancer Detection System** that classifies skin lesion images into two categories:

* **Benign** – Non-cancerous lesion
* **Malignant** – Cancerous lesion

The project compares traditional **Machine Learning (ML)** algorithms with a **Convolutional Neural Network (CNN)** for image classification.

The Machine Learning models use **HOG (Histogram of Oriented Gradients)** for feature extraction, while the CNN learns visual features directly from the skin lesion images.

---

## 🎯 Objectives

The main objectives of this project are:

* To develop an automated skin lesion classification system.
* To classify images as Benign or Malignant.
* To extract image features using HOG for traditional ML models.
* To implement and compare multiple ML algorithms.
* To develop a CNN-based Deep Learning model.
* To evaluate model performance using Accuracy, Precision, Recall, and F1-score.
* To develop a simple image prediction system for new skin lesion images.

---

## 📂 Dataset

The project uses the **Melanoma Cancer Dataset** obtained through KaggleHub.

Dataset source used in the notebook:

```text
bhaveshmittal/melanoma-cancer-dataset
```

The dataset contains two classes:

```text
Benign
Malignant
```

The dataset is organized into training and testing directories:

```text
dataset/
│
├── train/
│   ├── Benign/
│   └── Malignant/
│
└── test/
    ├── Benign/
    └── Malignant/
```

---

## 🔬 Methodology

The project follows two major approaches.

### 1. Traditional Machine Learning

The ML pipeline consists of:

```text
Skin Lesion Image
        ↓
Resize to 64 × 64
        ↓
Convert to Grayscale
        ↓
HOG Feature Extraction
        ↓
StandardScaler
        ↓
ML Classifier
        ↓
Benign / Malignant
```

### 2. Deep Learning

The CNN pipeline consists of:

```text
Skin Lesion Image
        ↓
Resize to 224 × 224
        ↓
Normalization
        ↓
Data Augmentation
        ↓
CNN
        ↓
Sigmoid Output
        ↓
Benign / Malignant
```

---

# 🤖 Machine Learning Models

The following traditional ML algorithms are implemented:

### Logistic Regression

Used as a baseline classification model for the extracted HOG features.

### K-Nearest Neighbors (KNN)

Classifies an image based on its nearest training samples.

Configuration:

```text
n_neighbors = 5
```

### Random Forest

An ensemble learning algorithm consisting of multiple decision trees.

Configuration:

```text
n_estimators = 200
random_state = 42
```

### Support Vector Machine (SVM)

An SVM with an RBF kernel is used for nonlinear classification.

Configuration:

```text
kernel = RBF
C = 10
```

---

# 🧠 Deep Learning Model

## Convolutional Neural Network (CNN)

A custom CNN is implemented for binary classification.

### Architecture

```text
Input
224 × 224 × 3
      ↓
Conv2D – 32 Filters
      ↓
MaxPooling
      ↓
Conv2D – 64 Filters
      ↓
MaxPooling
      ↓
Conv2D – 128 Filters
      ↓
MaxPooling
      ↓
Flatten
      ↓
Dense – 512
      ↓
Dropout – 0.5
      ↓
Dense – 1
      ↓
Sigmoid
      ↓
Benign / Malignant
```

The CNN uses:

* ReLU activation for convolutional and dense layers
* Sigmoid activation for the final binary classification
* Adam optimizer
* Binary Cross-Entropy loss
* Dropout for reducing overfitting
* Early Stopping
* ReduceLROnPlateau

The trained model is saved as:

```text
skin_cancer_cnn.h5
```

---

# 🖼️ Image Preprocessing

For the CNN:

```text
Image Size: 224 × 224
Normalization: Pixel / 255
```

Training data augmentation includes:

* Rotation
* Width shifting
* Height shifting
* Shearing
* Zooming
* Horizontal flipping

For traditional ML:

```text
Image Size: 64 × 64
Color: Grayscale
Feature Extraction: HOG
```

---

# 📊 Evaluation Metrics

The models are evaluated using:

### Accuracy

Measures the percentage of correctly classified images.

### Precision

Measures how many predicted positive cases are actually positive.

### Recall

Measures how many actual positive cases are correctly detected.

### F1-Score

Provides a balance between Precision and Recall.

### Confusion Matrix

Used to visualize:

* True Positive
* True Negative
* False Positive
* False Negative

---

# 📈 Model Comparison

The project generates a comparison table for all models:

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |        — |         — |      — |        — |
| KNN                 |        — |         — |      — |        — |
| Random Forest       |        — |         — |      — |        — |
| SVM                 |        — |         — |      — |        — |
| CNN                 |        — |         — |      — |        — |

The actual values depend on the results obtained when the notebook is executed.

---

# 🌐 Prediction Application

A Streamlit application is included to allow users to upload a skin lesion image and obtain a prediction.

The application:

1. Accepts JPG, JPEG, or PNG images.
2. Resizes the image to 224×224.
3. Normalizes the image.
4. Loads the trained CNN.
5. Performs prediction.
6. Displays the predicted class.
7. Displays prediction confidence.

Prediction rule:

```text
Prediction > 0.5
        ↓
Malignant

Prediction ≤ 0.5
        ↓
Benign
```

---

# 📁 Project Structure

```text
Skin-Cancer-Detection/
│
├── app.py
├── skin_cancer_cnn.h5
├── skin_cancer_DL.ipynb
├── README.md
│
└── dataset/
    ├── train/
    │   ├── Benign/
    │   └── Malignant/
    │
    └── test/
        ├── Benign/
        └── Malignant/
```

> The dataset itself does not need to be uploaded to GitHub if it is large. It can be downloaded separately using Kaggle/KaggleHub.

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Scikit-learn
* OpenCV
* Scikit-image
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Streamlit
* Pillow
* KaggleHub

---

# ⚙️ Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Skin-Cancer-Detection
```

Install the required packages:

```bash
pip install tensorflow
pip install streamlit
pip install numpy
pip install pandas
pip install pillow
pip install opencv-python
pip install scikit-image
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install kagglehub
```

Or install everything together:

```bash
pip install tensorflow streamlit numpy pandas pillow opencv-python scikit-image scikit-learn matplotlib seaborn kagglehub
```

---

# ▶️ Run the Application

Make sure these files are in the same directory:

```text
app.py
skin_cancer_cnn.h5
```

Then run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

# 🧪 Example Workflow

```text
User uploads image
        ↓
Image preprocessing
        ↓
CNN model
        ↓
Prediction probability
        ↓
Classification
        ↓
Benign / Malignant
```

---

# ⚠️ Disclaimer

This project is developed for **educational and research purposes**.

The prediction generated by this system should **not be considered a medical diagnosis**. Skin cancer diagnosis should be performed by qualified healthcare professionals using appropriate clinical examination and diagnostic procedures.

---

# 👩‍💻 Author

**Mithila Ishrat Khan**

Computer Science and Engineering

North East University Bangladesh

---

# ⭐ Future Improvements

Possible future improvements include:

* Using transfer learning models such as EfficientNet, ResNet, or MobileNet.
* Increasing the size and diversity of the dataset.
* Applying class balancing techniques.
* Hyperparameter optimization.
* Adding explainable AI techniques such as Grad-CAM.
* Improving the Streamlit user interface.
* Deploying the application online.
* Adding additional skin lesion categories.
* Using clinically validated datasets for further research.

---

## 📌 Conclusion

The developed prediction system provides a simple interface for testing new images using the trained CNN model. The project demonstrates how AI-based image classification can be explored as a research tool for skin lesion analysis while emphasizing that automated predictions should not replace professional medical diagnosis.
