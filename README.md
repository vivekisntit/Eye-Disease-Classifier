# 😎 AI-Based Eye Disease Prediction System
An end-to-end deep learning system for automated eye disease detection using retinal images.

---

## Project Overview
Eye diseases such as Diabetic Retinopathy, Glaucoma, and Cataract can lead to irreversible vision loss if not detected early. Manual diagnosis is time-consuming and requires expert ophthalmologists.

This project leverages Convolutional Neural Networks (CNNs) to analyze retinal fundus images and classify eye diseases automatically, assisting in early diagnosis and clinical decision support.

---

## Objectives
- To develop an AI-driven system for eye disease classification 
- To reduce manual effort in preliminary eye screening 
- To assist healthcare professionals with faster and reliable predictions

---

## Technologies Used
- Python  
- TensorFlow & Keras  
- OpenCV  
- NumPy 
- Convolutional Neural Networks (CNN)

---

## Methodology
1. **Data Collection**  
   A labeled dataset of eye images representing different eye conditions is used.

2. **Image Preprocessing**  
   Images are resized, normalized, and augmented to enhance model generalization.

3. **Model Development**  
   A CNN architecture is trained to learn visual patterns associated with various eye diseases.

4. **Evaluation & Prediction**  
   The trained model predicts the disease class along with a confidence score for new images.

---

## Model Architecture
- Convolutional Neural Network (CNN)
- Key components:
  - Convolution layers for feature extraction
  - Pooling layers for dimensionality reduction
  - Fully connected layers for classification
- Optimizer: Adam
- Loss Function: Categorical Cross-Entropy

---
## Dataset Split
- Training Set: 80%  
- Validation Set: 10%  
- Testing Set: 10%  

---
## Results
![Preccision | Recall | F1-score](results.png)

![Confusion matrix](Confusion_matrix.png)

---

## Project Structure
```bash
├── dataset/
│   ├── cataract/
│   ├── diabetic_retinopathy/
│   └── glaucoma/
├── models/
├── notebooks/
├── src/
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── model.py
│   └── train.py
├── results/
├── README.md
└── requirements.txt
```
---

## How to Run

Follow the steps below to set up the environment and train the model.

### Clone the Repository
```bash
git clone <https://github.com/vivekisntit/Eye-Disease-Classifier.git>
cd eye-disease-prediction
```
### Install Dependencies
```bash
pip install -r requirements.txt
```