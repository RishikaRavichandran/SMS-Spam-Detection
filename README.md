# SMS Spam Detection using Machine Learning and Deep Learning

## Overview
This project aims to classify SMS messages as either spam or ham (not spam) using various machine learning and deep learning techniques. The dataset is preprocessed, vectorized, and used to train models such as Naïve Bayes, Logistic Regression, Random Forest, and a Neural Network.

## Features
- **Text Preprocessing:** Cleaning and tokenizing SMS messages.
- **Vectorization:** TF-IDF and Bag of Words (BoW) techniques.
- **Machine Learning Models:** Naïve Bayes, Logistic Regression, and Random Forest.
- **Deep Learning Model:** Neural Network using TensorFlow/Keras.
- **Performance Evaluation:** Accuracy, Precision, Recall, and F1-score.

## Dataset
The dataset consists of SMS messages labeled as spam or ham. It undergoes preprocessing, including:
- Removing special characters and numbers
- Tokenization and lemmatization
- Stopword removal

## Model Performance
| Model                 | Vectorization  | Accuracy  | Precision | Recall  | F1-score |
|----------------------|---------------|-----------|-----------|---------|---------|
| Naïve Bayes          | TF-IDF        | 96.59%    | 100.00%   | 74.67%  | 85.49%  |
| Naïve Bayes          | Bag of Words  | 97.39%    | 87.58%    | 94.00%  | 90.67%  |
| Logistic Regression  | TF-IDF        | 95.51%    | 96.29%    | 69.33%  | 80.62%  |
| Logistic Regression  | Bag of Words  | 97.75%    | 100.00%   | 83.33%  | 90.90%  |
| Random Forest       | TF-IDF        | 97.75%    | 99.21%    | 84.00%  | 90.97%  |
| Random Forest       | Bag of Words  | 97.93%    | 100.00%   | 84.67%  | 91.70%  |
| Neural Network      | TF-IDF        | 98.20%    | 97.79%    | 88.67%  | 93.00%  |

## Installation
To run this project, install the necessary dependencies:
```bash
pip install numpy pandas scikit-learn tensorflow keras
```

## Usage
Run the following script to train and evaluate models:
```bash
python spam_classifier.py
```

## Contributions
Feel free to contribute by improving models, adding features, or optimizing preprocessing steps.

## License
This project is for educational purposes. Unauthorized sharing of the dataset may be restricted.

---
**Author:** Your Name  
**Contact:** your.email@example.com

