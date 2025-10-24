Medical Insurance Cost Prediction using Machine Learning

This project demonstrates the development and comparison of two machine learning models for predicting medical insurance costs:

Random Forest Regressor (Scikit-learn)

Neural Network (TensorFlow/Keras)

Project Overview
Data Preprocessing: Handling categorical variables (sex, smoker, region) and numerical features (age, BMI, children)

Model Development: Building both untuned and hyperparameter-tuned versions

Hyperparameter Tuning: Using GridSearchCV for Random Forest and KerasTuner for Neural Network

Performance Comparison: Comprehensive evaluation using MAE, MSE, and R² metrics

Key Features
Data exploration and visualization

Feature engineering and preprocessing pipeline

Hyperparameter optimization for both models

Performance comparison between traditional ML and deep learning approaches

Model evaluation and validation

Technologies Used
Python, Pandas, NumPy

Scikit-learn, TensorFlow, Keras

KerasTuner for neural network optimization

Matplotlib for visualization

Results
The tuned Random Forest model achieved better performance with lower MAE compared to the neural network, demonstrating the effectiveness of ensemble methods for this regression task.

