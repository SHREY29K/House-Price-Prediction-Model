# House-Price-Prediction-ModelHouse Price Prediction Model
This repository contains a machine learning model for predicting house prices using XGBoost and Random Forest Classification. The project also includes various visualizations to analyze and interpret the data and model performance.

Overview:
The goal of this project is to build predictive models that estimate house prices based on various features. We use both XGBoost and Random Forest algorithms to achieve high accuracy. Additionally, visualizations are provided to understand the data and model results better.

Project Structure:
data/: Contains datasets used for training and testing the models.
src/: Includes the code for data preprocessing, model training, evaluation, and visualization.
notebooks/: Jupyter notebooks with exploratory data analysis (EDA), visualizations, and model validation.
visualizations/: Contains saved plots and charts used in the analysis.
README.md: This file.
requirements.txt: List of Python packages required to run the code.

Getting Started:
Prerequisites
Make sure you have the following installed:

Python 3.x
XGBoost
Scikit-learn
Pandas
NumPy
Matplotlib
Seaborn

You can install the required packages using pip:
pip install -r requirements.txt

Data
Ensure that your data files are located in the data/ directory. The expected structure is:

train.csv: Training data with features and target prices.
test.csv: Testing data with features only (for predictions).

Usage
1) Data Preprocessing

The preprocessing script prepares the data for training and testing. Ensure that your data files are correctly placed in the data/ directory before running the script.

python src/preprocess.py

2) Model Training
Run the following scripts to train the models:

XGBoost Model:
python src/train_xgb_model.py

Random Forest Model:
python src/train_rf_model.py

=> These scripts will:

Load the training data.
Train the respective models.
Save the trained models to files.

3) Model Evaluation
Evaluate the trained models on the test data to check their performance:

XGBoost Evaluation:
python src/evaluate_xgb_model.py

Random Forest Evaluation:
python src/evaluate_rf_model.py

These scripts will:

Load the test data.
Make predictions.
Evaluate and print the model performance metrics.

4) Visualization
Generate and view various visualizations related to the data and model performance:

python src/visualize.py

This script will:
Produce and save plots such as feature importance, training vs. validation performance, and prediction distributions.

5) Prediction
To make predictions on new data, use:

python src/predict.py --input data/new_data.csv --output data/predictions.csv

Replace data/new_data.csv with your new data file and specify the output file for predictions.

Code Explanation
src/preprocess.py: Handles data preprocessing tasks such as feature scaling and splitting data into training and test sets.
src/train_xgb_model.py: Defines and trains the XGBoost model with early stopping and saves the model.
src/train_rf_model.py: Defines and trains the Random Forest model and saves the model.
src/evaluate_xgb_model.py: Loads the trained XGBoost model and evaluates its performance on test data.
src/evaluate_rf_model.py: Loads the trained Random Forest model and evaluates its performance on test data.
src/visualize.py: Generates and saves various visualizations for analysis.

Performance
The models use early stopping and various performance metrics for evaluation. Results are compared between XGBoost and Random Forest classifiers.

Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request with your changes. Ensure that your code is well-documented and tested.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Contact
For any questions or issues, please contact at shk29171112@gmail.com.
