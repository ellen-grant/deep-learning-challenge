# deep-learning-challenge
# Alphabet Soup Charity Funding Predictor

## Overview
This project involves building a deep learning model to predict the success of funding applicants for Alphabet Soup, a nonprofit foundation. The goal is to assist Alphabet Soup in making more informed decisions about which applicants to fund, thereby increasing the likelihood of funding success.

The project includes data preprocessing, model development, optimization, and evaluation to build a robust binary classifier.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Optimization Steps](#optimization-steps)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Dataset
The dataset provided by Alphabet Soup includes over 34,000 records with various metadata about organizations that have received funding. Key columns in the dataset are:
- **EIN**: Employer Identification Number (removed as it’s non-informative)
- **NAME**: Organization name (removed as it’s non-informative)
- **APPLICATION_TYPE**: Type of application submitted
- **AFFILIATION**: Sector affiliation of the organization
- **CLASSIFICATION**: Government classification of the organization
- **USE_CASE**: Intended use case for the funding
- **ORGANIZATION**: Type of organization (e.g., company, non-profit)
- **STATUS**: Status of the application (removed in the optimization phase)
- **INCOME_AMT**: Income classification
- **SPECIAL_CONSIDERATIONS**: Special considerations for the application
- **ASK_AMT**: Amount of funding requested
- **IS_SUCCESSFUL**: Target variable indicating funding success

## Technologies Used
- Python
- Pandas
- TensorFlow and Keras
- Scikit-learn
- Google Colab (for model training and testing)

## Model Architecture
The deep learning model was initially designed with a simple neural network structure and optimized through several iterations. The final model architecture includes:
- **Input Layer**: Matches the number of features after preprocessing
- **Hidden Layers**: Four hidden layers with different neuron counts to capture complex patterns
- **Output Layer**: One neuron with a sigmoid activation function for binary classification

The model uses `leaky_relu` activation functions in hidden layers and `sigmoid` in the output layer.

## Optimization Steps
Several steps were taken to optimize model performance, including:
1. **Dropping the `STATUS` column** as it was found to be non-beneficial.
2. **Switching from `relu` to `leaky_relu`** activation functions to prevent dead neurons.
3. **Adding additional layers** to increase model complexity, resulting in a total of four hidden layers.
4. **Using EarlyStopping** to prevent overfitting by halting training when the validation loss stopped improving.
5. **Increasing the number of epochs to 300** and **batch size to 64** to give the model more training time.
6. **Experimenting with the `tanh` activation function**, which did not significantly impact accuracy.

## Results
- **Initial Model Performance**: Loss = 0.5668, Accuracy = 72.5%
- **Optimized Model Performance**: Loss = 0.5576, Accuracy = 72.2%
- Although the optimized model did not achieve the target accuracy of 75%, it still provides valuable insights and guidance for Alphabet Soup’s funding decisions.

## Installation
To run this project locally, ensure that you have the following dependencies installed:
- Python 3.6 or higher
- Pandas
- TensorFlow
- Scikit-learn

To install the necessary libraries, use:
```bash
pip install pandas tensorflow scikit-learn
```
## Usage
1. Clone the repository.
2. Load the `charity_data.csv` dataset.
3. Run the data preprocessing steps as outlined in the `AlphabetSoupCharity_Optimization.ipynb`.
4. Train and evaluate the model.
5. The model can be saved as an HDF5 file using `model.save("AlphabetSoupCharity.h5")` for future predictions.

## Future Work
Given that the neural network model did not reach the target accuracy, future work could involve:

- Experimenting with alternative machine learning models, such as **Random Forest** or **Gradient Boosting**, which may better capture patterns in tabular data.
- Further feature engineering, including interaction terms and additional binning of continuous variables.
- Hyperparameter tuning using tools like **KerasTuner** for systematic optimization.
