# Digit Classification Using KNN and Naive Bayes

## Abstract
In this project, the K-Nearest Neighbors (KNN) and Naive Bayes (NB) machine learning models were implemented to attempt to build image recognition systems for classifying digits (MNIST dataset). The project used several iterations of different KNN and NB models, hyperparamater tuning, and data preprocessing techniques in order to improve classification accuracy. The following models/techniques were implemented within the project:

#### Models
- 1-Nearest Neighbor model
- 1-Nearest Neighbor model developed from scratch
- K-Nearest Neighbor models
- Linear Regression model
- Binomial Naive Bayes models
- Multinomial Naive Bayes models
- Guassian Naive Bayes model

#### Techniques
- Gaussian blurring
- Image deskewing via affine transformations
- Laplace smoothing
- Variance smoothing
- Data generation using Bayesian generative models
- Increasing training data set sizes
- Pixel value interval mapping (data binarization, trinarization, etc.)
- Hyperparameter search
- Confusion matrix
- Calibration evaluation

## Project Outcomes
It was determined that the following constraints led to the best performing model in terms of prediction accuracy:
- 5-Nearest Neighbor Model
- Use of the full training data set (60,000 digits)
- Deskewed digits
- Guassian blurring of data with radius = 1, sigma weighting = 1.2

## Tools Used

#### Languages:
- Python

#### Libraries:
- numpy
- pandas
- sklearn 
- matplotlib

## Research Dataset
The MNIST database hosted on OpenML was used for this project.
