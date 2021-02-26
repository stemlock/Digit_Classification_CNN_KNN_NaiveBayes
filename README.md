# Digit Classification Using K-Nearest Neighbors, Naive Bayes, Feed-Forward Neural Networks, and Convolutional Neural Networks

## Abstract
In this project, the K-Nearest Neighbors (KNN), Naive Bayes (NB), Feed-Forward Neural Network (NN), and Convolutional Neural Network (CNN) machine learning models were implemented to attempt to build image recognition systems for classifying digits (MNIST dataset). The project used several iterations of different KNN, NB, NN, and CNN models, hyperparamater tuning, and data preprocessing techniques in order to improve classification accuracy. The following models/techniques were implemented within the project:

#### Models
- 1-Nearest Neighbor model
- 1-Nearest Neighbor model developed from scratch
- K-Nearest Neighbor models
- Linear Regression model
- Binomial Naive Bayes models
- Multinomial Naive Bayes models
- Guassian Naive Bayes models
- Single Layer Feed-Forward Neural Network
- Multi-layered Feed-Forward Neural Networks
- Convolutional Neural Networks

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
- Batch gradient descent
- Stochastic gradient descent
- Various activation functions
- Dropout layers
- MaxPooling

## Project Outcomes
It was determined that the following constraints led to the best performing shallow learning model in terms of prediction accuracy:
- 5-Nearest Neighbor Model
- Use of the full training data set (60,000 digits)
- Deskewed digits
- Guassian blurring of data with radius = 1, sigma weighting = 1.2

The final prediction accuracy score reached for the KNN model was ~98.6%.

The best performing deep learning model had the following constraints:
- Convolutional Neural Network with 4 layers
	- 1st Convolutional Layer with 32 filters, relu activation
	- 2nd Convolutional Layer with 64 filters, relu activation
	- 2 by 2 MaxPooling
	- 3rd Fully Connected Layer with 50 nodes, relu activation
	- 4th Fully Connected Layer with 10 nodes, softmax activation
- Dropout of 0.5
- Stochastic gradient descent with batch size of 100
- Learning rate of 0.01
- 20 epochs

The final prediction accuracy score reached for the CNN model was ~98.4%.


## Tools Used

#### Languages:
- Python

#### Libraries:
- numpy
- pandas
- sklearn 
- Keras
- matplotlib


## Research Dataset
The MNIST database hosted on OpenML was used for this project.
