# Differentially Private Multi-Dimensional Analysis via Double-Sided Clipping and Prefix-Sum Cube

## Environment
*__If you need to run the code, you need to include the following package in your python environment__*

python

pandas

pprint

scipy

numpy

argparse

seaborn

matplotlib

cvxopt

itertools


## File Directory
### auto_trip.py
This code implements some functions for cropping the dataset, including data reading, data query preprocessing, threshold acquisition for adaptive bilateral cropping, etc. It is combined with unbiased processing of sparse vector technology to achieve bilateral cropping.

### count_query.py
This code is a script for generating and evaluating synthetic datasets, including querying the 1D, 2D, and 4D marginal distributions. And calculating the error between the 4D synthetic dataset and the actual 4D distribution. It also extend the synthetic mind dataset to target dimensions.

### data_preprocess.py
The main function is to perform data preprocessing and compress the value range of high-dimensional data sets to reduce the dimension of the dataset.

### estimation.py
This code implements a maximum entropy distribution synthesizer, which is mainly used to generate probability distributions with maximum entropy.The synthesizer can process data containing both one- and two-dimensional attributes, generating maximum entropy distributions by solving convex optimization problems.The core goal of this code is to generate a probability distribution that satisfies the principle of maximum entropy under given constraints, which can be used to simulate real-world data distribution.

### max_entropy.py
This code implements the application of the maximum entropy method in probability distribution estimation. Using the principle of maximum entropy, the code provides functions for estimating a target distribution from a given one- or two-dimensional marginal distribution.Through these functions, the code provides users with a tool for probability distribution estimation based on the principle of maximum entropy.

### utils.py
This code defines a cartesian class and a series of auxiliary functions, which are mainly used for calculation of Cartesian products and some related operations of probability statistics and information theory.This class provides functions for adding data lists, calculating Cartesian products, and other mathematical and set operations, and can be easily used in probability distribution and statistics application scenarios.

### prefix_sum.py
Combined with the algorithm mentioned in the above document, perform bilateral cropping and sum query based on prefix sum, and calculate the error


