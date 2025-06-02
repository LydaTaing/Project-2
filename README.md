# Project-2

## Part 1: Implementing the search algorithm only  

* Greedy forward that starts with empty set of selection then iteratively increase the number of feature.
* Using evalutaion function to find a score of each node : use a stub evaluation function that returns a random value
* Pirnt out the highest score at each step.
* Backward-elimination is very similar: it starts with the full, set of features and removes one feature at a time.

## Part 2: Nearest Neighbor Classifier and Validator
* Loads dataset from .txt files
* Normalizes selected features using Min-Max scaling.
* Performs 1-NN classification using Euclidean distance.
* Validates accuracy using Leave-One-Out Cross Validation.
* Accepts user input for file path and feature subset.

## Part 3: 
* Apply your feature selection algorithms (Forward Selection and Backward Elimination from Part I) using your actual evaluation function (1-Nearest Neighbor classifier with leave-one-out cross-validation from Part II) to the titanic_clean.txt dataset.
