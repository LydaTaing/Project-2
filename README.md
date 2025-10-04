# Search Algorithm 

## Part 1: Implementing the search algorithm only  

* Greedy forward that starts with empty set of selection then iteratively increase the number of feature.
* Using evalutaion function to find a score of each node : use a stub evaluation function that returns a random value
* Pirnt out the highest score at each step.
* Backward-elimination is very similar: it starts with the full, set of features and removes one feature at a time.

## Part 2: Nearest Neighbor Classifier and Validator
* Loads dataset from .txt files
* Normalizes selected features using Min-Max scaling - tranform all value to be in range of [0, 1]
* Performs 1-NN classification using Euclidean distance and return label of the closet training instance. 
* Validates accuracy using Leave-One-Out Cross Validation. count correct prediction to compute accuratecy rate. 
* Accepts user input for file path and feature subset.

## Part 3: 
* Apply your feature selection algorithms : Forward Selection and Backward Elimination by using the actual evaluation function k Nearest Neighbor classifier with leave-one-out cross-validation to the titanic_clean.txt dataset.
* k value - user input.
* update NearestNeighborClassifier class to use value k
* modified test function to select the top k closet distance value - the first index from a sorted list.
* modified the related function to accept k value. 

