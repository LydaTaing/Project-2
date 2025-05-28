import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

#class that chooses the class for the point based on eucidean distance
class NearestNeighborClassifier:
    #Initializes the classifier with a given k, default is k=1 for simplicity.
    def __init__(self, k=1):
        self.k = k

    #stores the training data: 
    #training_instances (np.ndarray): A 2D array of training data points.
    #training_labels (np.ndarray): A 1D array of labels for the training data.
    def train(self, trainingInstances, trainingLabels):
        self.trainingInstances = trainingInstances
        self.trainingLabels = trainingLabels

    #calulate euclidean distance
    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    #predct the class for single test
    #arg: test_instance (np.ndarray): A 1D array representing a single data point.
    #return: class label
    def test(self, testInstance):

        #distance for all the points
        distances = [
            self._euclidean_distance(testInstance, trainInstance)
            for trainInstance in self.trainingInstances
        ]
        
        #indices of the k nearest neighbors
        kNearestIndices = np.argsort(distances)[:self.k]
        
        #labels of the k nearest neighbors
        kNearestLabels = [
            self.trainingLabels[i]
            for i in kNearestIndices
        ]

        #count and return most common class label (majority vote)
        most_common = Counter(kNearestLabels).most_common(1)
        return most_common[0][0]
    

#goes through each point in the data set and then caluates the overall accuracy
#arg: features (np.ndarray): The full dataset features
#args: labels (np.ndarray): The full dataset labels
#args: k_neighbors (int): The number of neighbors (k) for the KNN classifier
#return: accuracy of classifier
def CrossValidation(features, labels, kNeighbors):
    #count the correct ad total to make accuracy 
    correctPredictions = 0
    numInstances = features.shape[0]
    
    #loop through all the instances
    for i in range(numInstances):
        #the test data for this round
        testInstance = features[i]
        testLabel = labels[i]
        
        #all other instances are the training data
        trainingInstances = np.delete(features, i, axis=0)
        trainingLabels = np.delete(labels, i, axis=0)
        
        #initialize and train classifier
        classifier = NearestNeighborClassifier(k=kNeighbors)
        classifier.train(trainingInstances, trainingLabels)
        
        #prediction
        prediction = classifier.test(testInstance)
        
        #is the prediction correct?
        if prediction == testLabel:
            correctPredictions += 1
            
    #calculate and return the accuracy
    accuracy = correctPredictions / numInstances
    return accuracy

#load all the data through the given file, the first column is the class, and the rest are features.
#arg: file_path (str): The path to the dataset file.
#return: tuple: A tuple containing (features, labels) as NumPy arrays.
def LoadDataset(file_path):
    try:
        data = np.loadtxt(file_path)
        labels = data[:, 0]
        features = data[:, 1:]
        return features, labels
    #catch if the file path is incorrect
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None
    #if the file is not able to read
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None
    
#combining all the functions together; Load dataset, select feature subset, normalize, and returns the classification accuracy using validation.
#arg: file_path (str): The path to the dataset file
#arg: feature_subset (list of int): A list of 1-based feature numbers to use
#return: float: The calculated accuracy, or None if an error occurs
from sklearn.preprocessing import MinMaxScaler
def EvaluateFeatures(filePath, featureSubset):
    #tell the user we are loading dataset
    print(f"Loading dataset: {filePath}...")
    features, labels = LoadDataset(filePath)
    
    #edge case: if the data did not load correctly
    if features is None:
        return None
        
    #print out all the features
    print(f"Evaluating with features: {featureSubset}")
    
    #convert the number to index by subtracting one ex.{2, 3, 7} --> {1, 2, 6}
    featureIndices = [
        i - 1
        for i in featureSubset
    ]

    #select only the chosen features
    selectedFeatures = features[:, featureIndices]
    
    #normalize the selected features -- use sklearn
    scaler = MinMaxScaler()
    normalizedFeatures = scaler.fit_transform(selectedFeatures)
    
    # Calculate accuracy using the previously defined validation function
    # Using k=1 for the nearest neighbor as per the base requirement
    accuracy = CrossValidation(normalizedFeatures, labels, kNeighbors=1)
    
    return accuracy

#define the main that asks for which data they want and return the correct output
def main():
    #ask user for choice of feature and dataset
    filePath = input("Enter the path to the dataset file (e.g., 'small-test-dataset.txt'): ")
    featuresInput = input("Enter the feature indices separated by commas (e.g., 3,5,7): ")
    #format into a comma seperated list
    features = [int(i.strip()) for i in featuresInput.split(',')]
   
    #print the statement of running tests with certain features
    print(f"--- Running evaluation for {filePath} with features {features} ---")
    
    #call the evaluation function with input variables
    accuracy = EvaluateFeatures(filePath, features)
    
    if accuracy is not None:
        print(f"\nAccuracy: {accuracy:.5f}\n")

# This is how you would run it
if __name__ == "__main__":
    main()
    