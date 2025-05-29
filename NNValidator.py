import numpy as np
import pandas as pd
from collections import Counter
<<<<<<< HEAD
#from sklearn.preprocessing import MinMaxScaler
=======
>>>>>>> 05c90dabb4b00e1e9d1e9a583a1497da275cc1cf

#class that chooses the class for the data point based on eucidean distance
class NearestNeighborClassifier:
    #Initializes the classifier with a given k, default is k=1
    def __init__(self, k=1):
        self.k = k

    #stores the training data: 
    #training_instances (np.ndarray): A 2D array of training data points
    #training_labels (np.ndarray): A 1D array of labels for the training data
    def train(self, trainingInstances, trainingLabels):
        self.trainingInstances = trainingInstances
        self.trainingLabels = trainingLabels

    #calulate euclidean distance of the two points
    def _euclidean_distance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    #predct the class for single test
    #arg: test_instance (np.ndarray): A 1D array representing a single data point
    #return: class label
    def test(self, testInstance):

        #distance for all points
        distances = [
            self._euclidean_distance(testInstance, trainInstance)
            for trainInstance in self.trainingInstances
        ]
        
        #create a tuple with the index of the distance and the value of the distance
        indexedDistances = [
            (i, d) 
            for i, d in enumerate(distances)
        ]
        #sort the distances by the second value in the tuple, the distance acending order
        indexedDistances.sort(key=lambda x: x[1])
        #get only the firstvalue in the tuple 
        kNearestIndices = [i for i, d in indexedDistances[:1]]

        #labels of k nearest neighbors
        kNearestLabels = [
            self.trainingLabels[i]
            for i in kNearestIndices
        ]

        #count and return most common label
        most_common = Counter(kNearestLabels).most_common(1)
        return most_common[0][0]
    

#goes through each point in the data set and caluates overall accuracy
#arg: features (np.ndarray): The full dataset features
#args: labels (np.ndarray): The full dataset labels
#args: k_neighbors (int): The number of neighbors for the KNN classifier
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
        
        #all other data points are the training data
        trainingInstances = np.delete(features, i, axis=0)
        trainingLabels = np.delete(labels, i, axis=0)
        
        #initialize and train classifier
        classifier = NearestNeighborClassifier(k=kNeighbors)
        classifier.train(trainingInstances, trainingLabels)
        
        #prediction of the class
        prediction = classifier.test(testInstance)
        
        #is the prediction correct?
        if prediction == testLabel:
            correctPredictions += 1
            
    #calculate and return the accuracy
    accuracy = correctPredictions / numInstances
    return accuracy

#load all the data in file--> the first column is the class, and the rest are features.
#arg: file_path (str): The path to the dataset file.
#return: tuple: A tuple containing (features, labels) as NumPy arrays.
def LoadDataset(file_path):
    #look at the dataset and find the class and features
    try:
        data = np.loadtxt(file_path)
        labels = data[:, 0]
        features = data[:, 1:]
        return features, labels
    #if the file path is incorrect
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None, None
    #if the file is not able to read
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None, None
    
#normalize the data using minMax
#arg: data (2D array): A 2D NumPy array where rows are instances and columns are features.
#return: the normalized 2D array
def normalizeData(data):
    #find row and column size
    rows = data.shape[0]
    cols = data.shape[1]

    #create a copy of data to normalize
    normalized = data.copy()
    #iterate through each column of data 
    for j in range(data.shape[1]): # data_array.shape[1] is the number of columns
        column = data[:, j]
        minimum = np.min(column)
        maximum = np.max(column)

        #change the values/check if the min and max are same
        if maximum == minimum:
            # If all values in the column are the same, set them to 0
            for i in range(rows):
                normalized[i, j] = 0
        else:
            #iterates through each element, the row, in the current column
            for i in range(rows):
                #original value
                value = data[i, j]
                
                #normalized value
                newValue = (value - minimum) / (maximum - minimum)
                
                #re-assign to correct position
                normalized[i, j] = newValue
    
    #return normalized 2D array
    return normalized

#combining all the functions together; Load dataset, select feature subset, normalize, and returns the classification accuracy using validation.
#arg: file_path (str): The path to the dataset file
#arg: feature_subset (list of int): A list of 1-based feature numbers to use
#return: float: The calculated accuracy, or None if an error occurs
def EvaluateFeatures(filePath, featureSubset):
    #tell the user we are loading dataset
    print(f"Loading dataset: {filePath}...")
    features, labels = LoadDataset(filePath)
    
    #edge case: if the data did not load correctly
    if features is None:
        return None
        
    #print out all the features that we are going to use
    print(f"Evaluating with features: {featureSubset}")
    
    #convert the number to index by subtracting one: real # -->{2, 3, 7} / index value --> {1, 2, 6}
    featureIndices = [
        i - 1
        for i in featureSubset
    ]

    #select only the chosen features (chooses the columns based on the index given)
    selectedFeatures = features[:, featureIndices]
    
    #normalize the selected features
    normalizedFeatures = normalizeData(selectedFeatures)
    
    # Calculate accuracy using the previously defined validation function
    # Using k=1 for the nearest neighbor as per the base requirement
    accuracy = CrossValidation(normalizedFeatures, labels, 1)
    
    return accuracy

#define the main that asks for which data they want and return the correct output
def main():
    #ask user for choice of feature and dataset
    filePath = input("Enter the path to the dataset file (ex., 'small-test-dataset.txt'): ")
    featuresInput = input("Enter the feature indices separated by commas (ex., 3,5,7): ")
    #format into a comma seperated list - split seperates the data based on certain character
    features = [int(i.strip()) for i in featuresInput.split(',')]
   
    #print the statement of running tests with certain features
    print(f"--- Running evaluation for {filePath} with features {features} ---")
    
    #call the evaluation function with input variables
    accuracy = EvaluateFeatures(filePath, features)
    
    #if there are no errors, give the output
    if accuracy is not None:
        print(f"\nAccuracy: {accuracy:.5f}\n")

# This is how you would run it
if __name__ == "__main__":
    main()
    