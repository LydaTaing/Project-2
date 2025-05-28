import numpy as np
from collections import Counter

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
def crossValidation(features, labels, kNeighbors):
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