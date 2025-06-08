import numpy as np
from collections import Counter

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
        kNearestIndices = [i for i, d in indexedDistances[:self.k]] 

        #labels of k nearest neighbors
        kNearestLabels = [self.trainingLabels[i] for i in kNearestIndices]

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
def EvaluateFeatures(filePath, featureSubset, kVale):
    #tell the user we are loading dataset
    # print(f"Loading dataset: {filePath}...")
    features, labels = LoadDataset(filePath)
    
    #edge case: if the data did not load correctly
    if features is None:
        return None
        
    #print out all the features that we are going to use
    #print(f"Evaluating with features: {featureSubset}")
    
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
    # Using k >= 1 for the nearest neighbor as per the base requirement 
    # k value is from user input. 
    accuracy = CrossValidation(normalizedFeatures, labels, kVale)
    
    #return in percentage 
    return round(accuracy *100, 1)

def ForwardSelection(num_feature, filePath, kVale):

    # initial set of feature 
    overallBestFeature = set()
    currentFeature = set()

    # print the first random evaluation for initial feature.
    score = EvaluateFeatures(filePath, list(currentFeature), kVale)
    print(f"Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of {score}%")
    print("Beginning search.")

    # best score overall level
    overallBestScore = score

    # each level, check the best score and feature
    for level in range(num_feature):
        currentBestScore = -1.0
        currentBestFeature = None

        # set the feature that not in current, therefore it need to add to caluculate. 
        availableFeature = set(range(1, num_feature +1)) - currentFeature

        # keep track of how many different features are evaluated as potential candidates to be added at the current level
        # featureCounter = 0

        # in each combiation of feature, check the best score
        for feature in availableFeature:
            #featureCounter += 1
            tempSet = currentFeature.copy()
            tempSet.add(feature)
            
            # modify : Evaluate the current feature subset and calculate its classification accuracy
            tempScore = EvaluateFeatures(filePath, list(tempSet), kVale)
            print(f"    Using feature(s) {tempSet} accuracy is {tempScore}%")

            # check if the current score is greater then the overall score
            if tempScore > currentBestScore:
                currentBestScore = tempScore
                currentBestFeature = tempSet.copy()
            
        # Update current_features to the best one found at this level
        currentFeature = currentBestFeature

        # Print level summary only if there was a choice
        #if featureCounter > 0:
        print(f"\nFeature set {currentFeature} was best, accuracy is {currentBestScore}%\n")
            
        # update overall best score and overall best feature
        if currentBestScore > overallBestScore:
            overallBestScore = currentBestScore
            overallBestFeature = currentBestFeature.copy()
        
    # print the best score 
    print(f"\nSearch finished! The best subset of features is {overallBestFeature}, which has an accuracy of {overallBestScore}%")


# the same intuition to Forward selection, but go backward. 
# It starts with the full set of features and removes one feature at a time.
def BackwardElimination(num_feature, filePath, kVale):

    # initial set of feature 
    currentFeature = set(range(1, num_feature + 1))
    overallBestFeature = currentFeature.copy()

    # print the first random evaluation for initial feature.
    score = EvaluateFeatures(filePath, list(currentFeature), kVale)
    print(f"Running nearest neighbor with all features using \"leaving-one-out\" evaluation, I get an accuracy of {score}%.")
    print("Beginning search.")
    # best score overall level
    overallBestScore = score

    # each level, check the best score and feature
    for level in range(num_feature - 1):
        currentBestScore = -1.0
        currentBestFeature = None

        for feature in currentFeature:
            tempSet = currentFeature - {feature}

            # modify : Evaluate the current feature subset and calculate its classification accuracy
            tempScore = EvaluateFeatures(filePath, list(tempSet), kVale)
            print(f"    Using feature(s) {tempSet} accuracy is {tempScore}%")

            if tempScore > currentBestScore:
                currentBestScore = tempScore
                currentBestSubset = tempSet

        currentFeature = currentBestSubset
        print(f"\nFeature set {currentFeature} was best, accuracy is {currentBestScore}%\n")

        if currentBestScore > overallBestScore:
            overallBestScore = currentBestScore
            overallBestFeature = currentBestSubset.copy()

    print(f"Search finished! The best subset of features is {overallBestFeature}, which has an accuracy of {overallBestScore}%")

def main():
    # input instruction prompt
    print("Welcome to Amshu Bellur (abell062), Lyda Taing (ltain005), and Gregory Wang (gwang086) Feature Selection Algorithm.")
    print("-------------------------------------------------------------------------------------------------------------------\n")

    filePath = input("Enter the path to the dataset file (ex., 'small-test-dataset.txt'): ")

    # 2D numpy array each row is datapoint -> .shape[1] and each column is a feature -> .shape[0]
    features = LoadDataset(filePath)[0] # adjust to make feature to 2D numpy 
    num_feature = features.shape[1]
    num_instances = features.shape[0]

    print("\n")
    print(f"This dataset has {num_feature} features (not including the class attribute), with {num_instances} instances.")
    print("Please wait while I normalize the data…  Done!")

    #get k value for K-NN classification 
    while True:
        try: 
            kVale = int(input("Please enter a value of k for k-NN classification: "))
            # validation k > 0 
            if kVale <= 0:
                print("Please enter the positive value. \n")
                continue
            else: 
                break
        except ValueError:
            print ("Invalid input, please enter a positive number. \n")

    # choose the algorithm 
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination") 

    # input validation - 1 or 2
    while True:
        try: 
            choice = int(input())
            if choice == 1:
                ForwardSelection(num_feature, filePath, kVale)
                break
            elif choice ==2:
                BackwardElimination(num_feature, filePath, kVale)
                break
            else:
                print("invalid input. Enter 1 or 2.")
        except ValueError:
            print("invalid input. Enter 1 or 2.")
            
if __name__ == "__main__":
    main()