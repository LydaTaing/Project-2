import random

# Using evalutaion function to find a score of each node : use a stub evaluation function that returns a random value
def evalFunction(featureSet):
    return round(random.uniform(10.0, 95.0), 1) 

# Greedy forward that starts with empty set of selection then iteratively increase the number of feature. 
# Using evalutaion function to find a score of each node :use a stub evaluation function that returns a random value
# Print out the highest score at each step. 
def ForwardSelection(num_feature):
    # print the first random evaluation for initial feature.
    score = evalFunction(set()) #empty set 
    print(f"Using no features and \"random\" evaluation, I get an accuracy of {score}%. Beginning search.\n")

    # initial set of feature 
    currentFeature = set()
    overallBestFeature = set()

    # best score overall level
    overallBestScore = score

    # each level, check the best score and feature
    for level in range(num_feature):
        currentBestScore = -1.0
        currentBestFeature = None

        # set the feature that not in current, therefore it need to add to caluculate. 
        availableFeature = set(range(1, num_feature +1)) - currentFeature

        # keep track of how many different features are evaluated as potential candidates to be added at the current level
        #featureCounter = 0

        # in each combiation of feature, check the best score
        for feature in availableFeature:
            #featureCounter += 1
            tempSet = currentFeature.copy()
            tempSet.add(feature)
            tempScore = evalFunction(tempSet)
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
def BackwardElimination(num_feature):

    # initial set of feature 
    currentFeature = set(range(1, num_feature + 1))
    overallBestFeature = currentFeature.copy()

    # print the first random evaluation for initial feature.
    score = evalFunction(currentFeature) #empty set 
    print(f"Using all features and \"random\" evaluation, I get an accuracy of {score}%. Beginning search.\n")

    # best score overall level
    overallBestScore = score

    # each level, check the best score and feature
    for level in range(num_feature - 1):
        currentBestScore = -1.0
        currentBestFeature = None
        
        #remove each feature per each iteration and evaluate a new subset
        for feature in currentFeature:
            tempSet = currentFeature - {feature}
            tempScore = evalFunction(tempSet)
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
    print("Welcome to the Feature Selection Algorithm.")
    num_feature = int (input("Please enter total number of features: "))
    
    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    
    # input validation - 1 or 2
    while True:
        try: 
            choice = int(input("\n"))
            if choice == 1:
                ForwardSelection(num_feature)
                break
            elif choice ==2:
                BackwardElimination(num_feature)
                break
            else:
                print("invalid input. Enter 1 or 2.")
        except ValueError:
            print("invalid input. Enter 1 or 2.")
            
if __name__ == "__main__":
    main()