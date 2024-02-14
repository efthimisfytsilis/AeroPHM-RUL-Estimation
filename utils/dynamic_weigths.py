import numpy as np

def weight_calculation(Xtrain, Xtest, a=5.75, b=0.5):
    '''Analyzes the similarity between the hi_ga sequences of the training and testing specimens and computes the weights
    for each testing specimen
    
    Parameters:
    - Xtrain (dict): dictionary where the keys are the specimen names and the values are the corresponding hi_ga sequences
    - Xtest (dict): dictionary where the keys are the specimen names and the values are the corresponding hi_ga sequences
    - a (float): parameter for the similarity computation
    - b (float): parameter for the similarity computation
    
    Returns:
    - weights (dict): dictionary where the keys are the testing specimen names and the values are the corresponding weights
    of shape (n_train = 15, len(Xtest))
    '''
    A = np.log(a)/(b**2)
    weights = {}
    for test_specimen, test_hi in Xtest.items():
        current_test_specimen = {}
        for train_specimen, train_hi in Xtrain.items():
            min_len = min(len(train_hi), len(test_hi))
            # Pointwise Euclidean distance squared
            dist_squared = (train_hi[:min_len] - test_hi[:min_len])**2
            # Similarity computation via a bell-shaped function
            similarity = np.exp(-A*dist_squared) 
            # Pad the similarity array with zeros to match the length of the test hi_ga sequence
            if len(similarity) < len(test_hi):
                similarity = np.pad(similarity, (0, len(test_hi)-len(similarity)), 'constant', constant_values=0)
            current_test_specimen[train_specimen] = similarity
        # Convert the dictionary to a numpy array and reshape it to match the dimensions of the test hi_ga sequence
        similarity_array = np.array(list(current_test_specimen.values())).reshape(len(Xtrain), -1)
        # Compute the distance score from the similarity array
        distance = 1 - similarity_array
        # Compute the weights for the current test specimen
        weights[test_specimen] = (1-distance)*np.exp(-(1/b)*distance)
    return weights