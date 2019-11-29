from collections import Counter
import numpy as np

def create_class_weights(train_generator):
    print('Calculating class weights...')

    sample_count = list(train_generator.classes[train_generator.index_array])
    sample_count = Counter(sample_count)

    weights = [sum(sample_count.values())/count for count in sample_count.values()]
    # Divide each weight by the average weight to bring the range of weights closer to 1.
    # This is so that the hyperparameters are still suitable for the problem.
    # The relative difference remains the same.
    weights = [w/np.mean(weights) for w in weights]

    class_weights = {i:val for i,val in enumerate(weights)}

    return class_weights
