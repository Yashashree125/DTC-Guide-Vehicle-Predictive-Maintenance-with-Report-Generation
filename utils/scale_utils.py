import numpy as np

def scale_engine_parameters(input_values):
    means_array = np.array([0.23888651, 0.86156558, 0.60725061, 0.7811556, 0.19284566, -0.42041162, -3.52319855, -0.12294865])
    std_array = np.array([1.24680145, 1.08455267, 1.326272, 1.03891892, 2.8243492, 0.90504288, 2.59540894, 0.96297799])
    
    input_values = np.array(input_values)
    scaled_values = (input_values - means_array) / std_array
    
    return scaled_values