import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

def OptDigits_extract_and_convert(filename):
    """
    returns images (2D matrices ready to be plotted) with integer values 0 to 16 and one-hot labels with binary values
    """
    dataset = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            dataset.append([int(y) for y in line.split(',')])
    dataset = jnp.array(dataset)
    images = []
    labels = []
    for case in dataset:
        image = case[:64].reshape(8,8)
        label = case[-1] 
        label_one_hot = [0]*10
        label_one_hot[label] = 1
        images.append(image)
        labels.append(label_one_hot)
    images = jnp.array(images)
    labels = jnp.array(labels)
    return images, labels

# converting to amplitudes:
def OptDigits_SL_map_features_and_labels(feature_multiplier, feature_constant, label_multiplier, amplitudes, outputn):
    
    amplitude_relative = jnp.mean(amplitudes[outputn])
    
    images, labels = OptDigits_extract_and_convert('./datasets/optical+recognition+of+handwritten+digits/optdigits.tes')
    # first divided by 16 to normalize to highest value
    features_converted = (images * 0.0625 * feature_multiplier + feature_constant).reshape(64)
    # shifted to match XOR definition
    labels_converted = (labels - 0.5) * 2  * label_multiplier + amplitude_relative

    return amplitude_relative, features_converted, labels_converted

def OptDigits_SL_map_features_and_labels_with_initial_randomization(feature_multiplier, feature_constant, label_multiplier, amplitudes, outputn, rng_key):
    key1, key2 = jax.random.split(rng_key)
    # Create proper random matrices (8x8 for MNIST-like 8x8 optdigits)
    mulMat = jax.random.uniform(key1, shape=(8,8), minval=-0.5, maxval=0.5)  
    addMat = jax.random.uniform(key2, shape=(8,8), minval=-0.5, maxval=0.5)
    
    amplitude_relative = jnp.mean(amplitudes[outputn])
    
    images, labels = OptDigits_extract_and_convert('./datasets/optical+recognition+of+handwritten+digits/optdigits.tes')
    images = images * mulMat + addMat
    # first divided by 16 to normalize to highest value
    features_converted = (images * 0.0625 * feature_multiplier + feature_constant).reshape(-1,64)
    # shifted to match XOR definition
    labels_converted = (labels - 0.5) * 2  * label_multiplier + amplitude_relative

    return amplitude_relative, features_converted, labels_converted

def OptDigits_separate_training_and_test(features, labels):
    how_many = [0]*10
    training_features = []
    training_labels = []
    test_features = []
    test_labels = []
    for f, l in zip(features,labels):
        label = int(jnp.argmax(l))
        how_many[label] += 1
        if how_many[label] < 121:
            training_features.append(f)
            training_labels.append(l)
        else:
            test_features.append(f)
            test_labels.append(l)
    return jnp.array(training_features), jnp.array(training_labels), jnp.array(test_features), jnp.array(test_labels)

def label_retrive_original(label, amplitude_relative, label_multiplier):
    return (label- amplitude_relative)/label_multiplier*0.5+0.5

def map_amplitudes_to_probabilities(amplitude, outputn, amplitude_relative, label_multiplier):
    """
    returns values from 0 to 1 converted from output amplitudes according to rules defined in OptDigits_SL_map_features_and_labels and clipped to 0 and 1.
    """
    results = amplitude[outputn]
    probabilities = label_retrive_original(results, amplitude_relative, label_multiplier)
    return jnp.clip(probabilities,a_min=0,a_max=1)

def OptDigits_SL_determine_accuracy(amplitude, label, outputn, amplitude_relative, label_multiplier):
    probabilities = map_amplitudes_to_probabilities(amplitude, outputn, amplitude_relative, label_multiplier)
    label_original = label_retrive_original(label, amplitude_relative, label_multiplier)
    return float(jnp.argmax(label) == jnp.argmax(probabilities))


