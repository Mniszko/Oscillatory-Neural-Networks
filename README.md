<h1>OSCILLATORY NEURAL NETWORKS</h1>

<p>
Repository contains package with methods written in python with JAX for training <a href="https://www.semanticscholar.org/paper/Training-Coupled-Phase-Oscillators-as-a-Platform-Wang-Wanjura/55e70b62b96f5bac8d91b233b55c157925d5c618">oscillatory neural networks (ONNs) with equilibrium propagation (EP)</a> and code utilizing these methods for two separate models of oscillations, namely Kuramoto (K) model (or two different K models) and Stuart-Landau (SL) model.
</p>

<h2>PREREQUISITES</h2>
There is currently no requirements file, though as for January 2025 code (except for SL-dynamics.py file, written mostly for debugging) runs on newest python3 with jax, numpy, matplotlib and scipy libraries.

<h2>REPOSITORY STRUCTURE AND INSTRUCTIONS</h2>
<p>
<strong>plot_dis_and_ac.py</strong> file plots distances and accuracies recorded in <em>.txt</em> format. Accuracy recordings have a suffix <em>_acc.txt</em> automatically added by training algorithms (SL-training.py for SL model and K-training.py for K model of the first kind). Running it requires command of type (replace text with brackets with corresponding value of choice):

    python3 plot_dis_and_ac.py <output_distances_name_without_filetype> <y_or_n_for_distance_plotting> <lin_or_log_for_distance_scale> <y_or_n_for_accuracy_plotting> <output_image_name_without_filetype>

<br>
<strong>SL-dynamics.py</strong> is a code with different dependencies for plotting main parameters (amplitude and phase of the oscillator) for debugging purposes mostly. It operates solely on 
<br>
<strong>SL-training.py, K-training.py</strong> are programs of nearly identical structure, each designed for different oscillatory model. They are run individually or via <em>auto_run.sh</em> script. Individual execution can be done with command:

    python3 SL-training.py <output_distances_name_without_filetype> <number_of_neurons> <y_for_saving_data_n_for_plotting> <number_of_epochs> <learning_rate>

<strong>auto_run.sh</strong> is a script for automattic runs of multiple training simulations in bash. It doesn't require additional instructions.
<br>
<strong>src</strong> directory contains module with methods separated into individual files. basics.py stores basic functionalities, DoubleXORProblem.py XORProblem.py have methods for initializing datasets (number of these files is subject to change), InitializationModule has methods for initializing simulation parameters (compiled in NeuralNetwork files), StuartLandauNeuralNet.py and KuramotoNeuralNetwork.py contain dynamics, full parameter initialization and such.
</p>

<h2>ADDITIONAL INFORMATIONS</h2>

<p>

1. Currently SL model and training runs with complex values. Before running the code for polaritonic network one should multiply weights_real and weights_real_matrix by 0 after initialization in <em>SL-training.py</em> and comment out line responsible for updating weight values in the same file, namely, line that reads:

        weights_real_matrix -= learning_rate * weight_real_gradient * weight_update_mask

2. training parameters are updated according to equation from our notes, though before running next set of simulations I should add update matrices and vectors normalization according to some arbitrary rules, at least in cases where gradient descent produces values too small to make change or approaching infinity.
</p>
