import jax.numpy as jnp

# Needs to be checked for accuracy:
def create_random_connections(num_neurons, min_connections=4, max_connections=None):
    """
    Create random, bidirectional connections between neurons with the following constraints:
    - Each neuron has at least `min_connections`.
    - Each neuron has at most `max_connections` (if specified).
    - Connections are bidirectional (symmetric).

    Args:
        num_neurons (int): Total number of neurons.
        min_connections (int): Minimum number of connections per neuron.
        max_connections (int or None): Maximum number of connections per neuron.

    Returns:
        jnp.ndarray: A binary connection matrix where connections[i, j] = 1 indicates
                     a connection between neuron i and neuron j.
    """
    # Initialize an empty connection matrix
    connections = np.zeros((num_neurons, num_neurons), dtype=int)

    # Ensure minimum connections for each neuron
    for neuron in range(num_neurons):
        while connections[neuron].sum() < min_connections:
            # Randomly choose another neuron to connect to
            target = np.random.choice([n for n in range(num_neurons) if n != neuron and connections[neuron, n] == 0])

            # Create bidirectional connection
            connections[neuron, target] = 1
            connections[target, neuron] = 1

    # Add some additional random connections to increase variability
    for _ in range(num_neurons * min_connections):
        source = np.random.choice(num_neurons)
        target = np.random.choice([n for n in range(num_neurons) if n != source and connections[source, n] == 0])

        # Check max_connections constraint
        if max_connections is not None:
            if connections[source].sum() >= max_connections or connections[target].sum() >= max_connections:
                continue

        # Create bidirectional connection
        connections[source, target] = 1
        connections[target, source] = 1

    return jnp.array(connections)

# Needs to be checked for accuracy:
def create_square_lattice_connections(grid_size):
    """
    Create connections between neurons arranged in a square lattice.
    Each neuron is connected to its immediate neighbors (up, down, left, right).

    Args:
        grid_size (int): The size of the square grid (e.g., grid_size x grid_size neurons).

    Returns:
        jnp.ndarray: A binary connection matrix where connections[i, j] = 1 indicates
                     a connection between neuron i and neuron j.
    """
    num_neurons = grid_size ** 2
    connections = np.zeros((num_neurons, num_neurons), dtype=int)

    def index(row, col):
        """Convert 2D grid coordinates to 1D index."""
        return row * grid_size + col

    for row in range(grid_size):
        for col in range(grid_size):
            current = index(row, col)

            # Connect to the right neighbor
            if col < grid_size - 1:
                right = index(row, col + 1)
                connections[current, right] = 1
                connections[right, current] = 1

            # Connect to the bottom neighbor
            if row < grid_size - 1:
                down = index(row + 1, col)
                connections[current, down] = 1
                connections[down, current] = 1

    return jnp.array(connections)


# Function to create weight update mask
def create_weight_update_mask(N, inputn):
    weight_update_mask = jnp.ones((N, N))
    """
    # uses selected weights as inputs
    for i in inputn:
        weight_update_mask = weight_update_mask.at[i, inputn].set(0)
        weight_update_mask = weight_update_mask.at[inputn, i].set(0)
    """
    # setting diagonal to 0s
    diagonal_indices = jnp.arange(weight_update_mask.shape[0])
    weight_update_mask = weight_update_mask.at[diagonal_indices, diagonal_indices].set(0)
    return weight_update_mask

def create_kuramoto_symmetric_weights(N, loc, scale, inputn, rng_key):
    """
    Create a symmetric weight matrix with random values, ensuring weights between neurons in inputn are zero.
    
    :param N: int, number of neurons
    :param loc: float, mean of the normal distribution
    :param scale: float, standard deviation of the normal distribution
    :param inputn: list or array, indices of neurons whose weights must be zero
    :param rng_key: jax.random.PRNGKey, random key for reproducibility
    :return: jnp.array, symmetric weight matrix
    """
    # Generate random numbers using jax.random
    random_values = jax.random.normal(rng_key, shape=(N, N)) * scale + loc
    # Create a lower triangular matrix
    lower_triangular = jnp.tril(random_values, k=-1)
    # Reflect it to make it symmetric
    symmetric_matrix = lower_triangular + lower_triangular.T
    # Set the diagonal to zero
    symmetric_matrix = symmetric_matrix.at[jnp.diag_indices(N)].set(0)
    # Set weights between neurons in inputn to zero
    for i in inputn:
        symmetric_matrix = symmetric_matrix.at[i, inputn].set(0)
        symmetric_matrix = symmetric_matrix.at[inputn, i].set(0)
    return symmetric_matrix

