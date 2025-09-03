# Function to simulate the correlated infection vector

def simulate_markov(n, transition_matrix):
    """
    Simulate an Infection Vector U^n of (0,1)'s using a transition_matrix
    The transition_matrix is a 2x2 matrix where the rows sum to 1
    The initial state is chose according to the stationary distribution
    parameters:
    n: int
        Length of the sequence to simulate
        transition_matrix: 2x2 array-like
    """