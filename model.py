# Function to simulate the correlated infection vector

import numpy as np

def simulate_markov(n, alpha, beta, seed=None):
    """
    Simulate an infection vector U^n from a 2-state Markov chain
    with transition probabilities:
        P(0->1) = alpha, P(0->0) = 1-alpha
        P(1->0) = beta,  P(1->1) = 1-beta
    
    Parameters
    ----------
    n : int
        Length of the sequence to simulate.
    alpha : float
        Transition probability from 0 to 1.
    beta : float
        Transition probability from 1 to 0.
    seed : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    U : np.ndarray
        Infection vector of length n with entries in {0,1}.
    pi : tuple
        Stationary distribution (pi0, pi1).
    """

    # Generate a random number. A constant seed aids reproducibility.
    rng = np.random.default_rng(seed)

    # stationary distribution
    pi0 = alpha / (alpha + beta)
    pi1 = beta / (alpha + beta)

    # initialize state
    U = np.empty(n, dtype=int)
    U[0] = rng.choice([0, 1], p=[pi0, pi1])

    # simulate chain
    for t in range(1, n):
        if U[t-1] == 0:
            U[t] = rng.choice([0, 1], p=[1 - alpha, alpha])
        else:
            U[t] = rng.choice([0, 1], p=[beta, 1 - beta])

    return U, (pi0, pi1)