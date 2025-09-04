# Function to simulate the correlated infection vector

import numpy as np

def simulate_markov(n, transition_matrix, random_state=None):
    """
    Simulate an infection vector U^n ∈ {0,1}^n from a 2-state Markov chain.

    Parameters
    ----------
    n : int
        Length of the sequence to simulate (n >= 1).
    transition_matrix : array-like of shape (2, 2)
        Row-stochastic matrix P where P[i, j] = P(X_{t+1}=j | X_t=i).
        Rows must be nonnegative and (approximately) sum to 1.
    random_state : int or np.random.Generator, optional
        Seed or Generator for reproducibility.

    Returns
    -------
    U : np.ndarray of shape (n,), dtype=int
        Simulated 0/1 infection vector.
    pi : np.ndarray of shape (2,)
        Stationary distribution used for the initial state.
    """
    P = np.asarray(transition_matrix, dtype=float)
    if P.shape != (2, 2):
        raise ValueError("transition_matrix must be 2x2.")
    if np.any(P < -1e-12):
        raise ValueError("transition_matrix must be nonnegative.")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Each row of transition_matrix must sum to 1.")

    # RNG setup
    rng = (np.random.default_rng(random_state) if not isinstance(random_state, np.random.Generator)
           else random_state)

    # Compute stationary distribution pi satisfying pi = pi P, sum(pi)=1.
    # Solve (P^T - I)^T * pi = 0 with constraint sum(pi)=1.
    A = np.vstack([P.T - np.eye(2), np.ones((1, 2))])
    b = np.array([0.0, 0.0, 1.0])
    # Least-squares to be numerically stable
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    # Numerical cleanup
    pi = np.clip(pi, 0.0, 1.0)
    pi /= pi.sum()

    # Sample initial state from stationary distribution
    U = np.empty(n, dtype=int)
    U[0] = rng.choice(2, p=pi)

    # Evolve the chain
    for t in range(1, n):
        U[t] = rng.choice(2, p=P[U[t-1], :])

    return U, pi