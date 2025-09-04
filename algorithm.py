# Functions to design testing matrices, compute test outcomes, and decode infection vectors

import numpy as np
import math

def independent_testing(n, num_tests, testing_prob, seed=None):
    """
    Create testing matrix X^{n x num_tests} based on independent coding.

    Each entry X[i,j] ~ Bernoulli(testing_prob), independently.

    Parameters
    ----------
    n : int
        The number of individuals.
    num_tests : int
        The number of tests to perform.
    testing_prob : float
        The probability of including an individual in a test.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    testing_matrix : np.ndarray of shape (n, num_tests)
        Binary matrix indicating which individuals are in which tests.
    """
    rng = np.random.default_rng(seed)
    testing_matrix = rng.binomial(1, testing_prob, size=(n, num_tests))
    return testing_matrix

def block_testing(n, num_tests, block_size, p1, p2, seed=None):
    """
    Create testing matrix X^{n x num_tests} based on the two-stage block design.

    For each test (each column):
      1) Draw block selectors W_ell ~ Bern(p1) independently for each block (n_blocks = ceil(n / block_size)).
      2) Draw item selectors Z_j ~ Bern(p2) independently for each individual j = 1..n.
      3) Set X_{j} = (W_{block(j)} AND Z_j).

    Parameters
    ----------
    n : int
        The number of individuals.
    num_tests : int
        The number of tests to perform.
    block_size : int
        The size C of each (contiguous) block; last block may be smaller if n % C != 0.
    p1 : float
        Probability a block is selected in the coarse phase.
    p2 : float
        Probability an item is selected in the fine phase (given its block is selected).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    testing_matrix : np.ndarray of shape (n, num_tests), dtype=int
        Binary matrix indicating which individuals are in which tests.
    """
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if n <= 0 or num_tests <= 0:
        return np.zeros((max(n, 0), max(num_tests, 0)), dtype=int)

    rng = np.random.default_rng(seed)
    testing_matrix = np.zeros((n, num_tests), dtype=int)

    n_blocks = math.ceil(n / block_size)

    # Precompute each individual's block index (0..n_blocks-1), handling a short last block.
    idx = np.arange(n)
    block_idx_of_item = idx // block_size  # integer division; ok even if last block is short

    for t in range(num_tests):
        # Step 1: block selectors W_ell ~ Bern(p1)
        W = rng.binomial(1, p1, size=n_blocks).astype(int)  # shape (n_blocks,)

        # Broadcast W to items: W_of_item[j] = W_{block(j)}
        W_of_item = W[block_idx_of_item]  # shape (n,)

        # Step 2: item selectors Z_j ~ Bern(p2)
        Z = rng.binomial(1, p2, size=n).astype(int)

        # Step 3: AND per the paper: X_{j} = W_{block(j)} * Z_j
        testing_matrix[:, t] = (W_of_item & Z)

    return testing_matrix

def compute_test_outcomes(infection_vector, testing_matrix):
    """
    Compute the test outcomes Y^{num_tests} based on the infection vector and testing matrix.

    A test outcome is 1 if at least one infected individual is included in the test.

    Parameters
    ----------
    infection_vector : array-like of shape (n,)
        The infection vector of 0's and 1's.
    testing_matrix : array-like of shape (n, num_tests)
        The testing matrix of 0's and 1's.

    Returns
    -------
    test_outcomes : np.ndarray of shape (num_tests,)
        Binary test outcomes.
    """
    infection_vector = np.asarray(infection_vector, dtype=int)
    testing_matrix = np.asarray(testing_matrix, dtype=int)

    # Matrix multiplication gives counts of infected individuals in each test
    infected_counts = infection_vector @ testing_matrix  

    # Convert counts to binary outcomes
    test_outcomes = (infected_counts > 0).astype(int)

    return test_outcomes

def decoding(test_outcomes, testing_matrix, threshold):
    """
    Decode the infection vector from the test outcomes and testing matrix
    parameters:
    test_outcomes: array-like of shape (num_tests,)
        The test outcomes of 0's and 1's
    testing_matrix: array-like of shape (n, num_tests)
        The testing matrix of 0's and 1's
    threshold: int
        The threshold for decoding
    """
    return decoded_infection_vector