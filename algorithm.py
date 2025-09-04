# Functions to design testing matrices, compute test outcomes, and decode infection vectors

import numpy as np

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

def block_testing(n, num_tests, block_size):
    """
    Create testing matrix X^{nxnum_tests} based on block coding
    parameters:
    n: int
        The number of individuals
    num_tests: int
        The number of tests to perform
    block_size: int
        The size of each block
    """    
    return testing_matrix

def compute_test_outcomes(infection_vector, testing_matrix):
    """
    Compute the test outcomes Y^{num_tests} based on the infection vector and testing matrix
    parameters:
    infection_vector: array-like of shape (n,)
        The infection vector of 0's and 1's
    testing_matrix: array-like of shape (n, num_tests)
        The testing matrix of 0's and 1's
    """
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