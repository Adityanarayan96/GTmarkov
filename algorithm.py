# Functions to design testing matrices, compute test outcomes, and decode infection vectors

def independent_testing(n, num_tests, testing_prob):
    """
    Create testing matrix X^{nxnum_tests} based on independent coding
    parameters:
    n: int
        The number of individuals
    num_tests: int
        The number of tests to perform
    testing_prob: float
        The probability of including an individual in a test    
    """    
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