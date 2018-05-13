import numpy as np
from itertools import product

# Function to get max scoring previos tag and corresponding index.
def getMaxScore(path_scores, trans_scores, emission_score, r, c, L):
    max_score = float("-Inf")
    max_idx = -1
    max_cell_score = 0

    for i in range(L):
        score = path_scores[r-1][i] + trans_scores[i][c]
        if score > max_score:
            max_score = score
            max_idx = i

    max_cell_score = max_score + emission_score

    return (max_cell_score, max_idx)

# Function to get the max scoring path.
def getMaxScoringPath(path_matrix, start):
    path = []
    i = len(path_matrix) - 1
    j = start

    while path_matrix[i][j] != -1 and i > -1:
        path.append(path_matrix[i][j])
        j = path_matrix[i][j]
        i -= 1
        

    return path


def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    # Initialize variables.
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]
    # Path matrix to store back pointers for the scores.
    path_matrix = [[-1 for x in range(L)] for y in range(N)]
    # Score matrix that stores the score of the best path to a word-tag combination.
    score_matrix = [[0 for x in range(L)] for y in range(N)]

    # Adding initial start scores to emission scores.
    for i in range(L):
        score_matrix[0][i] = emission_scores[0][i] + start_scores[i]

    # Computing scores for individual word-tag pairs.
    for i in range(1,N):
        for j in range(L):
            score_matrix[i][j], path_matrix[i][j] = getMaxScore(score_matrix, trans_scores, emission_scores[i][j], i, j, L)

    # Adding the end scores and finding the max scoring sequence.
    start_score = float('-inf')
    start_idx = -1
    for i in range(L):
        score_matrix[N-1][i] = score_matrix[N-1][i] + end_scores[i]
        if score_matrix[N-1][i] > start_score:
            start_score = score_matrix[N-1][i]
            start_idx = i

    # Getting the max scoring path.
    path = getMaxScoringPath(path_matrix, start_idx)

    path = path[::-1]
    path.append(start_idx)

    # Asserting that the scores and path length are as expected.
    assert len(path) == N
    assert start_score != float("-Inf")

    return (start_score, path)


# Just a single main function to test the viterbi implementation. Can be commented out.
if __name__ == "__main__":
    maxN = 7
    maxL = 2
    N = np.random.randint(1, maxN+1)
    L = np.random.randint(2, maxL+1)

    emission_scores = np.random.normal(0.0, 1.0, (N,L))
    trans_scores = np.random.normal(0.0, 1.0, (L,L))
    start_scores = np.random.normal(0.0, 1.0, L)
    end_scores = np.random.normal(0.0, 1.0, L)

    # run viterbi
    (viterbi_s,viterbi_y) = run_viterbi(emission_scores, trans_scores, start_scores, end_scores)

    best_y = []
    best_s = -np.inf
    for y in product(range(L), repeat=N): # all possible ys
        # compute its score
        score = 0.0
        score += start_scores[y[0]]
        for i in xrange(N-1):
            score += trans_scores[y[i], y[i+1]]
            score += emission_scores[i,y[i]]
        score += emission_scores[N-1,y[N-1]]
        score += end_scores[y[N-1]]
        # update the best
        if score > best_s:
            best_s = score
            best_y = list(y)

    print viterbi_s, viterbi_y
    print best_s, best_y