import random

import numpy as np


def generate_random_ksat_instance(N, M, K, seed=None):
    """
    Generate a random K-SAT instance with `num_vars` variables,
    `num_clauses` clauses, and order `K`.

    Prameters
    ---------
    N: int
         Number of variables
    M: int
         The number of clauses
    K: int
         K of the K-SAT problem
    max_num_tries: Optional[int]
         Maximum nummber of attempts to find a new clause (for very dense
         problems).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # initialize the positions in each clause
    positions = np.array([np.sort(random.sample(range(N), K)) for _ in range(M)])
    values = np.array([np.random.randint(2, size=K, dtype=bool) for _ in range(M)])
    return positions, values
