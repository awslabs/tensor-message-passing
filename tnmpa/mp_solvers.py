import copy
import time

import numpy as np
import quimb.tensor as qtn

from .tensor_factories import (
    bp_clause_tensor,
    bp_variable_tensor,
    sp_clause_tensor,
    sp_variable_tensor,
    two_norm_bp_clause_tensor,
)


def decimation(decimation_func, instance, envs_tensors):
    """
    Heuristic algorithm for fixing a variable in message passing algorithms.
    It collects the biases from `decimation_func` and it uses them to
    make a decision on what variable to fix.

    Parameters
    ----------
    decimation_func: callable[dict[Union[str, int], np.ndarray]]
        A callable with signature decimation_func(np.ndarray, instance, v)
    instance: ksat_instance.KsatInstance
        The current K-SAT instance
    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments

    Returns
    -------
    fix_var: int
        The variable we want to fix
    bias: float
        A number between -1 and 1. If bias > 0 we fix fix_var to 0, otherwise to 1
    """

    max_bias = 0.0
    for v in instance.variables:
        p0, p1 = decimation_func(envs_tensors[v], instance, v)
        # choose the variable with the largest absolute bias
        if abs(p0 - p1) >= max_bias:
            fix_var = v
            bias = p0 - p1
            max_bias = abs(bias)

    return fix_var, bias


def bp_biases(envs, instance, v):
    """
    Heuristic algorithm for fixing a variable, depending on the
    surrounding environments of belief propagation.

    Parameters
    ----------
    envs: dict[str, numpy.ndarray]
        The set of enviroments around variable `v`
    instance: ksat_instance.KsatInstance
        The current K-SAT instance
    v: int
        The current variable

    Returns
    -------
    p0, p1: tuple[float, float]
        The BP biases
    """
    if len(envs.values()) > 0:
        p = np.prod(list(envs.values()), axis=0)
        p /= p.sum()
        return p[0], p[1]
    else:
        return 0.5, 0.5


def sp_biases(envs, instance, v):
    """
    Heuristic algorithm for fixing a variable, depending on the
    surrounding environments of survey propagation.
    [Refer to Mezard, Montanari p481 for more details] or
    https://onlinelibrary.wiley.com/doi/abs/10.1002/rsa.20057

    Parameters
    ----------
    envs: dict[str, np.ndarray]
        The set of enviroments around variable `v`
    instance: ksat_instance.KsatInstance
        The current K-SAT instance
    v: int
        The current variable

    Returns
    -------
    w0, w1: tuple[float, float]
        The SP biases
    """

    p0 = 1.0
    p1 = 1.0
    ws = 1.0

    # depending on the sign of the variable for the corresponding
    # clause, we update the probabilities in different ways
    for c_label, data in envs.items():
        # get the clause that correspond to the given env
        c = instance.clauses[c_label]
        # check the sign of the variable in this clause
        if c(v):
            p0 *= data[1]
        else:
            p1 *= data[1]

    # compute the warning probabilities according to
    # Mezard Montanari p482
    w1 = p0 * (1 - p1)
    w0 = p1 * (1 - p0)
    ws = p1 * p0
    n = w0 + w1 + ws
    w0 /= n
    w1 /= n

    return w0, w1


def sp_decimation(instance, envs_tensors):
    return decimation(sp_biases, instance, envs_tensors)


def bp_decimation(instance, envs_tensors):
    return decimation(bp_biases, instance, envs_tensors)


def two_norm_bp_biases(envs, instance, v):
    """
    Heuristic algorithm for fixing a variable, depending on the
    surrounding environments of belief propagation.

    Parameters
    ----------
    envs: dict[str, numpy.ndarray]
        The set of enviroments around variable `v`
    instance: ksat_instance.KsatInstance
        The current K-SAT instance
    v: int
        The current variable

    Returns
    -------
    p0, p1: tuple[float, float]
        The BP biases
    """
    if len(envs.values()) > 0:
        p = np.prod(list(envs.values()), axis=0)
        return p[0] / (p[0] + p[3]), p[3] / (p[0] + p[3])
    else:
        return 0.5, 0.5


def two_norm_bp_decimation(instance, envs_tensors):
    return decimation(two_norm_bp_biases, instance, envs_tensors)


def mp_clause_dense_update(new_envs, envs_tensors, cc, v, clauses, which="BP"):
    """
    Compute the updated message from variable `cc` to clause `v`, using
    survey propagation.
    Update `envs_tensors` inplace by using `new_envs`.

    Qi = \prod_i P_U

    P_U are the messages coming from the variable update step.

    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments
    new_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        A deep copy of envs_tensors
    cc: str
        The clause receiving the updated messages
    v: int
        The variable we want to update
    clauses: dict[str, ksat_instance.Clause]
        All the clauses in the current problem
    """

    vc = clauses[cc]
    if which == "BP":
        dense_c = bp_clause_tensor(clauses[cc])
    elif which == "SP":
        dense_c = sp_clause_tensor(clauses[cc])

    envs = [
        qtn.Tensor(
            data=d,
            inds=(f"p{p}_{cc}",),
        )
        for p, d in new_envs[cc].items()
        if p != v
    ]
    tensors = [dense_c] + envs
    out_env = qtn.tensor_contract(*tensors).data
    envs_tensors[v][cc] = out_env / out_env.sum()

    return np.sum(np.absolute(envs_tensors[v][cc] - new_envs[v][cc]))


def mp_variable_dense_update(new_envs, envs_tensors, cc, v, clauses, which="BP"):
    """
    Compute the updated message from variable `cc` to clause `v`, using
    belief propagation.
    Equivalent to a contraction with a tensor with all ones, except the
    for the indices that correspond to the values that violate the clause.

    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments
    new_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        A deep copy of envs_tensors
    cc: str
        The clause receiving the updated messages
    v: int
        The variable we want to update
    clauses: dict[str, ksat_instance.Clause]
        All the clauses in the current problem
    """

    v_clauses = [clauses[c_] for c_ in new_envs[v].keys()]
    # we need to fix the issue with copy tensor MPS
    if which == "BP":
        dense_v = bp_variable_tensor(v, v_clauses)
    elif which == "SP":
        dense_v = sp_variable_tensor(v, v_clauses)
    envs = [
        qtn.Tensor(
            data=d,
            inds=(f"p{v}_{c_}",),
        )
        for c_, d in new_envs[v].items()
        if c_ != cc
    ]

    if isinstance(dense_v, qtn.MatrixProductState):
        tn = qtn.TensorNetwork(envs)
        tn |= dense_v
        out_env = tn.contract()
    else:
        tensors = [dense_v] + envs
        out_env = qtn.tensor_contract(*tensors)
    envs_tensors[cc][v] = out_env.data / out_env.data.sum()

    return np.sum(np.absolute(envs_tensors[cc][v] - new_envs[cc][v]))


def sp_variable_update(new_envs, envs_tensors, cc, v, clauses):
    """
    Compute the updated message from variable `v` to clause `cc`, using
    survey propagation.
    Update `envs_tensors` inplace by using `new_envs`.

    OUT_U = P_S * (1 - P_U)
    OUT_S = P_U * (1 - P_S)
    OUT_star = P_S * P_U

    where
    P_S = \prod_{i \in S} (1 - Qi)
    P_U = \prod_{i \in U} (1 - Qi)

    S is the set of clauses where `v` has the same sign as in `cc`
    U is the set of clauses where `v` has the opposite sign as in `cc`

    Qi are the messages coming from the clause update step.

    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments
    new_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        A deep copy of envs_tensors
    cc: str
        The clause receiving the updated messages
    v: int
        The variable we want to update
    clauses: dict[str, ksat_instance.Clause]
        All the clauses in the current problem
    """

    # contribution from all the clauses where `v` has the same sign as in `cc`
    prod_S = np.prod(
        [
            eu[1]
            for keu, eu in new_envs[v].items()
            if (clauses[keu](v) == clauses[cc](v)) and (keu != cc)
        ]
    )

    # contribution from all the clauses where `v` has opposite sign as in `cc`
    prod_U = np.prod(
        [
            eu[1]
            for keu, eu in new_envs[v].items()
            if (clauses[keu](v) != clauses[cc](v)) and (keu != cc)
        ]
    )

    # update rule
    envs_tensors[cc][v][2] = prod_S * (1 - prod_U)
    envs_tensors[cc][v][3] = prod_U * (1 - prod_S)
    envs_tensors[cc][v][4] = prod_S * prod_U

    # check there are no zeros, i.e. we reached a contradiction
    if sum(envs_tensors[cc][v][2:]) < 1e-12:
        print(cc, v, sum(envs_tensors[cc][v][2:]))
        raise ValueError("Division by zero")

    # normalize the environments
    envs_tensors[cc][v][2:] /= np.sum(envs_tensors[cc][v][2:])

    # return the distances
    return np.sum(np.absolute(envs_tensors[cc][v] - new_envs[cc][v]))


def sp_clause_update(new_envs, envs_tensors, cc, v, clauses):
    """
    Compute the updated message from variable `cc` to clause `v`, using
    survey propagation.
    Update `envs_tensors` inplace by using `new_envs`.

    Qi = \prod_i P_U

    P_U are the messages coming from the variable update step.

    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments
    new_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        A deep copy of envs_tensors
    cc: str
        The clause receiving the updated messages
    v: int
        The variable we want to update
    clauses: dict[str, ksat_instance.Clause]
        All the clauses in the current problem
    """

    envs_tensors[v][cc][0] = np.prod(
        [eu[2] for vv, eu in new_envs[cc].items() if v != vv]
    )
    envs_tensors[v][cc][1] = 1 - envs_tensors[v][cc][0]

    return np.sum(np.absolute(envs_tensors[v][cc] - new_envs[v][cc]))


def dense_sp_update(new_envs, envs_tensors, cc, v, clauses):
    """
    Survey propagation update for this clause and variable
    """
    dist_v = mp_variable_dense_update(
        new_envs, envs_tensors, cc, v, clauses, which="SP"
    )
    dist_c = mp_clause_dense_update(new_envs, envs_tensors, cc, v, clauses, which="SP")

    return dist_c, dist_v


def sp_update(new_envs, envs_tensors, cc, v, clauses):
    """
    Survey propagation update for this clause and variable
    """
    dist_v = sp_variable_update(new_envs, envs_tensors, cc, v, clauses)
    dist_c = sp_clause_update(new_envs, envs_tensors, cc, v, clauses)

    return dist_c, dist_v


def dense_survey_propagation(envs_tensors, instance, **kwargs):
    return message_passing(dense_sp_update, envs_tensors, instance, **kwargs)


def survey_propagation(envs_tensors, instance, **kwargs):
    return message_passing(sp_update, envs_tensors, instance, **kwargs)


def bp_variable_update(new_envs, envs_tensors, cc, v):
    """
    Compute the updated message from variable `v` to clause `cc`, using
    belief propagation.
    Equivalent to a contraction with a delta tensor.

    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments
    new_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        A deep copy of envs_tensors
    cc: str
        The clause receiving the updated messages
    v: int
        The variable we want to update
    """

    p0 = np.prod([p[0] for k, p in new_envs[v].items() if k != cc])
    p1 = np.prod([p[1] for k, p in new_envs[v].items() if k != cc])

    envs_tensors[cc][v][0] = p0 / (p0 + p1)
    envs_tensors[cc][v][1] = p1 / (p0 + p1)

    return np.sum(np.absolute(envs_tensors[cc][v] - new_envs[cc][v]))


def bp_clause_update(new_envs, envs_tensors, cc, v, clauses):
    """
    Compute the updated message from variable `cc` to clause `v`, using
    belief propagation.
    Equivalent to a contraction with a tensor with all ones, except the
    for the indices that correspond to the values that violate the clause.

    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments
    new_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        A deep copy of envs_tensors
    cc: str
        The clause receiving the updated messages
    v: int
        The variable we want to update
    clauses: dict[str, ksat_instance.Clause]
        All the clauses in the current problem
    """

    vc = clauses[cc]
    prob = np.prod([p[int(vc(k))] for k, p in new_envs[cc].items() if k != v])

    idx = 1 if vc(v) else 0
    envs_tensors[v][cc][idx] = (1 - prob) / (2 - prob)
    envs_tensors[v][cc][1 - idx] = 1 - envs_tensors[v][cc][idx]

    return np.sum(np.absolute(envs_tensors[v][cc] - new_envs[v][cc]))


def dense_bp_update(new_envs, envs_tensors, cc, v, clauses):
    dist_v = mp_variable_dense_update(new_envs, envs_tensors, cc, v, clauses)
    dist_c = mp_clause_dense_update(new_envs, envs_tensors, cc, v, clauses)

    return dist_v, dist_c


def dense_belief_propagation(envs_tensors, instance, **kwargs):
    return message_passing(dense_bp_update, envs_tensors, instance, **kwargs)


def bp_update(new_envs, envs_tensors, cc, v, clauses):
    dist_v = bp_variable_update(new_envs, envs_tensors, cc, v)
    dist_c = bp_clause_update(new_envs, envs_tensors, cc, v, clauses)

    return dist_v, dist_c


def belief_propagation(envs_tensors, instance, **kwargs):
    return message_passing(bp_update, envs_tensors, instance, **kwargs)


def two_norm_bp_variable_update(new_envs, envs_tensors, cc, v):
    envs_tensors[cc][v][0] = np.prod([p[0] for k, p in new_envs[v].items() if k != cc])
    envs_tensors[cc][v][3] = np.prod([p[3] for k, p in new_envs[v].items() if k != cc])

    envs_tensors[cc][v][1] = 0.0
    envs_tensors[cc][v][2] = 0.0

    # envs_tensors[cc][v] /= np.linalg.norm(envs_tensors[cc][v])
    envs_tensors[cc][v] /= envs_tensors[cc][v][0] + envs_tensors[cc][v][3]
    # envs_tensors[cc][v] /= envs_tensors[cc][v].sum()
    return np.sum(np.absolute(envs_tensors[cc][v] - new_envs[cc][v]))


def two_norm_bp_clause_update(new_envs, envs_tensors, cc, v, clauses):
    vc = clauses[cc]
    dense_c = two_norm_bp_clause_tensor(clauses[cc])

    envs = [
        qtn.Tensor(
            data=d,
            inds=(f"p{p}_{cc}",),
        )
        for p, d in new_envs[cc].items()
        if p != v
    ]
    tensors = [dense_c] + envs
    out_env = qtn.tensor_contract(*tensors).data
    # envs_tensors[v][cc] = out_env / np.linalg.norm(out_env)
    envs_tensors[v][cc] = out_env / (out_env[0] + out_env[3])
    # envs_tensors[v][cc] = out_env / out_env.sum()
    return np.linalg.norm(envs_tensors[v][cc] - new_envs[v][cc], ord=1.0)


def two_norm_bp_update(new_envs, envs_tensors, cc, v, clauses):
    dist_v = two_norm_bp_variable_update(new_envs, envs_tensors, cc, v)
    dist_c = two_norm_bp_clause_update(new_envs, envs_tensors, cc, v, clauses)

    return dist_v, dist_c


def two_norm_belief_propagation(envs_tensors, instance, **kwargs):
    return message_passing(two_norm_bp_update, envs_tensors, instance, **kwargs)


def message_passing(
    func_update, envs_tensors, instance, tol=1e-2, max_iter=1000, store_distances=False
):
    """
    Basic message passing algorithm to solve a K-SAT instance.
    It stores two copies of all the environments and uses `new_envs` to
    update `envs_tensors`.
    The algorithm runs until the maximum distance across all old and new environments
    is smaller than `tol`

    Parameters
    ----------
    func_update: callable
        A function that perform a local update of the environments inplace and it returns
        the maximum distance
    envs_tensors: dict[Union[str, int], dict[Union[str, int], np.ndarray]]
        The set of all current environments
    instance: ksat_instance.KsatInstance
        The current K-SAT instance
    tol: float
        Convergence threshold
    max_iter: int
        Maximum number of iterations

    Returns
    -------
    status: dict
        Summary of message passing
    """

    clauses = instance.clauses
    variables = instance.variables

    count = 0
    max_distances = []

    new_envs = copy.deepcopy(envs_tensors)
    t0 = time.time()

    while True:
        t1 = time.time()
        count += 1
        max_dist = None

        distances = []
        for v in variables:
            for cc in envs_tensors[v].keys():
                dist_v, dist_c = func_update(new_envs, envs_tensors, cc, v, clauses)
                dist_envs = max(dist_v, dist_c)
                if store_distances:
                    distances.append(dist_v)
                    distances.append(dist_c)
                if (max_dist is None) or (dist_envs > max_dist):
                    max_dist = dist_envs

        max_distances.append(max_dist)
        if count >= max_iter:
            return {
                "max_distances": max_distances,
                "bp_converged": False,
                "iterations": count,
                "max_distance": max_dist,
                "tot_time": time.time() - t0,
                "iter_time": time.time() - t1,
                "distances": distances,
            }
        if max_dist < tol:
            return {
                "max_distances": max_distances,
                "bp_converged": True,
                "iterations": count,
                "max_distance": max_dist,
                "tot_time": time.time() - t0,
                "iter_time": time.time() - t1,
                "distances": distances,
            }
        new_envs = copy.deepcopy(envs_tensors)
