import collections
import functools

import numpy as np
import quimb.tensor as qtn
from quimb.tensor.tensor_core import get_contractor, get_symbol


def HTN_ksat_random(
    n,
    k,
    alpha,
    seed=None,
    allow_repeat_variables=False,
    dtype="float64",
):
    """Generate a random k-SAT problem in the form of a hyper tensor network.

    These estimated satsifiability thresholds may be of interest (taken from
    arXiv:1810.05582)::

         k  |    alpha
         ---|---------
         2  |     1.00
         3  |     4.27
         4  |     9.93
         5  |    21.12
         6  |    43.37
         7  |    87.79
         8  |   176.54
         9  |   354.01
        10  |   708.92
        11  |  1418.71
        12  |  2838.28
        13  |  5677.41
        14  | 11355.67
        15  | 22712.20

    Parameters
    ----------
    n : int
        The number of variables in the problem. This will be the number of
        indices in the tensor network.
    k : int
        The number of literals per clause. This will be the number of
        dimensions each tensor has.
    alpha : float
        The density of clauses compared to variables. The mean number of
        tensors will be ``n * alpha``.
    seed : int, optional
        The seed for the random number generator.
    allow_repeat_variables : bool, optional
        Whether to allow variables to be repeated within a clause. If ``True``,
        Some of the tensors might have repeated indices, calling
        ``rank_simplify`` or ``full_simplify`` removes these and is probably
        desirable before contraction.
    dtype : str, optional
        The data type of the arrays, defaults to ``float64``.

    Returns
    -------
    TensorNetwork
    """
    rng = np.random.default_rng(seed)
    m = rng.poisson(alpha * n)

    def random_clause_data():
        # random OR clause: choose a single invalid local configuration
        # uniformly, i.e. a single entry to be zero, all others to be one
        data = np.ones([2] * k, dtype=dtype)
        flat_idx = rng.integers(0, data.size)
        idx = np.unravel_index(flat_idx, data.shape)
        data[idx] = 0
        return data

    all_vars = np.arange(n)

    def random_variables():
        return rng.choice(all_vars, size=k, replace=allow_repeat_variables)

    tn = qtn.TensorNetwork([])
    for i in range(m):
        data = random_clause_data()
        variables = random_variables()
        inds = [f"v{v}" for v in variables]
        tags = f"CLAUSE{i}"
        tn |= qtn.Tensor(data=data, inds=inds, tags=tags)

    return tn


def get_eq(N):
    """Generate a batched message contraction equation for order ``N``.

    Examples
    --------

        >>> get_eq(2)
        'da,db,dabc->dc'

    """
    ixf = get_symbol(N)
    ixb = get_symbol(N + 1)
    # vector factors
    terms = [ixb + get_symbol(i) for i in range(N)]
    # main tensor
    terms.append(ixb + "".join(map(get_symbol, range(N))) + ixf)
    return ",".join(terms) + "->" + ixb + ixf


def setup_vbp(
    tn,
    messages=None,
    optimize="dp",
):
    """This parses a tensor network into the required stacked
    arrays, expressions and masks to perform vectorized belief
    propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to perform belief propagation on. It should not have
        any hyper indices.
    messages : dict or callable, optional
        A dictionary of messages to initialize the algorithm with. The keys
        should be tuples of the form ``(ix, tid)`` where ``ix`` is the index
        and ``tid`` is the identifier of the tensor that the message is
        directed to. If not given assumed to be uniform.
        If a callable is given it should take a single argument, the size of
        the index, and return the initial message.

    Returns
    -------
    inputs : dict
        A dictionary of the input arrays for each order of message. The key is
        ``(N, k)`` where ``N`` is the order of the message and ``k`` is which
        term in the contraction equation it corresponds to.
    outs : dict
        A dictionary of the output arrays for each order of message, this will
        initially be the first set of messages. The key is the order of the
        contraction, ``N``.
    exprs : dict
        A dictionary of the contraction expressions for each order of message.
        The key is the order of the contraction, ``N``.
    maskin : dict
        A dictionary of the input masks for each order of message. The key is
        ``(No, Ni, k)`` where ``No`` is the order of the output message, ``Ni``
        is the order of the input message and ``k`` is which term in the
        contraction equation it corresponds to.
    maskout : dict
        The same as ``maskin`` but for the output masks.
    output_locs : dict
        A dictionary of the locations of the output messages. The key is
        ``(ix, tid)`` where ``ix`` is the index and ``tid`` is the identifier
        of the tensor that the message is directed to. The value is a tuple
        ``(N, B)`` where ``N`` is the order of the message and ``B`` is the
        batch index.
    """
    # the set of array inputs
    inputs = {}

    # where each message is output
    output_locs = {}
    outs = collections.defaultdict(list)

    # where each message needs to be input
    input_locs = collections.defaultdict(list)

    if messages is None:
        # default to uniform initial messages
        messages = functools.lru_cache(lambda d: np.ones(d) / d)

    if callable(messages):
        message_factory = messages
        messages = {}
        for ix, tids in tn.ind_map.items():
            d = tn.ind_size(ix)
            for tid in tids:
                messages[ix, tid] = message_factory(d)

    for tid_src, t in tn.tensor_map.items():
        # order of the expression: number of product vectors
        N = t.ndim - 1
        if N not in inputs:
            # initialize empty array stacks (N vectors + actual tensor)
            inputs[N] = [[] for _ in range(N + 1)]

        for ix in t.inds:
            # the other tensor tid, that we are 'pointing' to
            (tid_dest,) = (tid for tid in tn.ind_map[ix] if tid != tid_src)

            # index of the current stack
            B = len(inputs[N][0])

            # the message from tid_src to tid_dest via `ix` is stored here
            output_locs[(ix, tid_dest)] = N, B
            outs[N].append(messages[ix, tid_dest])

            # add every other index as a input
            inds = [jx for jx in t.inds if jx != ix]
            for k, jx in enumerate(inds):
                # k: which term in the expression to append the message to
                inputs[N][k].append(messages[jx, tid_src])
                input_locs[(jx, tid_src)].append((N, k, B))

            # need to transpose so output index is last
            inputs[N][N].append(np.moveaxis(t.data, t.inds.index(ix), -1))

    # -> convert from lists to arrays
    inputs = {k: list(map(np.stack, v)) for k, v in inputs.items()}
    outs = {k: np.stack(v) for k, v in outs.items()}

    # these masks control copying output messages into inputs
    maskin = collections.defaultdict(list)
    maskout = collections.defaultdict(list)

    for (ix, tid), (No, B) in output_locs.items():
        for Ni, k, jB in input_locs[(ix, tid)]:
            maskin[(No, Ni, k)].append(jB)
            maskout[(No, Ni, k)].append(B)

    maskin = {k: np.array(v) for k, v in maskin.items()}
    maskout = {k: np.array(v) for k, v in maskout.items()}

    # these are the actual contractions in the form of 'expressions'
    exprs = {}
    for N, arrays in inputs.items():
        eq = get_eq(N)
        shapes = [x.shape for x in arrays]
        exprs[N] = get_contractor(
            eq,
            *shapes,
            use_cotengra=True,
            optimize=optimize,
        )

    return inputs, outs, exprs, maskin, maskout, output_locs


def iterate_vbp(inputs, outs, exprs, maskin, maskout):
    """This performs one iteration of BP given the parsed data structures."""
    # compute new output messages
    for N, arrays in inputs.items():
        outs[N] = exprs[N](*arrays)

    # renormalize to prob distribution
    for N, out in outs.items():
        out /= np.expand_dims(out.sum(axis=1), axis=1)

    # copy output messages into inputs
    for no, ni, k in maskin:
        inputs[ni][k][maskin[no, ni, k]] = outs[no][maskout[no, ni, k]]

    return inputs, outs


def get_messages(outs, output_locs):
    """Extract the messages from the output arrays.

    Parameters
    ----------
    outs : dict
        The output arrays, keyed by the order of the message.
    output_locs : dict
        The locations of each message, keyed by ``(ix, tid)`` where ``ix`` is
        the index and ``tid`` is the identifier of the tensor that the message
        is directed to. The value is a tuple ``(N, B)`` where ``N`` is the
        order of the message and ``B`` is the batch index.

    Returns
    -------
    messages : dict
        The messages, keyed by ``(ix, tid)`` where ``ix`` is the index and
        ``tid`` is the identifier of the tensor that the message is directed
        to. The value is the message array.
    """
    return {(ix, tid): outs[N][B] for (ix, tid), (N, B) in output_locs.items()}
