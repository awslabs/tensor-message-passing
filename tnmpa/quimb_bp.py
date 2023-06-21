import functools
import operator

import autoray as ar
import numpy as np
from quimb.tensor.tensor_core import _inds_to_eq


def initialize_messages(tn):
    """Initialize messages for belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to initialize messages for.

    Returns
    -------
    messages : dict
        The initial messages. For every index and tensor id pair, there will
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
    """
    backend = ar.infer_backend(next(t.data for t in tn))
    _ones = ar.get_lib_fn(backend, "ones")

    messages = {}

    for ix, tids in tn.ind_map.items():
        d = tn.ind_size(ix)
        data = _ones(d) / d

        for tid in tids:
            # from hyperind to tensor
            messages[ix, tid] = data
            # from tensor to hyperind
            messages[tid, ix] = data

    return messages


def compute_all_hyperind_messages(ms, smudge_factor=1e-12):
    if len(ms) == 2:
        # shortcut for 2 messages
        return [ms[1], ms[0]]

    x = functools.reduce(operator.mul, ms)
    return [x / (m + smudge_factor) for m in ms]


def compute_all_hyperind_messages_alt(ms):
    ndim = len(ms)

    if len(ms) == 2:
        # shortcut for 2 messages
        return [ms[1], ms[0]]

    mouts = [None for _ in range(ndim)]
    queue = [(tuple(range(ndim)), 1, ms)]

    while queue:
        js, x, ms = queue.pop()

        ndim = len(ms)
        if ndim == 1:
            # reached single message
            mouts[js[0]] = x
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            mouts[js[0]] = x * ms[1]
            mouts[js[1]] = ms[0] * x
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        ml, mr = ms[:k], ms[k:]

        # contract the right messages to get new left array
        xl = functools.reduce(operator.mul, (*mr, x))

        # contract the left messages to get new right array
        xr = functools.reduce(operator.mul, (*ml, x))

        # add the queue for possible further halving
        queue.append((jl, xl, ml))
        queue.append((jr, xr, mr))

    return mouts


def compute_all_tensor_messages_alt(x, ms, _einsum):
    ndim = len(ms)

    if ndim == 2:
        # shortcut for 2 messages
        return [x @ ms[1], ms[0] @ x]

    js = tuple(range(ndim))
    eq = _inds_to_eq((js, *((j,) for j in js)), js)

    mx = _einsum(eq, x, *ms)
    mouts = []

    for j, g in enumerate(ms):
        eq = _inds_to_eq((js, (j,)), (j,))
        mouts.append(_einsum(eq, mx, 1 / (g + 1e-13)))

    return mouts


def compute_all_tensor_messages(x, ms, _einsum):
    ndim = len(ms)

    if ndim == 2:
        # shortcut for 2 messages
        return [x @ ms[1], ms[0] @ x]

    mouts = [None for _ in range(ndim)]
    queue = [(tuple(range(ndim)), x, ms)]

    while queue:
        js, x, ms = queue.pop()

        ndim = len(ms)
        if ndim == 1:
            # reached single message
            mouts[js[0]] = x
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            mouts[js[0]] = x @ ms[1]
            mouts[js[1]] = ms[0] @ x
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        ml, mr = ms[:k], ms[k:]

        # contract the right messages to get new left array
        eql = _inds_to_eq(
            (js, *((j,) for j in jr)),
            jl,
        )
        xl = _einsum(eql, x, *mr)

        # contract the left messages to get new right array
        eqr = _inds_to_eq(
            (js, *((j,) for j in jl)),
            jr,
        )
        xr = _einsum(eqr, x, *ml)

        # add the queue for possible further halving
        queue.append((jl, xl, ml))
        queue.append((jr, xr, mr))

    return mouts


def iterate_belief_propagation(tn, messages, smudge_factor=1e-12):
    """Run a single iteration of belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    messages : dict
        The current messages. For every index and tensor id pair, there should
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.

    Returns
    -------
    new_messages : dict
        The new messages.
    """
    backend = ar.infer_backend(next(iter(messages.values())))

    # _sum = ar.get_lib_fn(backend, "sum")
    # nb at small sizes python sum is faster than numpy sum
    _sum = sum
    _einsum = ar.get_lib_fn(backend, "einsum")
    _abs = ar.get_lib_fn(backend, "abs")

    new_messages = {}

    # hyper index messages
    for ix, tids in tn.ind_map.items():
        ms = compute_all_hyperind_messages(
            [messages[tid, ix] for tid in tids], smudge_factor
        )
        for tid, m in zip(tids, ms):
            new_messages[ix, tid] = m / _sum(m)

    # tensor messages
    for tid, t in tn.tensor_map.items():
        inds = t.inds
        ms = compute_all_tensor_messages(
            t.data, [messages[ix, tid] for ix in inds], _einsum
        )
        for ix, m in zip(inds, ms):
            new_messages[tid, ix] = m / _sum(m)

    max_dm = 0

    for k, m in new_messages.items():
        dm = _sum(_abs(m - messages[k]))
        max_dm = max(dm, max_dm)

    return new_messages, max_dm


def run_belief_propagation(
    tn,
    messages=None,
    max_iterations=1000,
    tol=1e-2,
    smudge_factor=1e-12,
    progbar=False,
):
    """Run belief propagation on a tensor network until it converges.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to run BP on.
    messages : dict, optional
        The current messages. For every index and tensor id pair, there should
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
        If not given, then messages are initialized as uniform.
    max_iterations : int, optional
        The maximum number of iterations to run for.
    tol : float, optional
        The convergence tolerance.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid division
        by zero. Note when this happens the numerator will also be zero.
    progbar : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    messages : dict
        The final messages.
    converged : bool
        Whether the algorithm converged.
    """
    # initialize messages
    if messages is None:
        messages = initialize_messages(tn)

    if progbar:
        import tqdm

        it = tqdm.tqdm(range(max_iterations))
    else:
        it = range(max_iterations)

    converged = False
    for _ in it:
        messages, max_dm = iterate_belief_propagation(
            tn,
            messages,
            smudge_factor=smudge_factor,
        )

        if progbar:
            it.set_description(f"{max_dm}", refresh=False)

        converged = max_dm < tol
        if converged:
            break

    return messages, converged


def get_marginals_from_messages(tn, messages):
    """Compute all index marginals from belief propagation messages.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute marginals for.
    messages : dict
        The belief propagation messages.

    Returns
    -------
    marginals : dict
        The marginals for each index.
    """
    marginals = {}
    for ix, tids in tn.ind_map.items():
        m = functools.reduce(operator.mul, (messages[tid, ix] for tid in tids))
        marginals[ix] = m / sum(m)
    return marginals


def sample_belief_propagation(
    tn,
    messages=None,
    max_iterations=1000,
    tol=1e-2,
    smudge_factor=1e-12,
    bias=False,
    progbar=False,
):
    """Sample all indices of a tensor network using belief propagation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to sample.
    messages : dict, optional
        The current messages. For every index and tensor id pair, there should
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
        If not given, then messages are initialized as uniform.
    max_iterations : int, optional
        The maximum number of iterations for each message passing run.
    tol : float, optional
        The convergence tolerance for each message passing run.
    smudge_factor : float, optional
        A small number to add to each message to avoid zeros. Making this large
        is similar to adding a temperature, which can aid convergence but likely
        produces less accurate marginals.
    bias : bool, optional
        Whether to bias the sampling towards the largest marginal. If ``False``
        (the default), then indices are sampled proportional to their
        marginals. If ``True``, then each index is 'sampled' to be its largest
        weight value always.
    progbar : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    config : dict[str, int]
        The sample configuration, mapping indices to values.
    tn_config : TensorNetwork
        The tensor network with all index values selected, leaving only
        scalars. The contracted value gives the weight of this sample, e.g.
        should be 1 for a SAT problem and valid assignment.
    omega : float
        The probability of choosing this sample (i.e. product of marginal
        values). Useful possibly for importance sampling.
    """
    tn_config = tn.copy()

    if messages is None:
        messages = initialize_messages(tn_config)

    config = {}
    omega = 1.0

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=len(tn_config.ind_map))
    else:
        pbar = None

    while tn_config.ind_map:
        messages, converged = run_belief_propagation(
            tn_config,
            messages,
            max_iterations=max_iterations,
            tol=tol,
            smudge_factor=smudge_factor,
        )

        if not converged:
            import warnings

            warnings.warn("BP did not converge.")

        marginals = get_marginals_from_messages(tn_config, messages)

        # choose largest bias
        ix, p = max(marginals.items(), key=lambda kv: abs(kv[1][0] - kv[1][1]))

        if bias:
            v = np.argmax(p)
            # in some sense omega is really 1.0 here
        else:
            # sample the value according to the marginal
            v = np.random.choice([0, 1], p=p)

        omega *= p[v]
        config[ix] = v

        # clean up messages
        for tid in tn_config.ind_map[ix]:
            del messages[ix, tid]
            del messages[tid, ix]

        # remove index
        tn_config.isel_({ix: v})

        if progbar:
            pbar.update(1)
            pbar.set_description(f"{ix}->{v}", refresh=False)

    if progbar:
        pbar.close()

    return config, tn_config, omega
