"""Vectorized belief propagation for arbitrary `quimb` tensor networks.
"""
import functools
import operator

import autoray as ar
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


def initialize_batch_messages(tn, messages=None):
    """Initialize batched messages for belief propagation, as the uniform
    distribution.
    """
    if messages is None:
        messages = initialize_messages(tn)

    input_locs = {}
    output_locs = {}

    backend = ar.infer_backend(next(iter(messages.values())))
    _stack = ar.get_lib_fn(backend, "stack")
    _array = ar.get_lib_fn(backend, "array")

    # prepare index messages
    f = 0
    batched_inputs = {}
    for ix, tids in tn.ind_map.items():
        rank = len(tids)
        try:
            batch = batched_inputs[f, rank]
        except KeyError:
            batch = batched_inputs[f, rank] = [[] for _ in range(rank)]

        for i, tid in enumerate(tids):
            batch_i = batch[i]
            # position in the stack
            b = len(batch_i)
            input_locs[tid, ix] = (f, rank, i, b)
            output_locs[ix, tid] = (f, rank, i, b)
            batch_i.append(messages[tid, ix])

    # prepare tensor messages
    f = 1
    batched_tensors = {}
    for tid, t in tn.tensor_map.items():
        rank = t.ndim
        if rank == 0:
            continue

        try:
            batch = batched_inputs[f, rank]
            batch_t = batched_tensors[rank]
        except KeyError:
            batch = batched_inputs[f, rank] = [[] for _ in range(rank)]
            batch_t = batched_tensors[rank] = []

        for i, ix in enumerate(t.inds):
            batch_i = batch[i]
            # position in the stack
            b = len(batch_i)
            input_locs[ix, tid] = (f, rank, i, b)
            output_locs[tid, ix] = (f, rank, i, b)
            batch_i.append(messages[ix, tid])

        batch_t.append(t.data)

    # stack into single arrays
    for key, batch in batched_inputs.items():
        batched_inputs[key] = _stack(
            tuple(_stack(batch_i) for batch_i in batch)
        )
    for rank, tensors in batched_tensors.items():
        batched_tensors[rank] = _stack(tensors)

    # make numeric masks for updating output to input messages
    masks = {}
    for pair in input_locs:
        (fi, ranki, ii, bi) = input_locs[pair]
        (fo, ranko, io, bo) = output_locs[pair]
        key = (fi, ranki, fo, ranko)
        try:
            maskin, maskout = masks[key]
        except KeyError:
            maskin, maskout = masks[key] = [], []
        maskin.append([ii, bi])
        maskout.append([io, bo])

    for key, (maskin, maskout) in masks.items():
        masks[key] = _array(maskin), _array(maskout)

    return batched_inputs, batched_tensors, input_locs, output_locs, masks


def compute_all_hyperind_batch_messages_tree(bm):
    ndim = len(bm)

    if ndim == 2:
        # shortcut for 'bonds', which just swap places
        return ar.do("flip", bm, (0,))

    backend = ar.infer_backend(bm)
    _prod = ar.get_lib_fn(backend, "prod")
    _empty_like = ar.get_lib_fn(backend, "empty_like")

    bmo = _empty_like(bm)
    queue = [(tuple(range(ndim)), 1, bm)]

    while queue:
        js, x, bm = queue.pop()

        ndim = len(bm)
        if ndim == 1:
            # reached single message
            bmo[js[0]] = x
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            bmo[js[0]] = x * bm[1]
            bmo[js[1]] = bm[0] * x
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        bml, bmr = bm[:k], bm[k:]

        # contract the right messages to get new left array
        xl = x * _prod(bmr, axis=0)

        # contract the left messages to get new right array
        xr = _prod(bml, axis=0) * x

        # add the queue for possible further halving
        queue.append((jl, xl, bml))
        queue.append((jr, xr, bmr))

    return bmo


def compute_all_hyperind_batch_messages_prod(bm, smudge_factor=1e-12):
    backend = ar.infer_backend(bm)
    _prod = ar.get_lib_fn(backend, "prod")
    _reshape = ar.get_lib_fn(backend, "reshape")

    ndim = len(bm)
    if ndim == 2:
        # shortcut for 'bonds', which just swap
        return ar.do("flip", bm, (0,))

    combined = _prod(bm, axis=0)
    return _reshape(combined, (1, *ar.shape(combined))) / (bm + smudge_factor)


def compute_all_tensor_batch_messages_tree(bx, bm):
    """Compute all output messages for a stacked tensor and messages."""
    backend = ar.infer_backend_multi(bx, bm)
    _einsum = ar.get_lib_fn(backend, "einsum")
    _stack = ar.get_lib_fn(backend, "stack")

    ndim = len(bm)
    mouts = [None for _ in range(ndim)]
    queue = [(tuple(range(ndim)), bx, bm)]

    while queue:
        js, bx, bm = queue.pop()

        ndim = len(bm)
        if ndim == 1:
            # reached single message
            mouts[js[0]] = bx
            continue
        elif ndim == 2:
            # shortcut for 2 messages left
            mouts[js[0]] = _einsum("Xab,Xb->Xa", bx, bm[1])
            mouts[js[1]] = _einsum("Xa,Xab->Xb", bm[0], bx)
            continue

        # else split in two and contract each half
        k = ndim // 2
        jl, jr = js[:k], js[k:]
        ml, mr = bm[:k], bm[k:]

        # contract the right messages to get new left array
        eql = _inds_to_eq(((-1, *js), *((-1, j) for j in jr)), (-1, *jl))
        xl = _einsum(eql, bx, *(mr[i] for i in range(mr.shape[0])))

        # contract the left messages to get new right array
        eqr = _inds_to_eq(((-1, *js), *((-1, j) for j in jl)), (-1, *jr))
        xr = _einsum(eqr, bx, *(ml[i] for i in range(ml.shape[0])))

        # add the queue for possible further halving
        queue.append((jl, xl, ml))
        queue.append((jr, xr, mr))

    return _stack(tuple(mouts))


def compute_all_tensor_batch_messages_prod(bx, bm, smudge_factor=1e-12):
    backend = ar.infer_backend_multi(bx, bm)
    _einsum = ar.get_lib_fn(backend, "einsum")
    _stack = ar.get_lib_fn(backend, "stack")

    ndim = len(bm)
    x_inds = (-1, *range(ndim))
    m_inds = [(-1, i) for i in range(ndim)]
    eq = _inds_to_eq((x_inds, *m_inds), x_inds)
    bmx = _einsum(
        eq,
        bx,
        *bm,
        # optimize='greedy',
    )

    bminv = 1 / (bm + smudge_factor)

    mouts = []
    for i in range(ndim):
        # sum all but ith index, apply inverse gate to that
        eq = _inds_to_eq((x_inds, m_inds[i]), m_inds[i])
        mouts.append(_einsum(eq, bmx, bminv[i]))

    return _stack(mouts)


def _compute_output_single(
    f,
    rank,
    bm,
    batched_tensors,
    _reshape,
    _sum,
    smudge_factor=1e-12,
):
    if f:
        # tensor message
        bx = batched_tensors[rank]
        bmo = compute_all_tensor_batch_messages_tree(bx, bm)
        # bmo = compute_all_tensor_batch_messages_prod(bx, bm, smudge_factor)
    else:
        # index message
        # bmo = compute_all_hyperind_batch_messages_tree(bm)
        bmo = compute_all_hyperind_batch_messages_prod(bm, smudge_factor)

    # normalize
    bmo /= _reshape(_sum(bmo, axis=-1), (*ar.shape(bmo)[:-1], 1))

    return bmo


def compute_outputs(
    batched_inputs,
    batched_tensors,
    smudge_factor=1e-12,
    _pool=None,
):
    """Given stacked messsages and tensors, compute stacked output messages."""
    backend = ar.infer_backend(next(iter(batched_inputs.values())))
    _sum = ar.get_lib_fn(backend, "sum")
    _reshape = ar.get_lib_fn(backend, "reshape")

    batched_outputs = {}

    if _pool is None:
        for (f, rank), bm in batched_inputs.items():
            bmo = _compute_output_single(
                f, rank, bm, batched_tensors, _reshape, _sum, smudge_factor
            )
            batched_outputs[f, rank] = bmo

    else:
        for (f, rank), bm in batched_inputs.items():
            batched_outputs[f, rank] = _pool.submit(
                _compute_output_single,
                f,
                rank,
                bm,
                batched_tensors,
                _reshape,
                _sum,
                smudge_factor,
            )
        for key, f in batched_outputs.items():
            batched_outputs[key] = f.result()

    return batched_outputs


def _update_output_to_input_single(
    bi,
    bo,
    maskin,
    maskout,
    _max,
    _sum,
    _abs,
):
    # do a vectorized update
    select_in = (maskin[:, 0], maskin[:, 1], slice(None))
    select_out = (maskout[:, 0], maskout[:, 1], slice(None))
    bim = bi[select_in]
    bom = bo[select_out]

    # first check the change
    dm = _max(_sum(_abs(bim - bom), axis=-1))

    # update the input
    bi[select_in] = bom

    return dm


def update_outputs_to_inputs(
    batched_inputs, batched_outputs, masks, _pool=None
):
    """Update the stacked input messages from the stacked output messages."""
    backend = ar.infer_backend(next(iter(batched_outputs.values())))
    _max = ar.get_lib_fn(backend, "max")
    _sum = ar.get_lib_fn(backend, "sum")
    _abs = ar.get_lib_fn(backend, "abs")

    if _pool is None:
        dms = (
            _update_output_to_input_single(
                batched_inputs[fi, ranki],
                batched_outputs[fo, ranko],
                maskin,
                maskout,
                _max,
                _sum,
                _abs,
            )
            for (fi, ranki, fo, ranko), (maskin, maskout) in masks.items()
        )
    else:
        fs = [
            _pool.submit(
                _update_output_to_input_single,
                batched_inputs[fi, ranki],
                batched_outputs[fo, ranko],
                maskin,
                maskout,
                _max,
                _sum,
                _abs,
            )
            for (fi, ranki, fo, ranko), (maskin, maskout) in masks.items()
        ]
        dms = (f.result() for f in fs)

    return max(dms)


def extract_messages_from_inputs(batched_inputs, input_locs):
    """Get all messages as a dict from the batch stacked input form."""
    return {
        pair: batched_inputs[f, rank][i, b, :]
        for pair, (f, rank, i, b) in input_locs.items()
    }


def extract_messages_from_outputs(batched_outputs, output_locs):
    """Get all messages as a dict from the batch stacked output form."""
    return {
        pair: batched_outputs[f, rank][i, b, :]
        for pair, (f, rank, i, b) in output_locs.items()
    }


def iterate_vec_belief_propagation(
    batched_inputs,
    batched_tensors,
    masks,
    smudge_factor=1e-12,
    _pool=None,
):
    """ """
    batched_outputs = compute_outputs(
        batched_inputs,
        batched_tensors,
        smudge_factor=smudge_factor,
        _pool=_pool,
    )
    max_dm = update_outputs_to_inputs(
        batched_inputs, batched_outputs, masks, _pool=_pool
    )
    return batched_inputs, max_dm


def maybe_get_thread_pool(thread_pool):
    """Get a thread pool if requested."""
    if thread_pool is False:
        return None

    if thread_pool is True:
        import quimb as qu

        return qu.get_thread_pool()

    if isinstance(thread_pool, int):
        import quimb as qu

        return qu.get_thread_pool(thread_pool)

    return thread_pool


def run_belief_propagation(
    tn,
    messages=None,
    max_iterations=1000,
    tol=1e-2,
    smudge_factor=1e-12,
    thread_pool=False,
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

    pool = maybe_get_thread_pool(thread_pool)

    (
        batched_inputs,
        batched_tensors,
        input_locs,
        _,
        masks,
    ) = initialize_batch_messages(tn, messages)

    if progbar:
        import tqdm

        it = tqdm.tqdm(range(max_iterations))
    else:
        it = range(max_iterations)

    converged = False
    for _ in it:
        batched_inputs, max_dm = iterate_vec_belief_propagation(
            batched_inputs,
            batched_tensors,
            masks,
            smudge_factor=smudge_factor,
            _pool=pool,
        )

        if progbar:
            it.set_description(f"{max_dm}", refresh=False)

        converged = max_dm < tol
        if converged:
            break

    messages = extract_messages_from_inputs(batched_inputs, input_locs)

    return messages, converged


def compute_index_marginal(tn, ind, messages):
    """Compute the marginal for a single index given ``messages``.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute the marginal for.
    ind : int
        The index to compute the marginal for.
    messages : dict
        The messages to use, which should match ``tn``.

    Returns
    -------
    marginal : array_like
        The marginal probability distribution for the index ``ind``.
    """
    tids = tn.ind_map[ind]
    m = functools.reduce(operator.mul, (messages[tid, ind] for tid in tids))
    return m / ar.do("sum", m)


def compute_tensor_marginal(tn, tid, messages):
    """Compute the marginal for a single tensor/factor given ``messages``.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute the marginal for.
    tid : int
        The tensor id to compute the marginal for.
    messages : dict
        The messages to use, which should match ``tn``.

    Returns
    -------
    marginal : array_like
        The marginal probability distribution for the tensor/factor ``tid``.
    """
    t = tn.tensor_map[tid]

    t_eq_inds = tuple(range(t.ndim))
    eq_inds = [t_eq_inds]
    arrays = [t.data]

    for i, ix in enumerate(t.inds):
        mix = functools.reduce(
            operator.mul,
            (messages[otid, ix] for otid in tn.ind_map[ix] if otid != tid),
        )
        eq_inds.append((i,))
        arrays.append(mix)

    eq = _inds_to_eq(tuple(eq_inds), t_eq_inds)
    m = ar.do("einsum", eq, *arrays)
    return m / ar.do("sum", m)


def compute_entropy_from_messages(tn, messages, smudge_factor=1e-12):
    """Compute the entropy of a tensor network given ``messages``. The relevant
    equation is (14.25) of Mézard & Montanari.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute the entropy for.
    messages : dict
        The messages to use, which should match ``tn``.
    smudge_factor : float, optional
        A small number to add to the denominator of messages to avoid log(0).

    Returns
    -------
    H : float
        The entropy of the tensor network.
    """
    backend = ar.infer_backend(next(iter(messages.values())))
    _sum = ar.get_lib_fn(backend, "sum")
    _log = ar.get_lib_fn(backend, "log")

    H = 0

    for ix, tids in tn.ind_map.items():
        m = compute_index_marginal(tn, ix, messages)
        H -= (1 - len(tids)) * _sum(m * _log(m + smudge_factor))

    for tid in tn.tensor_map:
        m = compute_tensor_marginal(tn, tid, messages)
        H -= _sum(m * _log(m + smudge_factor))

    return H


def compute_free_entropy_from_messages(tn, messages):
    """Compute the free entropy, i.e. the log of the partition function,
    directly from the index and tensor messages without constructing any
    marginals. The relevant equation is (14.27) of Mézard & Montanari.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to compute the free entropy of.
    messages : dict
        The messages.

    Returns
    -------
    F : float
        The free entropy of the tensor network.
    """
    backend = ar.infer_backend(next(iter(messages.values())))
    _einsum = ar.get_lib_fn(backend, "einsum")
    _sum = ar.get_lib_fn(backend, "sum")
    _log = ar.get_lib_fn(backend, "log")

    F = 0.0

    for tid, t in tn.tensor_map.items():
        t_eq_inds = tuple(range(t.ndim))
        eq_inds = [t_eq_inds]
        arrays = [t.data]
        for i, ix in enumerate(t.inds):
            eq_inds.append((i,))
            arrays.append(messages[ix, tid])
        eq = _inds_to_eq(tuple(eq_inds), t_eq_inds)
        mt = _einsum(eq, *arrays)

        F += _log(_sum(mt))

    for ix, tids in tn.ind_map.items():
        mi = functools.reduce(
            operator.mul,
            (messages[tid, ix] for tid in tids),
        )
        F += _log(_sum(mi))

        for tid in tids:
            mit = messages[ix, tid] * messages[tid, ix]
            F -= _log(_sum(mit))

    return F


def compute_all_index_marginals_from_messages(tn, messages):
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
    return {ix: compute_index_marginal(tn, ix, messages) for ix in tn.ind_map}


def sample_belief_propagation(
    tn,
    messages=None,
    output_inds=None,
    max_iterations=1000,
    tol=1e-2,
    smudge_factor=1e-12,
    bias=False,
    thread_pool=False,
    seed=None,
    progbar=False,
):
    """Sample all indices of a tensor network using repeated belief propagation
    runs and decimation.

    Parameters
    ----------
    tn : TensorNetwork
        The tensor network to sample.
    messages : dict, optional
        The current messages. For every index and tensor id pair, there should
        be a message to and from with keys ``(ix, tid)`` and ``(tid, ix)``.
        If not given, then messages are initialized as uniform.
    output_inds : sequence of str, optional
        The indices to sample. If not given, then all indices are sampled.
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
    thread_pool : bool, int or ThreadPoolExecutor, optional
        Whether to use a thread pool for parallelization. If an integer, then
        this is the number of threads to use. If ``True``, then the number of
        threads is set to the number of cores. If a ``ThreadPoolExecutor``,
        then this is used directly.
    seed : int, optional
        A random seed to use for the sampling.
    progbar : bool, optional
        Whether to show a progress bar.

    Returns
    -------
    config : dict[str, int]
        The sample configuration, mapping indices to values.
    tn_config : TensorNetwork
        The tensor network with all index values (or just those in
        `output_inds` if supllied) selected. Contracting this tensor network
        (which will just be a sequence of scalars if all index values have been
        sampled) gives the weight of the sample, e.g. should be 1 for a SAT
        problem and valid assignment.
    omega : float
        The probability of choosing this sample (i.e. product of marginal
        values). Useful possibly for importance sampling.
    """
    import numpy as np

    pool = maybe_get_thread_pool(thread_pool)
    rng = np.random.default_rng(seed)

    tn_config = tn.copy()

    if messages is None:
        messages = initialize_messages(tn_config)

    if output_inds is None:
        output_inds = tn_config.ind_map.keys()
    output_inds = set(output_inds)

    config = {}
    omega = 1.0

    if progbar:
        import tqdm

        pbar = tqdm.tqdm(total=len(tn_config.ind_map))
    else:
        pbar = None

    while output_inds:
        messages, converged = run_belief_propagation(
            tn_config,
            messages,
            max_iterations=max_iterations,
            tol=tol,
            smudge_factor=smudge_factor,
            thread_pool=pool,
        )

        if not converged:
            import warnings

            warnings.warn("BP did not converge.")

        marginals = compute_all_index_marginals_from_messages(
            tn_config, messages
        )

        # choose largest bias
        ix, p = max(
            (m for m in marginals.items() if m[0] in output_inds),
            key=lambda kv: abs(kv[1][0] - kv[1][1]),
        )

        if bias:
            v = np.argmax(p)
            # in some sense omega is really 1.0 here
        else:
            # sample the value according to the marginal
            v = rng.choice([0, 1], p=p)

        omega *= p[v]
        config[ix] = v

        # clean up messages
        for tid in tn_config.ind_map[ix]:
            del messages[ix, tid]
            del messages[tid, ix]

        # remove index
        tn_config.isel_({ix: v})
        output_inds.remove(ix)

        if progbar:
            pbar.update(1)
            pbar.set_description(f"{ix}->{v}", refresh=False)

    if progbar:
        pbar.close()

    return config, tn_config, omega
