import numpy as np
import quimb.tensor as qtn


def two_norm_bp_clause_tensor(clause):
    """
    Create a tensor representation of a clause for belief propagation.
    The indices of the tensor are chosen is such way that they can be
    composed with varaible tensors to form a TensorNetwork

    Parameters
    ---------
    clause: vectorized_tensorKSAT.cluase
        A representation of a clause

    Returns
    --------
    Tensor
    """

    data_tc = np.ones([2] * len(clause.pos))
    sl = tuple(1 if clause(p) else 0 for p in clause.pos)
    data_tc[sl] = 0

    tc = qtn.Tensor(
        data=np.kron(data_tc, data_tc),
        inds=tuple(f"p{p}_{clause.label}" for p in clause.pos),
        tags=("CLAUSE",) + clause.tags,
    )
    return tc


def bp_clause_tensor(clause):
    """
    Create a tensor representation of a clause for belief propagation.
    The indices of the tensor are chosen is such way that they can be
    composed with varaible tensors to form a TensorNetwork

    Parameters
    ---------
    clause: vectorized_tensorKSAT.cluase
        A representation of a clause

    Returns
    --------
    Tensor
    """

    data_tc = np.ones([2] * len(clause.variables))
    sl = tuple(1 if clause(p.name) else 0 for p in clause.variables)
    data_tc[sl] = 0

    tc = qtn.Tensor(
        data=data_tc,
        inds=tuple(f"{p.name}_{clause.name}" for p in clause.variables),
    )
    return tc


def bp_variable_tensor(variable, clauses, factor_threshold=8, open_leg=False):
    """
    Create a tensor representation of a variable for belief propagation.
    The indices of the tensor are chosen is such way that they can be
    composed with clause tensors to form a TensorNetwork.
    Essentially returns a GHZ tensor

    Parameters
    ---------
    variable: int
        The number assigned to a given variable
    clauses: sequence
        The sequaence of clauses the variable is connected to
    factor_threshold: int
        The threshold that determines if we want a dense or MPS representation

    Returns
    --------
    Tensor or MatrixProductState
    """

    # These are just delta tensors:
    # t[0,..,0] = 1, or
    # t[1,..,1] = 1,
    # and zero otherwise
    clabels = [c.name for c in clauses]
    copy_inds = [f"{variable}_" + c for c in clabels]

    if open_leg:
        copy_inds.append(f"{variable}")

    if len(clabels) >= factor_threshold:
        arrays = [
            cp.data for cp in qtn.tensor_core.COPY_mps_tensors(d=2, inds=copy_inds)
        ]
        mps = qtn.MatrixProductState(arrays)
        for i in range(mps.L):
            mps.reindex_sites(copy_inds[i], where=[i], inplace=True)
        mps.add_tag("VARIABLE")
        mps.add_tag("MPS")
        mps.add_tag(f"T{variable}")
        return mps
    else:
        return qtn.tensor_core.COPY_tensor(
            d=2, inds=copy_inds, tags=("VARIABLE", "MPS", f"T{variable}")
        )


def sp_variable_tensor(variable, clauses, factor_threshold=5):
    """
    Dispatch to dense or MatrixProductState representation of the
    tensors
    """

    if len(clauses) < factor_threshold:
        return dense_sp_var_tensor(variable, clauses)
    else:
        return factorized_sp_var_tensor(variable, clauses)


def sp_clause_tensor(clause):
    """
    Create a tensor representation of a clause for survey propagation.
    The indices of the tensor are chosen is such way that they can be
    composed with variable tensors to form a TensorNetwork

    In order to write tensors for survey propagation, we fist need to define
    input and ouput dimensions. For clauses tensors each leg is
    dimension five. By convention, we choose the input dimensions to be
    2, 3, 4 and output dimensions 0, 1 for all the indices.

    This tensors represent the survey propagation update rule:

    (Mezard, Montanari p.478)
    \hat{Q}_{ai} = \prod _{j \in \partial a \backslash i} Q^U_{ja}

    Parameters
    ---------
    clause: vectorized_tensorKSAT.cluase
        A representation of a clause

    Returns
    --------
    Tensor
    """

    # initialize an empty array
    data = np.zeros([5] * len(clause.pos))
    for k, p in enumerate(clause.pos):
        # Q^U is input in dimension 2
        # we collect all inputs for Q^U and output to 0
        idx = [2] * len(clause.pos)
        idx[k] = 0
        # set the transition to 1
        data[tuple(idx)] = 1

        # all other combinations input to 1
        idx = [slice(2, 5)] * len(clause.pos)
        idx[k] = 1
        data[tuple(idx)] = 1

        # the three lines above include an unwanted transition
        # the one from all Q^U to 1, we set it to zero
        idx = [2] * len(clause.pos)
        idx[k] = 1
        data[tuple(idx)] = 0

    inds = tuple(f"p{po}_{clause.label}" for po in clause.pos)
    return qtn.Tensor(data=data, inds=inds, tags=("CLAUSE",) + clause.tags)


def dense_sp_var_tensor(variable, clauses):
    """
    Create a tensor representation of a variable for survey propagation.
    The indices of the tensor are chosen is such way that they can be
    composed with clause tensors to form a TensorNetwork

    In order to write tensors for survey propagation, we fist need to define
    input and ouput dimensions. For clauses tensors each leg is
    dimension five. By convention, we choose the input dimensions to be
    0, 1 and output dimensions 2, 3, 4 for all the indices.

    This tensors represent the survey propagation update rule:

    (Mezard, Montanari p.478)
    Q^U_{ja} = \prod_{b \in S} (1-\hat{Q}_{bj}) (1 - \prod_{b \in U} (1-\hat{Q}_{bj}))
    Q^S_{ja} = \prod_{b \in U} (1-\hat{Q}_{bj}) (1 - \prod_{b \in S} (1-\hat{Q}_{bj}))
    Q^*_{ja} = \prod_{b \in \partial j \backslash a} (1-\hat{Q}_{bj})

    \hat{Q}_{ai} = \prod _{j \in \partial a \backslash i} Q^U_{ja}

    Parameters
    ---------
    variable: int
        The number assigned to a given variable
    clauses: sequence
        The sequaence of clauses the variable is connected to

    Returns
    --------
    Tensor
    """

    rank = len(clauses)
    data = np.zeros((5,) * rank)

    for a in range(rank):
        add_directional_tensor(a, rank, data, clauses, variable)

    clabels = [c.label for c in clauses]
    inds = tuple(f"p{variable}_" + cl for cl in clabels)

    return qtn.Tensor(data=data, inds=inds, tags=("VARIABLE", f"T{variable}"))


def add_directional_tensor(a, rank, data, clauses, variable):
    # see page 478 Mezard, Montanari
    # indices for output in Q^U
    idxU = [slice(2)] * rank
    # indices for output in Q^S
    idxS = [slice(2)] * rank
    for b in range(rank):
        if b != a and clauses[b](variable) == clauses[a](variable):
            # subset of clauses where the constraint
            # has the same sign as the output clause.
            # Thus they can be forced in the same direction
            # By construction, the clause tensors output \hat{Q}
            # on index zero. Thus we always have to pick index one
            idxU[b] = 1
        elif b != a and clauses[b](variable) != clauses[a](variable):
            # subset of clauses where the constraint
            # has the opposite sign as the output clause.
            # Thus they can be forced in the opposite direction
            # By construction, the clause tensors output \hat{Q}
            # on index zero. Thus we always have to pick index one
            idxS[b] = 1

    # by convention we output Q^U at index 2
    idxU[a] = 2
    data[tuple(idxU)] = 1

    # Exclude the case where the inputs are all ones.
    # it means there is no warning, the variable can take any value
    idxU = [1] * rank
    idxU[a] = 2
    data[tuple(idxU)] = 0

    # by convention we output Q^S at index 3
    idxS[a] = 3
    data[tuple(idxS)] = 1

    # Exclude the case where the inputs are all ones.
    # it means there is no warning, the variable can take any value
    idxS = [1] * rank
    idxS[a] = 3
    data[tuple(idxS)] = 0

    # send no warning to the star state at index 4
    idxstar = [1] * rank
    idxstar[a] = 4
    data[tuple(idxstar)] = 1


def factorized_sp_var_tensor(variable, clauses):
    """
    MatrixProductRepresentation of the same tensors `dense_sp_var_tensor`.
    Refer to documentation in `dense_sp_var_tensor`.
    """

    rank = len(clauses)
    zero_one = np.zeros(5)
    zero_one[:2] = 1
    one = np.zeros(5)
    one[1] = 1
    two = np.zeros(5)
    two[2] = 1
    three = np.zeros(5)
    three[3] = 1
    four = np.zeros(5)
    four[4] = 1

    for a in range(rank):
        arsU = [zero_one] * rank
        arsS = [zero_one] * rank
        for b in range(rank):
            if b != a:
                if clauses[b](variable) == clauses[a](variable):
                    arsU[b] = one
                else:
                    arsS[b] = one

        arsU[a] = two
        if a == 0:
            data = qtn.MPS_product_state(arsU)
        else:
            data += qtn.MPS_product_state(arsU)

        arsU = [one] * rank
        arsU[a] = two
        data -= qtn.MPS_product_state(arsU)

        arsS[a] = three
        data += qtn.MPS_product_state(arsS)

        arsS = [one] * rank
        arsS[a] = three
        data -= qtn.MPS_product_state(arsS)

        arsstar = [one] * rank
        arsstar[a] = four
        data += qtn.MPS_product_state(arsstar)

    clabels = [c.label for c in clauses]
    new_inds = tuple(f"p{variable}_" + cl for cl in clabels)
    index_map = {oi: ni for oi, ni in zip(data.site_inds, new_inds)}
    data.reindex(index_map, inplace=True)
    data.add_tag(f"T{variable}")
    data.add_tag("VARIABLE")
    data.add_tag("MPS")
    data.drop_tags([f"I{v}" for v in range(len(clabels))])

    return data
