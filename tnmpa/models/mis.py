from typing import Sequence

import networkx as nx
import numpy as np

from tnmpa.generators.random import generate_random_ksat_instance
from tnmpa.models import Factor, FactorGraph, Variable


class MIS(FactorGraph):
    def __init__(self, N, M, seed=None):
        graph = nx.gnm_random_graph(N, M, seed)
        nx.relabel_nodes(graph, {i: f"v{i}" for i in range(N)}, copy=False)
        factors = [
            Factor(f"c{k}", [Variable(e[0]), Variable(e[1])])
            for k, e in enumerate(graph.edges)
        ]
        super().__init__(factors)

    def hyper_tn(self):
        import quimb.tensor as qtn

        import tnmpa.solvers.tensor_factories as tfac

        tn = qtn.TensorNetwork([])
        data = np.ones((2, 2))
        data[1, 1] = 0.0
        for c in self.factors:
            # add one tensor for each clause
            tn_c = qtn.Tensor(
                data=data, inds=(c.variables[0].name, c.variables[1].name)
            )
            tn.add(tn_c, virtual=True)
        for v in tn.outer_inds():
            tn.add(
                qtn.tensor_core.COPY_tensor(
                    d=2,
                    inds=(v,),
                )
            )
        return tn
