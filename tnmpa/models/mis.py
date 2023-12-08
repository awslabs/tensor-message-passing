from typing import Sequence, Union

import networkx as nx
import numpy as np

from tnmpa.generators.random import generate_random_ksat_instance
from tnmpa.models import Factor, FactorGraph, Variable


class MIS(FactorGraph):
    """A simple class to represent the maximally independent set problem
    P(x) = exp(\sum_i mu[i] \sigma_i) \Prod_{ij} (1 - \sigma_i \sigma_j)
    For more stat mech details, see https://arxiv.org/abs/1306.4121
    """

    def __init__(
        self, N: int, M: int, mu: Union[float, Sequence[float]] = 1.0, seed: int = None
    ):
        """Initialize a graph for a MIS problem with N vertices and M edges."""
        self.mu = mu
        super().__init__([])
        self.graph = nx.gnm_random_graph(N, M, seed)
        remap = {n: f"v{n}" for n in self.graph.nodes}
        nx.relabel_nodes(self.graph, remap, copy=False)

    def hyper_tn(self):
        import quimb.tensor as qtn

        tn = qtn.TensorNetwork([])
        data = np.ones((2, 2))
        data[1, 1] = 0.0
        for e in self.graph.edges:
            # add one tensor for each clause
            tn_c = qtn.Tensor(data=data, inds=e)
            tn.add(tn_c, virtual=True)
        for v in self.graph.nodes:
            if hasattr(self.mu, "__len__"):
                data = np.exp([1, self.mu[v]])
            else:
                data = np.exp([1, self.mu])
            tn_c = qtn.Tensor(data=data, inds=(v,))
            tn.add(tn_c, virtual=True)

        for v in tn.outer_inds():
            tn.add(
                qtn.tensor_core.COPY_tensor(
                    d=2,
                    inds=(v,),
                )
            )
        return tn
