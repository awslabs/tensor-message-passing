from typing import Sequence

import networkx as nx

from tnmpa.generators.random import generate_random_ksat_instance
from tnmpa.models import Factor, FactorGraph, Variable


class Clause(Factor):
    def __init__(
        self, name: str, variables: Sequence[int], values: Sequence[bool]
    ) -> None:
        super().__init__(name, list([Variable(f"v{p}") for p in variables]))
        self.values = tuple(values)
        self.unsat_index = tuple(
            1 if self.__call__(p.name) else 0 for p in self.variables
        )

    def __eq__(self, other):
        return (
            (self.name == other.name)
            and (self.variables == other.variables)
            and (self.values == other.values)
        )

    def __call__(self, var):
        # return the value of var in this clause
        variables = [v.name for v in self.variables]
        return self.values[variables.index(var)]

    def __repr__(self):
        return (
            f"Clause({self.name}, variables: {self.variables}, values: {self.values})"
        )

    def remove_variable(self, var):
        # remove a variable from this clause
        # and update the other attributes accordingly
        variables = list(self.variables)
        values = list(self.values)

        idx_var = variables.index(var)

        del variables[idx_var]
        del values[idx_var]

        self.variables = tuple(variables)
        self.values = tuple(values)


class KSAT(FactorGraph):
    def __init__(self, N, M, K, seed=None):
        self.N = N
        self.M = M
        self.K = K
        self.seed = seed
        positions, values = generate_random_ksat_instance(N, M, K, seed=seed)
        factors = [Clause(f"c{k}", positions[k], values[k]) for k in range(M)]
        super().__init__(factors)

    def fix_variable(self, variable: str, value: bool) -> None:
        neighbors = list(self.graph.neighbors(variable))
        v = self.graph.nodes[variable]["data"]

        for clause in neighbors:
            if (
                self.graph.nodes[clause]["data"](v.name) != value
                or len(self.graph.nodes[clause]["data"].variables) == 0
            ):
                self.graph.remove_node(clause)
            else:
                self.graph.nodes[clause]["data"].remove_variable(v)

        self.graph.remove_node(variable)

    def get_clause(self, clause_name: str) -> Clause:
        return self.graph.nodes[clause_name]["data"]

    def hyper_tn(self):
        import quimb.tensor as qtn

        import tnmpa.solvers.tensor_factories as tfac

        tn = qtn.TensorNetwork([])
        for c in self.factors:
            # add one tensor for each clause
            tn_c = tfac.bp_clause_tensor(c, hyper_tn=True)
            tn.add(tn_c, virtual=True)
        for v in tn.outer_inds():
            tn.add(
                qtn.tensor_core.COPY_tensor(
                    d=2,
                    inds=(v,),
                )
            )
        return tn
