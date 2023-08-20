from typing import Sequence

import networkx as nx

from tnmpa.generators.random import generate_random_ksat_instance


class Variable:
    """Base class for a variables, in a factor."""

    def __init__(self, name: str, dim: int = 2) -> None:
        self.name = name
        self.dim = dim

    def __eq__(self, other) -> bool:
        return (self.name == other.name) and (self.dim == other.dim)

    def __repr__(self) -> str:
        return f"Variable({self.name}, dim={self.dim})"


class Factor:
    """Base class for a factor, an element of a factor graph."""

    def __init__(self, name: str, variables: Sequence[Variable]) -> None:
        self.name = name
        self.variables = tuple(variables)

    def __eq__(self, other) -> bool:
        return (self.name == other.name) and (self.variables == other.variables)


class FactorGraph:
    def __init__(self, factors: Sequence[Factor] = []):
        self.graph = nx.Graph()
        for f in factors:
            self.add_factor(f)

    def add_factor(self, factor: Factor):
        self.graph.add_node(factor.name, bipartite=1, data=factor)
        for v in factor.variables:
            self.graph.add_node(v.name, bipartite=0, data=v)
            self.graph.add_edge(factor.name, v.name)

    @property
    def factors(self):
        return [
            self.graph.nodes[n]["data"]
            for n in self.graph.nodes
            if self.graph.nodes[n]["bipartite"] == 1
        ]

    @property
    def variables(self):
        return [n for n in self.graph.nodes if self.graph.nodes[n]["bipartite"] == 0]

    @property
    def degree_variables(self):
        return {
            k: v
            for k, v in dict(self.graph.degree).items()
            if self.graph.nodes[k]["bipartite"] == 0
        }

    @property
    def degree_factors(self):
        return {
            k: v
            for k, v in dict(self.graph.degree).items()
            if self.graph.nodes[k]["bipartite"] == 1
        }


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


class KsatInstance(FactorGraph):
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

        import tnmpa.tensor_factories as tfac

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
