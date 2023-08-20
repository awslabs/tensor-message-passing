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
