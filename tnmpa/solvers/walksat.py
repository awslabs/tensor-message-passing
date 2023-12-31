import random

from networkx.algorithms import bipartite


class WalkSAT:
    """Basic implementation of WalkSAT to solve a k-SAT instance."""

    def __init__(self, instance):
        self.instance = instance
        init_guess = [bool(random.getrandbits(1)) for _ in self.instance.variables]
        self.solution = dict(zip(self.instance.variables, init_guess))
        self.collect_violated_clauses()

    def collect_violated_clauses(self):
        count = 0
        self.violated_clauses = []
        clause_names, _ = bipartite.sets(self.instance.graph)
        for c in clause_names:
            c = self.instance.graph.nodes[c]["data"]
            # if the solutino is only partial, this will fail
            vals = tuple(self.solution[p.name] for p in c.variables)
            # if a clause is violated, the variables take the same values as in the unsat_index
            if vals == c.unsat_index:
                self.violated_clauses.append(c.name)

    def solve(self, max_flips, mixing):
        """Solve this instnace"""
        iter_no = 0

        for iter_no in range(max_flips):
            if iter_no % 500 == 0:
                print(f"Iter: {iter_no}, energy: {len(self.violated_clauses)}")

            if len(self.violated_clauses) == 0:
                print(f"Solution found")
                return True

            r = random.random()

            if r < 1 - mixing:
                min_delta = None
                for v in self.instance.variables:
                    delta = self.compute_delta(v)
                    if min_delta == None:
                        min_delta = delta
                        flip_var = v
                    elif delta < min_delta:
                        min_delta = delta
                        flip_var = v

            else:
                flip_clause = random.sample(self.violated_clauses, 1)[0]
                v = random.sample(
                    self.instance.graph.nodes[flip_clause]["data"].variables, 1
                )[0]
                flip_var = v.name
            # delta = self.compute_delta(flip_var)
            self.solution[flip_var] = not self.solution[flip_var]
            self.update_clauses(flip_var)
        return False

    def update_clauses(self, flip_var):
        clauses_at_v = self.instance.graph.neighbors(flip_var)
        for c in clauses_at_v:
            clause = self.instance.graph.nodes[c]["data"]
            vals = tuple(self.solution[p.name] for p in clause.variables)
            if (vals == clause.unsat_index) and (c not in self.violated_clauses):
                self.violated_clauses.append(c)
            elif (vals != clause.unsat_index) and (c in self.violated_clauses):
                self.violated_clauses.remove(c)

    def compute_delta(self, v):
        clauses_at_v = self.instance.graph.neighbors(v)
        v_val = self.solution[v]

        delta = 0
        for c in clauses_at_v:
            clause = self.instance.graph.nodes[c]["data"]
            variables = [p.name for p in clause.variables]
            vals = tuple(self.solution[p] for p in variables)
            en = 1 if vals == clause.unsat_index else 0

            idx_v = variables.index(v)
            f_vals = list(vals)
            f_vals[idx_v] = not f_vals[idx_v]
            f_vals = tuple(f_vals)

            en_flipped = 1 if f_vals == clause.unsat_index else 0

            delta += en_flipped - en
        return delta
