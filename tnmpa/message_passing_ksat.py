import copy
from collections import defaultdict
from itertools import compress

import numpy as np

from tnmpa.base.factor_graph import Factor, Variable

from .mp_solvers import (
    belief_propagation,
    bp_decimation,
    dense_belief_propagation,
    dense_survey_propagation,
    sp_decimation,
    survey_propagation,
    two_norm_belief_propagation,
    two_norm_bp_decimation,
)


class MessagePassingKsatSolver:
    """
    A tensor-network based ksat solver.
    Solve a `KsatInstance` with a tensor-network implementation of belief propagation ("BP") and
    survey propagation ("SP").

    Attributes
    ----------
    fixed_vars: list[int]
        Variables fixed so far
    vals: list[bool]
        Values of the fixed variables
    instance: KsatInstance
        An instance of a KSAT problem
    envs_tensors: dict
        A dictionary that maps the order of each tensor representation of all the
        environments in this ksat instance to the corresponding data.
        The number of environments corresponds to the order
    """

    def __init__(self, instance):
        """
        Create an instance of a KSAT solver.

        Parameters
        ----------
        instance: KsatInstance
            The problem to be solved
        """
        # placeholders for the solution of the problem
        self.fixed_vars = []
        self.vals = []

        self.instance = instance

        # keep a full copy of the initial clauses to check the solution
        # as self.instance will be modified while solving
        self._check_clauses = copy.deepcopy(self.instance)
        self.dict_tensors()

    def dict_tensors(self):
        """
        Build the tensor representation of the problem depending on the solver.
        Construct three dictionaries, clauses_tensors and variables_tensors,
        that map clases label and variable index to the corresponding quimb Tensor.
        Additionally, create a dictionary that at the same key contains the environements
        connected to a given tensor.
        """

        # placeholder for the tensors
        self.envs_tensors = {}

        for node in self.instance.graph.nodes:
            neighbors = self.instance.graph.neighbors(node)
            if self.instance.graph.nodes[node]["bipartite"] == 0:
                self.envs_tensors[node] = {
                    ngbr: self.rand_env(self.dim_envs, which="variable")
                    for ngbr in neighbors
                }
            elif self.instance.graph.nodes[node]["bipartite"] == 1:
                self.envs_tensors[node] = {
                    ngbr: self.rand_env(self.dim_envs, which="clause")
                    for ngbr in neighbors
                }

    def check_solution(self, fixed_vars, vals):
        """
        Compute the number of contradictions for this (potentially partial) solution of the problem
        """
        sol = dict(zip(fixed_vars, vals))
        count = 0
        for c in self.instance.factors:
            try:
                # if the solutino is only partial, this will fail
                vals = tuple(sol[p] for p in c.variables)
                # if a clause is violated, the variables take the same values as in the unsat_index
                if vals == c.unsat_index:
                    count += 1
            except:
                continue
        return count, sol

    def var_value(self, bias):
        """
        Compute value of the variable based on the bias.
        If `bias=0.0` we can fix any variable to any value. We choose the
        a variable with the largest order.
        """
        if bias != 0.0:
            return False if bias > 0.0 else True
        else:
            # pick variable with largest degree
            var = max(
                self.instance.degree_variables, key=self.instance.degree_variables.get
            )
            # fix it to the value that satisfies most of its clauses
            vals_var = [
                self.instance.graph.nodes[n]["data"](var)
                for n in self.instance.graph.neighbors(var)
            ]
            return vals_var.count(False) > vals_var.count(True)

    def kick_envs(self, noise=None):
        if noise == None:
            return
        for n in self.instance.graph.nodes:
            for ngbr in self.instance.graph.neighbors(n):
                if isinstance(n, Variable):
                    self.envs_tensors[n][ngbr] += noise * self.rand_env(
                        self.dim_envs, which="variable"
                    )
                if isinstance(n, Factor):
                    self.envs_tensors[n][ngbr] += noise * self.rand_env(
                        self.dim_envs, which="clause"
                    )

    def prune_envs(self):
        """
        Remove the environments based on the current instance state.
        Optionally add noise to the remaining environments
        """
        to_delete = []

        for k1, envs in self.envs_tensors.items():
            if k1 not in self.instance.graph.nodes:
                to_delete.append((k1,))
            else:
                for k2 in envs.keys():
                    if k2 not in self.instance.graph.nodes:
                        to_delete.append((k1, k2))

        for d in to_delete:
            if len(d) == 1:
                del self.envs_tensors[d[0]]
            elif len(d) == 2:
                del self.envs_tensors[d[0]][d[1]]

    def solve(
        self,
        threshold_bias=None,
        threshold_alpha=None,
        verbose=True,
        env_noise=0.0,
        **tnbp_kwargs,
    ):
        # show progress
        if verbose:
            log = ""
            log += f"M"
            log += f"\tN"
            log += f"\talpha"
            log += f"\tcount"
            log += f"\tfxvar"
            log += f"\tbias"
            log += f"\t\tval"
            log += f"\titers"
            log += f"\tdist"
            log += f"\t\tMP time"
            log += f"\titer time"
            print(log)

        # need these just for the first call of converged()
        bias = 1.0
        count = 0

        def converged():
            """
            Check if the algorithm converged, or not.
            """
            # check if there are still variables to fix
            convergence_criteria = [
                len(self.instance.variables) > 0 and len(self.instance.factors) > 0
            ]

            # check if the current solution does not satisfy the instance
            convergence_criteria.append(count == 0)

            # Sometimes we don't want to run SP until we fixed all the variables
            # We might want to run it until the biases or alpha, the ratio between the number of clases and variables,
            # become samller than a certain threshold.
            if threshold_bias is not None:
                convergence_criteria.append(abs(bias) > threshold_bias)
            if threshold_alpha is not None:
                convergence_criteria.append(
                    len(self.instance.factors) / len(self.instance.variables)
                    > threshold_alpha
                )

            return all(convergence_criteria)

        # Actual loop. Fix a variable for each loop
        while converged():
            # Run Belief Propagation on this tensor network
            # with these initial enviroments
            status = self.MP(self.envs_tensors, self.instance, **tnbp_kwargs)
            if not status["bp_converged"]:
                print("MP did not converge")
                return status

            # compute the variable with the maximum absolute bias
            var, bias = self.decimation(self.instance, self.envs_tensors)
            if self.maybe_stop(bias, tnbp_kwargs["tol"]):
                print(f"Bias ({bias}) smaller than tol ({tnbp_kwargs['tol']})")
                return status

            # calculate the value of the variable
            val = self.var_value(bias)

            # Save fixed variable and values: the partial solution of the problem
            self.fixed_vars.append(var)
            self.vals.append(bool(val))

            # remove the variable from the instance, together with all the
            # clauses (and potentially other variables) that are influenced by that
            removed_clauses = self.instance.fix_variable(var, val)

            # check if something went wrong
            count, _ = self.check_solution(self.fixed_vars, self.vals)
            if count > 0:
                print(f"count = {count} after remove_var")
                return

            self.prune_envs()

            # Monitor progress
            if verbose:
                log = ""
                log += f"{len(self.instance.factors)}\t{len(self.instance.variables)}"
                if len(self.instance.variables) > 0:
                    log += f"\t{len(self.instance.factors)/len(self.instance.variables):.2f}"
                else:
                    log += f"\t0"
                log += f"\t{count}"
                log += f"\t{var}\t{bias:.2e}\t{val}"
                log += f"\t{status['iterations']}"
                log += f"\t{status['max_distance']:.2e}"
                log += f"\t{status['tot_time']:.2f}"
                log += f"\t{status['iter_time']:.3f}"
                print(log)
        status["success"] = True
        return status


def rand_env_sp(dim, which="variable"):
    """
    Randomize environments for survey propagation
    """
    env = np.random.rand(dim)
    if which == "variable":
        env[2:] = 0
    elif which == "clause":
        env[:2] = 0
    else:
        raise ValueError(f"{which} not supported")
    return env / env.sum()


def stop_sp(bias, tol):
    """
    Stopping criterium for survey propagation
    """
    return abs(bias) < tol


class SurveyPropagation(MessagePassingKsatSolver):
    def __init__(self, instance):
        self.decimation = sp_decimation
        self.rand_env = rand_env_sp
        self.dim_envs = 5
        self.maybe_stop = stop_sp
        self.MP = survey_propagation
        super().__init__(instance)


def rand_env_bp(dim, which=""):
    """
    Randomize environments for belief propagation
    """
    env = np.random.rand(dim)
    return env / env.sum()


def stop_bp(bias, tol):
    """
    Stopping criterium for balief propagation.
    BP does not need to stop for small biases
    """
    return False


class BeliefPropagation(MessagePassingKsatSolver):
    def __init__(self, instance):
        self.decimation = bp_decimation
        self.rand_env = rand_env_bp
        self.dim_envs = 2
        self.MP = belief_propagation
        self.maybe_stop = stop_bp
        super().__init__(instance)

    def variable_marginal(self, variable):
        """
        Compute the single variable marginal
        """
        m1 = 1.0
        m0 = 1.0
        for e in self.envs_tensors[variable].values():
            m1 *= e[1]
            m0 *= e[0]
        return np.array([m0 / (m1 + m0), m1 / (m1 + m0)])

    def clause_marginal(self, clause_label, return_array=True):
        """
        Compute the multivariable marginal for this clause

        Paramters
        ---------
        clause_label: str
            Clause label
        return_array: bool
            Whether to return a dense array, or a factorized tensor network

        Returns
        -------
        marginal: Union[quimb.tensor.TensorNetwork, array]
            The corresponding *non-normalized* multivariable marginal
        """

        import quimb.tensor as qtn

        from .tensor_factories import bp_clause_tensor, bp_variable_tensor

        clause = self.instance.clauses[clause_label]
        tc = bp_clause_tensor(clause)

        tvs = []
        tve = []
        for v in clause.pos:
            clause_var = [
                self.instance.clauses[cl]
                for cl in self.instance.var_clause_map[f"V{v}"]
            ]
            tvs.append(bp_variable_tensor(v, clause_var, open_leg=True))
            for c in clause_var:
                if c.label != clause_label:
                    tve.append(
                        qtn.Tensor(
                            data=self.envs_tensors[v][c.label],
                            inds=(f"p{v}_{c.label}",),
                            tags=("ENV",),
                        )
                    )
        tn = qtn.TensorNetwork(tvs + tve + [tc])
        if return_array:
            return tn.contract().data.flatten()
        else:
            return tn

    @property
    def entropy(self):
        entropy = 0
        degrees = self.instance.degree_variables
        for v in self.instance.variables:
            v_rank = degrees[v]
            m = self.variable_marginal(v)
            entropy -= (1 - v_rank) * m.dot(
                np.log(m, out=np.zeros_like(m), where=(m != 0))
            )
        for c in self.instance.factors:
            m = self.clause_marginal(c)
            m /= sum(m)
            entropy -= m.dot(np.log(m, out=np.zeros_like(m), where=(m != 0)))
        return entropy


class TensorBeliefPropagation(BeliefPropagation):
    """
    Dense version of BeliefPropagation
    """

    def __init__(self, instance):
        super().__init__(instance)
        self.MP = dense_belief_propagation


class TensorSurveyPropagation(SurveyPropagation):
    """
    Dense version of SurveyPropagation
    """

    def __init__(self, instance):
        super().__init__(instance)
        self.MP = dense_survey_propagation


def rand_env_two_norm_bp(dim, which=""):
    """
    Randomize environments for belief propagation
    """
    env = np.random.rand(dim)
    env[1] = env[2]
    return env / np.linalg.norm(env)


class TwoNormBeliefPropagation(MessagePassingKsatSolver):
    def __init__(self, instance):
        self.decimation = two_norm_bp_decimation
        self.rand_env = rand_env_two_norm_bp
        self.dim_envs = 4
        self.MP = two_norm_belief_propagation
        self.maybe_stop = stop_bp
        super().__init__(instance)

    def variable_marginal(self, variable):
        """
        Compute the single variable marginal
        """
        m1 = 1.0
        m0 = 1.0
        for e in self.envs_tensors[variable].values():
            m1 *= e[3]
            m0 *= e[0]
        return np.array([m0 / (m1 + m0), m1 / (m1 + m0)])
