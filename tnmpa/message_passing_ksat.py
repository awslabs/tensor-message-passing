import copy
from collections import defaultdict
from itertools import compress

import numpy as np

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
        self._check_clauses = copy.deepcopy(self.instance.clauses)
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

        # loop over all instances of the problem
        for kc, vc in self.instance.clauses.items():
            # add corresponding random, initial envs tensors
            self.envs_tensors[kc] = {
                v: self.rand_env(self.dim_envs, which="clause") for v in vc.pos
            }

        # loop over all variables of the problem
        for var in self.instance.variables:
            # add corresponding random, initial envs tensors
            self.envs_tensors[var] = {
                c: self.rand_env(self.dim_envs, which="variable")
                for c in self.instance.var_clause_map[f"V{var}"]
            }

    def check_solution(self, fixed_vars, vals):
        """
        Compute the number of contradictions for this (potentially partial) solution of the problem
        """
        sol = dict(zip(fixed_vars, vals))
        count = 0
        for c in self._check_clauses.values():
            try:
                # if the solutino is only partial, this will fail
                vals = tuple(sol[p] for p in c.pos)
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
            ranks = list(map(len, self.instance.var_clause_map.values()))
            # pick variable with largest order
            var = int(
                list(self.instance.var_clause_map.keys())[ranks.index(max(ranks))][1:]
            )
            # fix it to the value that satisfies most of its clauses
            vals_var = [
                self.instance.clauses[c](var)
                for c in self.instance.var_clause_map[f"V{var}"]
            ]
            return vals_var.count(False) > vals_var.count(True)

    def check_contradictions(self, len_one_clauses):
        # check there are no contradictions
        positions = []
        same_pos = {}
        # collect length one clauses at same position
        for k, c in enumerate(len_one_clauses):
            if c.pos[0] not in positions:
                positions.append(c.pos[0])
                same_pos[c.pos[0]] = [k]
            else:
                same_pos[c.pos[0]].append(k)
        # if two length one clauses have different values at the same position,
        # we reached a contradiciton. If they all have the same value,
        for pos, ixs in same_pos.items():
            if len(ixs) == 1:
                continue
            if len(set([len_one_clauses[k](pos) for k in ixs])) != 1:
                for k in ixs:
                    print(
                        len_one_clauses[k].pos,
                        len_one_clauses[k].vals,
                        len_one_clauses[k].label,
                    )
                raise ValueError("Contradiction reached")
        return len_one_clauses

    def remove_len1_clauses(self):
        """
        Iteratively remove all cluases of length one. If a clause of lenght one is left
        in the instance, the corresponding variable must take the value that satisfies the clause.
        """
        while True:
            # collect all the indices of len one clauses
            idx_l1_clauses = (
                np.array([len(c.pos) for c in self.instance.clauses.values()]) == 1
            )
            rm_clauses = list(compress(self.instance.clauses.values(), idx_l1_clauses))

            # if empty, exit
            if len(rm_clauses) == 0:
                break
            # else:
            #    print(f"rm len 1 clauses: {[c.label for c in rm_clauses]}")
            # check no contradiction is reached
            rm_clauses = self.check_contradictions(rm_clauses)

            # loop over all len one clauses
            for c in rm_clauses:
                # fix the variable to satisfy the clause
                v, val = c.pos[0], bool(not c(c.pos[0]))

                # fix variable and value to satisfy the clause
                self.fixed_vars.append(v)
                self.vals.append(val)

                # remove it from the variables
                self.instance.variables = self.instance.variables[
                    self.instance.variables != v
                ]

                # remove the variable from the instance, update neighboring clauses and variables
                clauses, (variables, vals) = self.instance.remove_var(v, val)

    def kick_envs(self, noise=None):
        if noise == None:
            return
        for cc, envs in self.envs_tensors.items():
            for vv in envs.keys():
                if isinstance(cc, str):
                    self.envs_tensors[cc][vv] += noise * self.rand_env(
                        self.dim_envs, which="clause"
                    )
                else:
                    self.envs_tensors[cc][vv] += noise * self.rand_env(
                        self.dim_envs, which="variable"
                    )

    def prune_envs(self, noise=None):
        """
        Remove the environments based on the current instance state.
        Optionally add noise to the remaining environments
        """
        to_delete = []
        for cc, envs in self.envs_tensors.items():
            if isinstance(cc, str):
                if cc not in self.instance.clauses.keys():
                    to_delete.append((cc,))
                else:
                    for vv in envs.keys():
                        if vv not in self.instance.variables:
                            to_delete.append((cc, vv))
                        elif noise != None:
                            self.envs_tensors[cc][vv] += noise * self.rand_env(
                                self.dim_envs, which="clause"
                            )
            else:
                if cc not in self.instance.variables:
                    to_delete.append((cc,))
                else:
                    for vv in envs.keys():
                        if vv not in self.instance.clauses.keys():
                            to_delete.append((cc, vv))
                        elif noise != None:
                            self.envs_tensors[cc][vv] += noise * self.rand_env(
                                self.dim_envs, which="variable"
                            )

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
        which_condition="any",
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
                len(self.instance.variables) > 0 and len(self.instance.clauses) > 0
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
                    len(self.instance.clauses) / len(self.instance.variables)
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
            update_clauses, (update_variables, update_vals) = self.instance.remove_var(
                var, val
            )

            # check if something went wrong
            count, _ = self.check_solution(self.fixed_vars, self.vals)
            if count > 0:
                print(f"count = {count} after remove_var")
                return

            # if there is any length-one clause, remove it and fix its variable
            # to satisfy it. Update instance and tensors accordingly
            # self.remove_len1_clauses()
            self.prune_envs(noise=env_noise)

            # Monitor progress
            if verbose:
                log = ""
                log += f"{len(self.instance.clauses)}\t{len(self.instance.variables)}"
                if len(self.instance.variables) > 0:
                    log += f"\t{len(self.instance.clauses)/len(self.instance.variables):.2f}"
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
        for v in self.instance.variables:
            v_rank = len(self.instance.var_clause_map[f"V{v}"])
            m = self.variable_marginal(v)
            entropy -= (1 - v_rank) * m.dot(
                np.log(m, out=np.zeros_like(m), where=(m != 0))
            )
        for c in self.instance.clauses.keys():
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
