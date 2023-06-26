import random

import numpy as np


class Clause:
    """
    A represetation of a clause in a KSAT instance.

    Attributes:
    ----------
    pos: tuple[int]
        Variables in this clause
    vals: tuple[int]
        Values of each variable in this clause
    label: str
        label of the clause
    unsat_index: tuple[bool]
        the single configuration that does not satisfy the clause
    """

    def __init__(self, pos, vals, label, tags=()):
        """
        Create an instance of this clause

        Parameters
        ----------
        pos: tuple[int]
            Variables in this clause
        vals: tuple[int]
            Values of each variable in this clause
        label: str
            label of the clause
        """
        if not (len(pos) == len(vals)):
            raise ValueError("len(pos) and len(vals) must match")
        self.pos = tuple(pos)
        self.vals = tuple(vals)
        self.label = label
        self.unsat_index = tuple(1 if self.__call__(p) else 0 for p in self.pos)

        if tags is not None:
            self.tags = tags
        else:
            self.tags = ()

    def __call__(self, var):
        # return the value of var in this clause
        return self.vals[self.pos.index(var)]

    def __eq__(self, other):
        # compare two clauses
        # two clasues are the same only if
        # pos, vals and label are all the same
        return (
            (self.pos == other.pos)
            and (self.vals == other.vals)
            and (self.label == other.label)
        )

    def __repr__(self):
        return f"Clause(label: {self.label}, pos: {self.pos}, vals: {self.vals})"

    def remove_var(self, var):
        # remove a variable from this clause
        # and update the other attributes accordingly

        pos = list(self.pos)
        vals = list(self.vals)
        unsat_index = list(self.unsat_index)

        idx_var = pos.index(var)

        del pos[idx_var]
        del vals[idx_var]
        del unsat_index[idx_var]

        self.pos = tuple(pos)
        self.vals = tuple(vals)
        self.unsat_index = tuple(unsat_index)


class KsatInstance:
    """
    A class that represents a ksat instance

    Attributes:
    -----------
    clauses: list[clause]
        A list that define the KSAT problem
    variables: numpy.array
        Variables of the problem
    var_clause_map: dict[tuple[clause,]]
        A table that maps each variable to the clauses it is attached to

    Methods
    -------
    create_link_map()
        Create var_clause_map
    generate_random_instance(num_vars, num_clauses, K, max_num_tries=100)
        Generate a random instance of a K-SAT problem with `num_vars` variables,
        `num_clauses` clauses and order `K`.
    remove_var(var, val)
        Remove/fix a variable from this ksat instance
    tn()
        Return a TensorNetwork representation of this ksat instance
    """

    def __init__(self, num_vars, num_clauses, K, seed=None):
        """
        Create a random instance of a K-SAT problem.

        Parameters
        ----------
        num_vars: int
             Number of variables
        num_clauses: int
             The number of clauses
        K: int
             K of the K-SAT problem
        """
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.K = K
        self.seed = seed
        self.generate_random_instance(num_vars, num_clauses, K, seed=seed)
        self.create_link_map()

    def __eq__(self, other):
        compare_clauses = [
            c == o for c, o in zip(self.clauses.values(), other.clauses.values())
        ]
        return all(compare_clauses)

    def create_link_map(self):
        # Create/update var_clause_map
        self.var_clause_map = {}
        for v in self.variables:
            clss = []
            for kc, vc in self.clauses.items():
                if v in vc.pos:
                    clss.append(kc)
            self.var_clause_map[f"V{v}"] = tuple(clss)

    def generate_random_instance(
        self, num_vars, num_clauses, K, max_num_tries=100, seed=None
    ):
        """
        Genrate a random K-SAT instance with `num_vars` variables,
        `num_clauses` clauses, and order `K`.

        Prameters
        ---------
        num_vars: int
             Number of variables
        num_clauses: int
             The number of clauses
        K: int
             K of the K-SAT problem
        max_num_tries: Optional[int]
             Maximum nummber of attempts to find a new clause (for very dense
             problems).
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # initialize the positions in each clause
        positions = np.array(
            [np.sort(random.sample(range(num_vars), K)) for n in range(num_clauses)]
        )
        # extract the label of the varialbes
        self.variables = np.unique(positions.flatten())
        clauses = []
        for a, p in enumerate(positions):
            try_again = True
            count = 0
            while try_again:
                # generate random boolean values for each clause
                v = np.random.randint(2, size=K, dtype=bool)
                # create the clause
                c = Clause(p, v, f"c{a}", tags=(f"L{a}",))
                # make sure all the clasues are different (not sure this is necessary)
                if c not in clauses:
                    clauses.append(c)
                    try_again = False

                count += 1
                if count > max_num_tries:
                    raise ValueError(
                        f"Can't find {num_clauses} different clauses for k-SAT"
                        f"instance with num_vars = {num_vars}, and K = {K}"
                    )
        self.clauses = dict(zip([c.label for c in clauses], clauses))

    def remove_var(self, var, val):
        """
        Remove/fix a variable form this ksat instance. Once we fix a variable,
        we need to update all the neighboring clauses and variables.
        For example, if a variable is fixed to a value that does satisfy one of
        its clauses, then we can remove that claues since it will be satisfied
        by any value taken by the remaining variables.

        Parameters
        ----------
        var: int
             Which variable to fix
        val: bool
             Which value to the variable takes

        Returns
        -------
        update_clauses: list[clauses]
             The clauses that need to be updated after fixing this variable
        update_variables, update_vals: tuple[list[clauses], list[bool]]
             The variables and the correponing values that need to be updated
             after fixinv this varaible
        """

        # remove var from the variables
        self.variables = self.variables[self.variables != var]

        update_variables = []
        update_vals = []
        update_clauses = []

        # collect all the clauses where var appears
        clauses = self.var_clause_map[f"V{var}"]

        for c_label in clauses:
            # check the value var takes for this clause
            cvar = bool(self.clauses[c_label](var))
            # remove var from this clause
            self.clauses[c_label].remove_var(var)

            if (cvar != bool(val)) or (len(self.clauses[c_label].pos) == 0):
                # record all the variables in the removed clause:
                #     they need to be updated
                for v in self.clauses[c_label].pos:
                    if v not in update_variables:
                        update_variables.append(v)
                        update_vals.append(not self.clauses[c_label](v))
                # if var is fixed to a value that satisfies the clause,
                #     remove the clause
                del self.clauses[c_label]
            else:
                # if var is fixed to a value that doesn't satisfy the clause
                # keep the clause and update it
                update_clauses.append(c_label)

        # update the variable to clause dictionary
        self.create_link_map()
        return update_clauses, (update_variables, update_vals)

    def tn(self, which="BP", factor_threshold=8):
        """
        Construct a tensor network representation of this ksat instance

        Parameters
        ----------
        which: str
            Either "BP" for belief propagation or "SP" for survey propagation

        Returns
        -------
        TensorNetwork
        """
        import quimb.tensor as qtn

        import tnmpa.tensor_factories as tfac

        tn = qtn.TensorNetwork([])

        if which == "BP":
            self._tensor_factories = {
                "clause": tfac.bp_clause_tensor,
                "variable": tfac.bp_variable_tensor,
            }
        elif which == "SP":
            self._tensor_factories = {
                "clause": tfac.sp_clause_tensor,
                "variable": tfac.sp_variable_tensor,
            }

        for c in self.clauses.values():
            # add one tensor for each clause
            tn_c = self._tensor_factories["clause"](c)
            tn.add(tn_c, virtual=True)

        for var in self.variables:
            # add one tensor for each variable
            clauses = [self.clauses[cl] for cl in self.var_clause_map[f"V{var}"]]
            tn_v = self._tensor_factories["variable"](
                var,
                clauses,
                factor_threshold=factor_threshold,
            )
            tn.add(tn_v, virtual=True, check_collisions=False)

        return tn

    @property
    def num_missing(self):
        """The number variables that appear in no clauses."""
        all_vars = set(range(self.num_vars))
        for clause in self.clauses.values():
            for v in clause.pos:
                all_vars.discard(v)
        return len(all_vars)

    def to_cnf_tuples(self, fix=None):
        """Build the CNF (Conjunctive Normal Form) of this ksat instance as a
        list of tuples. Each tuple represents a clause and each element of the
        tuple is a variable or its negation. For example, the tuple (1, -2, 3)
        represents the clause (x1 or not x2 or x3). Note variables are 1
        indexed!

        """
        cnf = []
        for clause in self.clauses.values():
            cnf.append(
                tuple((-1) ** val * (p + 1) for val, p in zip(clause.vals, clause.pos))
            )

        if fix is not None:
            for var, val in fix.items():
                cnf.append(((1 if val else -1) * (var + 1),))

        return cnf

    def to_cnf_file(self, filename, fix=None):
        """Write a conjunctive normal form (CNF) file for this ksat instance."""
        # TODO: account for other methods changing the number of variables

        clauses = self.to_cnf_tuples(fix=fix)
        lines = [
            f"c random {self.K}-SAT instance, seed = {self.seed}",
            f"p cnf {self.num_vars} {len(clauses)}",
        ]
        for clause in clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        lines = "\n".join(lines)
        with open(filename, "w") as f:
            f.write(lines)

    def count_ganak(self, ganak_path="ganak", fix=None):
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".cnf") as tmp:
            self.to_cnf_file(tmp.name, fix=fix)
            sr = subprocess.run([ganak_path, tmp.name], capture_output=True)
            solution_lines = [
                line
                for line in sr.stdout.decode().split("\n")
                if line and line[0] == "s"
            ]
            return int(solution_lines[1].split()[2])

    def compute_marginals_with_ganak(self, ganak_path="ganak", Z=None):
        margs = np.tile(0.5, (self.num_vars, 2))

        Z00 = self.count_ganak(ganak_path=ganak_path, fix={0: 0})

        if Z is None:
            Z01 = self.count_ganak(ganak_path=ganak_path, fix={0: 1})
            Z = Z00 + Z01
        else:
            Z01 = Z - Z00

        if Z == 0:
            # unsat
            return margs

        else:
            margs[0, 0] = Z00 / Z
            margs[0, 1] = Z01 / Z

        for i in range(1, self.num_vars):
            Zi0 = self.count_ganak(ganak_path=ganak_path, fix={i: 0})
            Zi1 = Z - Zi0
            margs[i, 0] = Zi0 / Z
            margs[i, 1] = Zi1 / Z

        return margs

    def htn(self, **kwargs):
        """This builds a hyper tensor network with both decomposed clauses and
        variables. Note variables are 1 indexed!
        """
        import quimb.tensor as qtn

        return qtn.HTN_from_clauses(self.to_cnf_tuples(), **kwargs)
