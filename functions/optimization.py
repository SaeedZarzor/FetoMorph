"""Multi-objective optimisation of brain morphometric slices using NSGA-II/III.

Given a DataFrame of per-slice measurements (GI, sulci counts, depths, area,
cell density), finds Pareto-optimal slices that best satisfy the user's
selected objectives and constraints.

Objectives and constraints are chosen per run from whatever columns the
selected Excel files contain, so a metric added to the exporter becomes
optimisable without touching this module.

Key concepts:
    * **pymoo minimises** by default — objectives that should be *maximised*
      are sign-flipped (``-val``) before passing to the solver.
    * An objective is normally just a DataFrame column name. ``OBJ_TO_COLUMN``
      only maps the legacy internal names (``"perimeter_rate"`` → ``"LGI"``)
      that predate column-based selection; unknown names pass through
      unchanged.
    * Constraints are ``{"column", "op", "value"}`` rows, expressed to pymoo
      as ``g(x) ≤ 0``: ``CellDensity ≤ 2500`` becomes
      ``CellDensity - 2500 ≤ 0``, and ``SulciCount ≥ 2`` its mirror image
      ``2 - SulciCount ≤ 0``.
"""

from __future__ import annotations

from deps import *
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.core.evaluator import Evaluator
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
from itertools import combinations

# Map internal objective names → DataFrame column names. This indirection
# lets the UI use descriptive names while the DataFrame columns match the
# measurement pipeline output. Single source of truth — the dispatcher and
# the scatter plots import this rather than keeping their own copies.
OBJ_TO_COLUMN = {
    "perimeter_rate": "LGI",
    "cell_density": "CellDensity",
    "min_d_value": "MinDepth",
    "max_min_d_value": "MinDepth",
    "min_min_d_value": "MinDepth",
    "mean_d_value": "MeanDepth",
    "max_d_value": "MaxDepth",
    "area": "area",
}

# Legacy constraint key → the column it bounds. Only used to read the old
# ``{"max_cell_density": 2500}`` dict form; the dialog now names columns
# directly and states the comparison explicitly.
CONSTRAINT_TO_COLUMN = {
    "max_cell_density": "CellDensity",
    "number_SulciCount": "SulciCount",
    "max_MaxDepth": "MaxDepth",
}


def normalize_constraints(constraints) -> list[dict]:
    """Coerce any accepted constraint format into ``[{column, op, value}]``.

    Two formats are accepted:

    * The current one, a list of ``{"column": "SulciCount", "op": ">=",
      "value": 1}`` dicts, where any column can be bounded in either
      direction.
    * The legacy ``{"max_cell_density": 2500, "number_SulciCount": 2}`` dict,
      whose keys are looked up in :data:`CONSTRAINT_TO_COLUMN` and which is
      always an upper bound — that was the only thing the old solver could
      express.

    Unknown keys and malformed rows are dropped with a message rather than
    raising, so a stale saved configuration cannot break a run.
    """
    if not constraints:
        return []

    normalized: list[dict] = []

    if isinstance(constraints, dict):
        for key, value in constraints.items():
            if value is None:
                continue
            column = CONSTRAINT_TO_COLUMN.get(key, key)
            try:
                normalized.append({"column": column, "op": "<=", "value": float(value)})
            except (TypeError, ValueError):
                print(f"[Optimization] Ignoring constraint '{key}': "
                      f"non-numeric bound {value!r}.")
        return normalized

    for row in constraints:
        try:
            column = str(row["column"])
            op = str(row.get("op", "<=")).strip()
            value = float(row["value"])
        except (TypeError, ValueError, KeyError):
            print(f"[Optimization] Ignoring malformed constraint: {row!r}")
            continue
        if op not in ("<=", ">="):
            print(f"[Optimization] Ignoring constraint on '{column}': "
                  f"unsupported operator {op!r}.")
            continue
        normalized.append({"column": column, "op": op, "value": value})
    return normalized


class MyProblem(Problem):
    """pymoo Problem wrapper that maps DataFrame rows to objective values.

    Each decision variable is a continuous index into the DataFrame; the
    ``_evaluate`` method floors it to an integer row index and looks up
    the requested metric columns.
    """

    def __init__(self, data, objectives = None, constraints = None, objective_directions=None):
        self.df = data.copy()
        # Keep only objectives backed by a column, so n_obj always matches
        # the width of out["F"]. optimization() rejects missing columns up
        # front, so in practice nothing is dropped here.
        self.objectives = [o for o in (objectives or [])
                           if OBJ_TO_COLUMN.get(o, o) in self.df.columns]
        self.constraints = normalize_constraints(constraints)
        self.objective_directions = objective_directions or {}

        # A constraint only applies when its column is present. Building the
        # active list once keeps n_constr in step with out["G"] — counting a
        # constraint here that _evaluate then skips makes pymoo fail.
        self.active_constraints = []
        for row in self.constraints:
            column = row["column"]
            if column not in self.df.columns:
                print(f"[Optimization] Ignoring constraint on '{column}': "
                      f"column not in data.")
                continue
            self.active_constraints.append((column, row["op"], row["value"]))

        super().__init__(
            n_var=1,
            n_obj=len(self.objectives),
            n_constr=len(self.active_constraints),
            xl=np.array([0]),
            xu=np.array([len(data) - 1]),
            type_var=np.double,
        )



    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objectives and constraints for a population of solutions."""
        # Convert continuous decision variables to integer DataFrame row indices.
        indices = np.floor(x).astype(int).flatten()
        indices = np.clip(indices, 0, len(self.df) - 1)

        def values(column):
            """Metric column sampled at the population's row indices."""
            return self.df[column].to_numpy()[indices]

        # Only the selected objectives are read — pulling every metric up
        # front breaks on sheets that legitimately omit one (the exporter
        # drops all-empty columns).
        F = []
        for obj in self.objectives:
            column = OBJ_TO_COLUMN.get(obj, obj)
            if column not in self.df.columns:
                continue
            direction = str(self.objective_directions.get(obj, "maximize")).lower()
            val = values(column)
            # pymoo minimises by default — flip sign for maximisation objectives.
            F.append(-val if direction == "maximize" else val)

        out["F"] = np.column_stack(F)

        # pymoo treats a constraint as satisfied when g(x) ≤ 0, so an upper
        # bound is (value - bound) and a lower bound is its mirror image.
        G = []
        for column, op, bound in self.active_constraints:
            val = values(column)
            G.append(((val - bound) if op == "<=" else (bound - val)).reshape(-1, 1))
        if G:
            out["G"] = np.column_stack(G)

def optimization(
    parent,
    df1: pd.DataFrame,
    out_dir: str,
    objectives=None,
    objective_directions=None,
    constraints=None,
    algorithms="NSGA-III",
    n_gen: int = 200,
):
    """Run NSGA-II or NSGA-III multi-objective optimisation on measurement data.

    Args:
        parent: Qt parent widget for message boxes.
        df1: DataFrame with one row per slice and metric columns.
        out_dir: Directory for output Excel and scatter plots.
        objectives: Objective names. Normally plain ``df1`` column names
            (``"LGI"``, ``"Compactness"``); the legacy keys of
            :data:`OBJ_TO_COLUMN` still resolve.
        objective_directions: Dict mapping objective → ``"maximize"``/``"minimize"``.
        constraints: Constraints in either format accepted by
            :func:`normalize_constraints`, e.g.
            ``[{"column": "SulciCount", "op": ">=", "value": 2}]``.
        algorithms: ``"NSGA-II"`` or ``"NSGA-III"``.
        n_gen: Number of generations (termination criterion).

    Returns:
        Tuple of ``(pareto_df, scatter_pngs, n_optimal)``.
    """
    if df1 is None or df1.empty:
        return None, [], 0

    #    objectives = ["perimeter_rate", "max_d_value", "cell_density", "max_min_d_value", "min_min_d_value", "mean_d_value", "area"]  
    #    constraints = {"max_cell_density": 2500, "number_SulciCount": 2, "max_MaxDepth": 2.4}


    if not objectives:
        QMessageBox.critical(parent, "Optimization Failed",
                            "No objectives selected. Please select at least one objective before proceeding!")
        return None, [], 0

    # Every selected objective needs its column. Report all missing ones at
    # once rather than failing later on the first KeyError.
    missing = sorted({OBJ_TO_COLUMN.get(obj, obj) for obj in objectives}
                     - set(df1.columns))
    if missing:
        # Show what the file DID provide — without it there is no way to tell
        # whether the metric is absent, named differently, or simply empty.
        found = [str(c) for c in df1.columns if not str(c).startswith("__")]
        sources = []
        if "__source_excel_name" in df1.columns:
            sources = sorted({str(s) for s in df1["__source_excel_name"]})
        print(f"[Optimization] Missing column(s): {', '.join(missing)}")
        print(f"[Optimization] Columns found in the data: {found}")
        print(f"[Optimization] Source file(s): {sources}")
        QMessageBox.critical(
            parent, "Optimization Failed",
            "The selected Excel data has no column for: "
            f"{', '.join(missing)}.\n\n"
            f"Columns found in the file: {', '.join(found) or '(none)'}\n\n"
            f"File(s) read: {', '.join(sources) or '(unknown)'}\n\n"
            "Check that the selected objectives match the measurements in "
            "the file.")
        return None, [], 0

    # Coerce the columns in play to numbers and drop rows that have no value
    # for one of them — a NaN would silently poison the Pareto front.
    constraints = normalize_constraints(constraints)
    required_cols = sorted({OBJ_TO_COLUMN.get(obj, obj) for obj in objectives}
                           | {row["column"] for row in constraints
                              if row["column"] in df1.columns})
    df1 = df1.copy()
    for col in required_cols:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")
    n_before = len(df1)
    df1 = df1.dropna(subset=required_cols).reset_index(drop=True)
    n_dropped = n_before - len(df1)
    if n_dropped:
        print(f"[Optimization] Dropped {n_dropped} row(s) with missing values "
              f"in: {', '.join(required_cols)}")
    if df1.empty:
        QMessageBox.critical(
            parent, "Optimization Failed",
            "No rows have values for all of the selected objectives: "
            f"{', '.join(required_cols)}.")
        return None, [], 0

    objective_directions = objective_directions or {obj: "maximize" for obj in objectives}
    problem = MyProblem(
        df1,
        objectives=objectives,
        constraints=constraints,
        objective_directions=objective_directions,
    )

    n_rows = len(df1)

    # scale with data size: 20% of rows, but keep sane bounds
    adaptive_pop_size = max(40, min(300, int(np.ceil(0.2 * n_rows))))
    adaptive_pop_size = min(adaptive_pop_size, n_rows)

    if algorithms == "NSGA-II":
            algorithm = NSGA2(
                pop_size=adaptive_pop_size,
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(prob=0.2, eta=20),
                eliminate_duplicates=True,
            )

    elif algorithms == "NSGA-III":

        ref_dirs = get_reference_directions("das-dennis", len(objectives), n_partitions=12)
        pop_size = max(adaptive_pop_size, len(ref_dirs))

        # Define NSGA-III algorithm with Das-Dennis reference directions.
        algorithm = NSGA3(
            pop_size=pop_size,
            ref_dirs=ref_dirs,
            crossover=SBX( prob=0.9, eta=15),  
            mutation=PM(prob=0.2, eta=20),  
            eliminate_duplicates=True,
        )

    adaptive_n_gen = max(50, min(200, int(np.ceil(0.5 * n_rows))))
    n_gen = int(n_gen) if n_gen is not None else adaptive_n_gen
    n_gen = max(1, n_gen)
    termination = get_termination("n_gen", n_gen)

    # Save optimization configuration for reproducibility.
    params_txt_path = os.path.join(out_dir, "optimization_parameters.txt")
    with open(params_txt_path, "w", encoding="utf-8") as f:
        f.write("Optimization Parameters\n")
        f.write("=======================\n")
        f.write(f"Algorithm: {algorithms}\n")
        f.write(f"Rows in input data: {n_rows}\n")
        f.write(f"Adaptive population size: {adaptive_pop_size}\n")
        if algorithms == "NSGA-III":
            f.write(f"Reference directions count: {len(ref_dirs)}\n")
            f.write(f"Final population size: {pop_size}\n")
        f.write(f"Termination criterion (n_gen): {n_gen}\n")
        f.write(f"Objectives ({len(objectives)}): {', '.join(objectives)}\n")
        f.write("Objective directions:\n")
        for obj in objectives:
            f.write(f"  - {obj}: {objective_directions.get(obj, 'maximize')}\n")
        if constraints:
            f.write("Constraints:\n")
            for row in constraints:
                f.write(f"  - {row['column']} {row['op']} {row['value']}\n")
        else:
            f.write("Constraints: none\n")
    print(f"[Optimization] Saved optimization parameters: {params_txt_path}")

    # Perform optimization
    res = minimize(problem, algorithm, termination, seed=1, verbose=True, evaluator=Evaluator())

    print(res)

    # Every constraint can be violated at once, in which case pymoo returns no
    # solution at all. Reported here, because reading res.X below would
    # otherwise fail with an opaque TypeError.
    if res.X is None or res.F is None:
        summary = "\n".join(f"  - {row['column']} {row['op']} {row['value']}"
                            for row in constraints)
        print("[Optimization] No feasible solution — constraints unsatisfiable.")
        QMessageBox.critical(
            parent, "Optimization Failed",
            "No slice satisfies all of the constraints:\n\n"
            f"{summary}\n\n"
            "Relax or disable a constraint and try again.")
        return None, [], 0

    # Each decision variable *is* a row index, so the Pareto rows are read back
    # directly. Matching on objective values instead (df[col].isin(...)) would
    # also pull in any other row that happens to share a value, and could not
    # be written for columns not known ahead of time.
    indices = np.floor(np.asarray(res.X, dtype=float)).astype(int).flatten()
    indices = np.unique(np.clip(indices, 0, len(df1) - 1))
    filtered_df1 = df1.iloc[indices].reset_index(drop=True)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    saved_pngs = []

    # Save scatter plots for every pair of selected objectives.
    os.makedirs(out_dir, exist_ok=True)
    
    for i, j in combinations(range(len(objectives)), 2):
        x_obj = objectives[i]
        y_obj = objectives[j]
        
        # Get corresponding column names
        x_col = OBJ_TO_COLUMN.get(x_obj, x_obj)
        y_col = OBJ_TO_COLUMN.get(y_obj, y_obj)
        
        # Get values from filtered_df1 using the actual column names
        x_vals = filtered_df1[x_col]
        y_vals = filtered_df1[y_col]

        plt.figure(figsize=(8, 6))
        plt.scatter(x_vals, y_vals, alpha=0.75, edgecolor="k", s=28)
        labels = (
            filtered_df1["File"].astype(str).tolist()
            if "File" in filtered_df1.columns
            else [f"Point {k}" for k in range(len(filtered_df1))]
        )
        for x, y, label in zip(x_vals, y_vals, labels):
            plt.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.75,
            )

        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Pareto Scatter: {x_col} vs {y_col}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        img_name = f"scatter_{x_col}_vs_{y_col}.png"
        img_path = os.path.join(out_dir, img_name)
        plt.savefig(img_path, dpi=200)
        saved_pngs.append(img_path)
        plt.close()
        print(f"[Optimization] Saved scatter plot: {img_path}")


    xlsx_path = os.path.join(out_dir, "Pareto-optimal solutions.xlsx")
    filtered_df1.to_excel(xlsx_path, index=False)
    print("[Optimization]The optimal solutions saved to 'Pareto-optimal solutions.xlsx'")

    orignal_file = os.path.join(out_dir, "original_data.xlsx")
    df1.to_excel(orignal_file, index=False)
    n_optimal_results = len(filtered_df1)
    return filtered_df1, saved_pngs, n_optimal_results
