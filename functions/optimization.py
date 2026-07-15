"""Multi-objective optimisation of brain morphometric slices using NSGA-II/III.

Given a DataFrame of per-slice measurements (GI, sulci counts, depths, area,
cell density), finds Pareto-optimal slices that best satisfy the user's
selected objectives and constraints.

Key concepts:
    * **pymoo minimises** by default — objectives that should be *maximised*
      are sign-flipped (``-val``) before passing to the solver.
    * ``obj_to_column`` maps internal objective names (e.g. ``"perimeter_rate"``)
      to the DataFrame column names (e.g. ``"LGI"``).
    * Constraints are expressed as ``g(x) ≤ 0``: e.g.
      ``cell_density - max_cell_density ≤ 0``.
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

# Constraint key → the column it bounds.
CONSTRAINT_TO_COLUMN = {
    "max_cell_density": "CellDensity",
    "number_SulciCount": "SulciCount",
    "max_MaxDepth": "MaxDepth",
}


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
        self.constraints = constraints or {}
        self.objective_directions = objective_directions or {}

        # A constraint only applies when its column is present. Building the
        # active list once keeps n_constr in step with out["G"] — counting a
        # constraint here that _evaluate then skips makes pymoo fail.
        self.active_constraints = []
        for key, column in CONSTRAINT_TO_COLUMN.items():
            bound = self.constraints.get(key)
            if bound is None:
                continue
            if column not in self.df.columns:
                print(f"[Optimization] Ignoring constraint '{key}': "
                      f"column '{column}' not in data.")
                continue
            self.active_constraints.append((column, float(bound)))

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

        # g(x) ≤ 0 — each active constraint is an upper bound on its column.
        G = [(values(column) - bound).reshape(-1, 1)
             for column, bound in self.active_constraints]
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
        objectives: List of objective names (keys of ``obj_to_column``).
        objective_directions: Dict mapping objective → ``"maximize"``/``"minimize"``.
        constraints: Dict of upper-bound constraints, e.g.
            ``{"max_cell_density": 2500, "number_SulciCount": 2}``.
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
    required_cols = sorted({OBJ_TO_COLUMN.get(obj, obj) for obj in objectives}
                           | {col for key, col in CONSTRAINT_TO_COLUMN.items()
                              if (constraints or {}).get(key) is not None
                              and col in df1.columns})
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
    constraints = constraints or {}
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
            for key, value in constraints.items():
                f.write(f"  - {key}: {value}\n")
        else:
            f.write("Constraints: none\n")
    print(f"[Optimization] Saved optimization parameters: {params_txt_path}")

    # Perform optimization
    res = minimize(problem, algorithm, termination, seed=1, verbose=True, evaluator=Evaluator())

    print(res)
    # Results in a DataFrame for better readability
    results_with_indices = pd.DataFrame(
        {
            objectives[i]: (
                -res.F[:, i]
                if str(objective_directions.get(objectives[i], "maximize")).lower() == "maximize"
                else res.F[:, i]
            )
            for i in range(len(objectives))
        }
    )


    # Display of Pareto-optimal solutions
    mask = pd.Series([True] * len(df1))
    if "perimeter_rate" in objectives and "perimeter_rate" in results_with_indices.columns and "LGI" in df1.columns:
        mask &= df1["LGI"].isin(results_with_indices["perimeter_rate"])
    if "cell_density" in objectives and "cell_density" in results_with_indices.columns and "CellDensity" in df1.columns:
        mask &= df1["CellDensity"].isin(results_with_indices["cell_density"])
    if "max_min_d_value" in objectives and "max_min_d_value" in results_with_indices.columns and "MinDepth" in df1.columns:
        mask &= df1["MinDepth"].isin(results_with_indices["max_min_d_value"])
    if "min_min_d_value" in objectives and "min_min_d_value" in results_with_indices.columns and "MinDepth" in df1.columns:
        mask &= df1["MinDepth"].isin(results_with_indices["min_min_d_value"])
    if "mean_d_value" in objectives and "mean_d_value" in results_with_indices.columns and "MeanDepth" in df1.columns:
        mask &= df1["MeanDepth"].isin(results_with_indices["mean_d_value"])
    if "max_d_value" in objectives and "max_d_value" in results_with_indices.columns and "MaxDepth" in df1.columns:
        mask &= df1["MaxDepth"].isin(results_with_indices["max_d_value"]) 
    if "area" in objectives and "area" in results_with_indices.columns and "area" in df1.columns:
        mask &= df1["area"].isin(results_with_indices["area"])  

    
    filtered_df1 = df1[mask].reset_index(drop=True)
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
