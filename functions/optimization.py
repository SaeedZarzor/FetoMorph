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

class MyProblem(Problem):
    """pymoo Problem wrapper that maps DataFrame rows to objective values.

    Each decision variable is a continuous index into the DataFrame; the
    ``_evaluate`` method floors it to an integer row index and looks up
    the requested metric columns.
    """

    def __init__(self, data, objectives = None, constraints = None, objective_directions=None):
        self.df = data.copy()
        self.objectives = objectives or []
        self.constraints = constraints or {}
        self.objective_directions = objective_directions or {}

        n_constr = 0
        if self.constraints.get("max_cell_density") is not None:
            n_constr += 1 
        if self.constraints.get("number_SulciCount") is not None:
            n_constr += 1 
        if self.constraints.get("max_MaxDepth") is not None:
            n_constr += 1 


        super().__init__(
            n_var=1,
            n_obj=len(self.objectives),
            n_constr=n_constr,
            xl=np.array([0]),
            xu=np.array([len(data) - 1]),
            type_var=np.double,
        )



    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objectives and constraints for a population of solutions."""
        # Convert continuous decision variables to integer DataFrame row indices.
        indices = np.floor(x).astype(int).flatten()
        indices = np.clip(indices, 0, len(self.df) - 1)

        perimeter_rate = self.df["LGI"].iloc[indices].values
        sulci_count = self.df["SulciCount"].iloc[indices].values
        max_d_value = self.df["MaxDepth"].iloc[indices].values
        min_d_value = self.df["MinDepth"].iloc[indices].values
        mean_d_value = self.df["MeanDepth"].iloc[indices].values
        area = self.df["area"].iloc[indices].values
        if "CellDensity" in self.df.columns:
            cell_density = self.df["CellDensity"].iloc[indices].values

        F = []
        objective_values = {
            "perimeter_rate": perimeter_rate,
            "max_d_value": max_d_value,
            "max_min_d_value": min_d_value,
            "min_min_d_value": min_d_value,
            "mean_d_value": mean_d_value,
            "area": area,
        }
        if "CellDensity" in self.df.columns:
            objective_values["cell_density"] = cell_density

        for obj in self.objectives:
            if obj not in objective_values:
                continue
            direction = str(self.objective_directions.get(obj, "maximize")).lower()
            val = objective_values[obj]
            # pymoo minimises by default — flip sign for maximisation objectives.
            F.append(-val if direction == "maximize" else val)

        out["F"] = np.column_stack(F)

        G = [] 
        if "CellDensity" in self.df.columns:
            max_cd= self.constraints.get("max_cell_density")
            if max_cd is not None:
                G.append((cell_density - max_cd).reshape(-1, 1))

        number_su= self.constraints.get("number_SulciCount")
        if number_su is not None:
            G.append((sulci_count - number_su).reshape(-1, 1))
        
        max_md= self.constraints.get("max_MaxDepth")
        if max_md is not None:
            G.append((max_d_value - max_md).reshape(-1, 1))

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

    if "cell_density" in objectives and "CellDensity" not in df1.columns:
            QMessageBox.critical(parent, "Optimization Failed",
                                "Objective 'Cell Density' selected but 'CellDensity' column not found in data!")
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
    
    # Map internal objective names → DataFrame column names.
    # This indirection allows the UI to use descriptive names while the
    # DataFrame columns match the measurement pipeline output.
    obj_to_column = {
        "perimeter_rate": "LGI",
        "cell_density": "CellDensity",
        "max_min_d_value": "MinDepth",
        "min_min_d_value": "MinDepth",
        "mean_d_value": "MeanDepth",
        "max_d_value": "MaxDepth",
        "area": "area"
    }
    
    for i, j in combinations(range(len(objectives)), 2):
        x_obj = objectives[i]
        y_obj = objectives[j]
        
        # Get corresponding column names
        x_col = obj_to_column.get(x_obj, x_obj)
        y_col = obj_to_column.get(y_obj, y_obj)
        
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
