"""Multi-objective optimisation configuration dialog.

The objectives and constraints offered are built from the columns of the
Excel files the user selected, so any metric the measurement pipeline
exports can be optimised without changing this module. Known columns get a
descriptive label and a hover explanation; unknown ones fall back to the raw
column name.
"""

from deps import *


class OptimizationOptionsDialog(QtWidgets.QDialog):
    """Dialog for configuring multi-objective optimisation parameters.

    The user picks which of the file's columns to optimise (and in which
    direction), adds constraints bounding any column from either side,
    chooses a solver algorithm (NSGA-II / NSGA-III), and sets the number of
    generations.
    """

    OBJECTIVE_DIRECTIONS = ["maximize", "minimize"]

    DIRECTION_LABELS = {
        "maximize": "Maximize",
        "minimize": "Minimize",
    }

    # A Pareto front needs something to trade off against, so a single
    # objective is rejected — that is a plain sort, not an optimisation.
    MIN_OBJECTIVES = 2

    # Friendly labels for the columns FetoMorph exports. Anything not listed
    # is shown under its own column name.
    column_to_name = {
        "LGI": "Local gyrification index",
        "CellDensity": "The max cortical cell density",
        "MinDepth": "The minimum sulci depth",
        "MeanDepth": "The mean sulci depth",
        "MaxDepth": "The maximum sulci depth",
        "area": "The area",
        "Perimeter": "The perimeter",
        "Compactness": "Compactness",
        "NormalizedDepth": "Normalized sulci depth",
        "SulciCount": "Number of sulci",
        "PrimarySulciCount": "Number of primary sulci",
        "SecondarySulciCount": "Number of secondary sulci",
        "TertiarySulciCount": "Number of tertiary sulci",
        "UnclassifiedSulciCount": "Number of unclassified sulci",
        "PrimaryMeanDepth": "Mean depth of primary sulci",
        "SecondaryMeanDepth": "Mean depth of secondary sulci",
        "TertiaryMeanDepth": "Mean depth of tertiary sulci",
        "UnclassifiedMeanDepth": "Mean depth of unclassified sulci",
    }

    column_to_description = {
        "LGI": (
            "Local gyrification index — ratio of the cortical contour length "
            "to the length of its smoothed outer envelope. Higher values mean "
            "a more folded cortex."
        ),
        "CellDensity": (
            "Peak cortical cell density reached in the simulation (cells per "
            "unit area). Drives how strongly the cortex grows and folds."
        ),
        "MinDepth": (
            "Depth of the shallowest sulcus, measured from the outer contour "
            "to the sulcal fundus."
        ),
        "MeanDepth": (
            "Average depth across all detected sulci — an overall measure of "
            "how deep the folding is."
        ),
        "MaxDepth": (
            "Depth of the deepest sulcus. Sensitive to a single dominant fold "
            "rather than the overall pattern."
        ),
        "area": (
            "Total cortical area enclosed by the contour. Grows with both "
            "expansion and folding."
        ),
        "Perimeter": (
            "Length of the cortical contour. Grows as the cortex folds, since "
            "folding adds contour without adding much area."
        ),
        "Compactness": (
            "How close the slice is to a circle (4π·area / perimeter²). A "
            "smooth disc approaches 1; folding pushes it toward 0."
        ),
        "NormalizedDepth": (
            "Mean sulci depth divided by the maximum depth — describes how "
            "even the folding is, independent of overall brain size."
        ),
        "SulciCount": (
            "Total number of detected sulci, summed across all sulcus classes."
        ),
        "PrimarySulciCount": "Number of sulci classified as primary.",
        "SecondarySulciCount": "Number of sulci classified as secondary.",
        "TertiarySulciCount": "Number of sulci classified as tertiary.",
        "UnclassifiedSulciCount": (
            "Number of detected sulci that could not be assigned to a class."
        ),
        "PrimaryMeanDepth": "Average depth across the primary sulci only.",
        "SecondaryMeanDepth": "Average depth across the secondary sulci only.",
        "TertiaryMeanDepth": "Average depth across the tertiary sulci only.",
        "UnclassifiedMeanDepth": (
            "Average depth across the sulci that could not be classified."
        ),
    }

    # Objective rows added when the dialog opens, in preference order.
    # Restricted to what the file actually provides.
    DEFAULT_OBJECTIVES = ["LGI", "MaxDepth", "CellDensity", "MinDepth",
                          "MeanDepth", "area"]

    # Constraints applied by default, when their column is present.
    DEFAULT_CONSTRAINTS = [
        {"column": "CellDensity", "op": "<=", "value": 1000},
        {"column": "SulciCount", "op": ">=", "value": 1},
    ]

    OPERATORS = ["<=", ">="]

    OPERATOR_LABELS = {
        "<=": "≤  (at most)",
        ">=": "≥  (at least)",
    }

    ALGORITHMS = ["NSGA-II", "NSGA-III"]

    algo_to_description = {
        "NSGA-II": (
            "Non-dominated Sorting Genetic Algorithm II — fast and reliable "
            "for up to 3 objectives. Uses crowding distance to spread the "
            "Pareto front."
        ),
        "NSGA-III": (
            "Non-dominated Sorting Genetic Algorithm III — extends NSGA-II "
            "with reference directions, keeping the Pareto front well spread "
            "when 4 or more objectives are optimised."
        ),
    }

    TERMINATION_TOOLTIP = (
        "Number of generations the solver evolves before stopping "
        "(10–500, in steps of 10). More generations give a better converged "
        "Pareto front but take proportionally longer to run."
    )

    def __init__(self, parent=None, columns: list[str] | None = None,
                 column_ranges: dict[str, tuple[float, float]] | None = None):
        """Initialise the optimisation options dialog.

        Args:
            parent: Parent widget.
            columns: Column names offered as objectives and constraints,
                from :func:`helpers.read_excel.get_optimizable_columns`.
            column_ranges: ``{column: (min, max)}`` observed in the data,
                used to bound the constraint spin boxes.
        """
        super().__init__(parent)
        self.setWindowTitle("Optimization Options")
        self.setModal(True)
        self.resize(720, 640)

        self.columns = list(columns or [])
        self.column_ranges = dict(column_ranges or {})

        self.objective_rows: list[dict] = []
        self.constraint_rows: list[dict] = []

        self.selected_algorithm = QtWidgets.QComboBox()
        self.selected_algorithm.addItems(self.ALGORITHMS)
        self.termination_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        # discrete steps: 1..50 mapped to 10..500
        self.termination_slider.setRange(1, 50)
        self.termination_slider.setSingleStep(1)
        self.termination_slider.setPageStep(1)
        self.termination_slider.setTickInterval(1)
        self.termination_slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.termination_slider.setMinimumWidth(420)
        self.termination_value_label = QtWidgets.QLabel("200")
        self.termination_value_label.setMinimumWidth(40)

        self._build_ui()
        self._set_defaults()

    # ---------------------------------------------------------------- labels

    def display_name(self, column: str) -> str:
        """Return the friendly label for *column*, or the column name itself."""
        return self.column_to_name.get(column, column)

    def column_tooltip(self, column: str) -> str:
        """Return the hover description for *column*.

        Columns with no curated description still get the observed value
        range, which is the only thing that can be said about a metric this
        dialog has never heard of.
        """
        description = self.column_to_description.get(column)
        if description is None:
            description = f"'{column}' — a numeric column read from the selected Excel file."
        span = self.column_ranges.get(column)
        if span is not None:
            description += f"\n\nRange in the selected data: {span[0]:g} to {span[1]:g}."
        return description

    # ------------------------------------------------------------------- ui

    def _build_ui(self):
        """Construct the full dialog layout: objectives, constraints, solver settings."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        obj_group = QtWidgets.QGroupBox("Objectives")
        obj_outer = QtWidgets.QVBoxLayout(obj_group)
        obj_outer.setContentsMargins(8, 8, 8, 8)
        obj_outer.setSpacing(6)

        # Many objective rows would otherwise grow the dialog past the screen.
        obj_scroll = QtWidgets.QScrollArea()
        obj_scroll.setWidgetResizable(True)
        obj_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        obj_scroll.setMaximumHeight(220)
        obj_inner = QtWidgets.QWidget()
        self.objective_layout = QtWidgets.QVBoxLayout(obj_inner)
        self.objective_layout.setContentsMargins(0, 0, 0, 0)
        self.objective_layout.setSpacing(6)
        self.objective_layout.addStretch(1)
        obj_scroll.setWidget(obj_inner)
        obj_outer.addWidget(obj_scroll)

        self.no_objectives_label = QtWidgets.QLabel(
            "No numeric metric columns were found in the selected file(s)."
            if not self.columns
            else f"No objectives — add at least {self.MIN_OBJECTIVES}."
        )
        self.no_objectives_label.setWordWrap(True)
        obj_outer.addWidget(self.no_objectives_label)

        obj_add_row = QtWidgets.QHBoxLayout()
        self.btn_add_objective = QtWidgets.QPushButton("+ Add objective")
        self.btn_add_objective.setToolTip(
            "Optimise another column. At least "
            f"{self.MIN_OBJECTIVES} objectives are required — the solver "
            "trades them off against each other."
        )
        self.btn_add_objective.clicked.connect(lambda: self._add_objective_row())
        self.btn_add_objective.setEnabled(bool(self.columns))
        obj_add_row.addWidget(self.btn_add_objective)
        obj_add_row.addStretch(1)
        obj_outer.addLayout(obj_add_row)
        root.addWidget(obj_group)

        con_group = QtWidgets.QGroupBox("Constraints")
        con_outer = QtWidgets.QVBoxLayout(con_group)
        con_outer.setContentsMargins(8, 8, 8, 8)
        con_outer.setSpacing(6)
        self.constraint_layout = QtWidgets.QVBoxLayout()
        self.constraint_layout.setSpacing(6)
        con_outer.addLayout(self.constraint_layout)

        self.no_constraints_label = QtWidgets.QLabel("No constraints — every slice is eligible.")
        self.no_constraints_label.setToolTip(
            "Without constraints the optimiser considers every row in the file."
        )
        con_outer.addWidget(self.no_constraints_label)

        add_row = QtWidgets.QHBoxLayout()
        self.btn_add_constraint = QtWidgets.QPushButton("+ Add constraint")
        self.btn_add_constraint.setToolTip(
            "Bound another column. Only slices satisfying every constraint are considered."
        )
        self.btn_add_constraint.clicked.connect(lambda: self._add_constraint_row())
        self.btn_add_constraint.setEnabled(bool(self.columns))
        add_row.addWidget(self.btn_add_constraint)
        add_row.addStretch(1)
        con_outer.addLayout(add_row)
        root.addWidget(con_group)

        algo_group = QtWidgets.QGroupBox("Solver parameters")
        algo_layout = QtWidgets.QFormLayout(algo_group)
        algo_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        algo_layout.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        algo_layout.setVerticalSpacing(10)
        algo_layout.setHorizontalSpacing(12)
        algo_row = QtWidgets.QHBoxLayout()
        algo_row.addWidget(self.selected_algorithm)
        algo_row.addStretch(1)
        algo_holder = QtWidgets.QWidget()
        algo_holder.setLayout(algo_row)
        algo_label = QtWidgets.QLabel("Algorithm:")
        algo_label.setToolTip("Evolutionary solver used to search for the Pareto-optimal set.")
        algo_layout.addRow(algo_label, algo_holder)
        self.selected_algorithm.currentTextChanged.connect(self._update_algorithm_tooltip)
        self._update_algorithm_tooltip(self.selected_algorithm.currentText())

        self.termination_slider.setToolTip(self.TERMINATION_TOOLTIP)
        self.termination_value_label.setToolTip(self.TERMINATION_TOOLTIP)
        term_row = QtWidgets.QHBoxLayout()
        term_row.addWidget(self.termination_slider)
        self.termination_value_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.termination_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.termination_value_label.setMinimumWidth(52)
        term_row.addWidget(self.termination_value_label)
        term_holder = QtWidgets.QWidget()
        term_holder.setLayout(term_row)
        term_label = QtWidgets.QLabel("Termination Criterion (n_gen):")
        term_label.setToolTip(self.TERMINATION_TOOLTIP)
        algo_layout.addRow(term_label, term_holder)
        root.addWidget(algo_group)
        self.termination_slider.valueChanged.connect(
            lambda v: self.termination_value_label.setText(str(v * 10))
        )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_defaults = QtWidgets.QPushButton("Defaults")
        buttons.addButton(btn_defaults, QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        btn_defaults.clicked.connect(self._set_defaults)
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ----------------------------------------------------------- objectives

    def _add_objective_row(self, column: str | None = None,
                           direction: str = "maximize"):
        """Append an objective row (column / direction) to the dialog."""
        if not self.columns:
            return

        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)
        chk.toggled.connect(self._update_algorithm_availability)

        combo = QtWidgets.QComboBox()
        for col in self.columns:
            combo.addItem(self.display_name(col), col)
        combo.setMinimumWidth(240)

        dir_combo = QtWidgets.QComboBox()
        for key in self.OBJECTIVE_DIRECTIONS:
            dir_combo.addItem(self.DIRECTION_LABELS[key], key)
        dir_combo.setMinimumWidth(120)

        btn_remove = QtWidgets.QToolButton()
        btn_remove.setText("✕")
        btn_remove.setToolTip("Remove this objective.")

        row.addWidget(chk)
        row.addWidget(combo)
        row.addWidget(dir_combo)
        row.addWidget(btn_remove)
        row.addStretch(1)

        entry = {"holder": holder, "check": chk, "column": combo,
                 "direction": dir_combo, "remove": btn_remove}
        self.objective_rows.append(entry)
        # The layout ends with a stretch that keeps the rows top-aligned, so
        # new rows go before it rather than below it.
        self.objective_layout.insertWidget(self.objective_layout.count() - 1, holder)

        combo.currentIndexChanged.connect(lambda _i, e=entry: self._sync_objective_row(e))
        dir_combo.currentIndexChanged.connect(lambda _i, e=entry: self._sync_objective_row(e))
        btn_remove.clicked.connect(lambda _checked=False, e=entry: self._remove_objective_row(e))

        if column in self.columns:
            combo.setCurrentIndex(self.columns.index(column))
        dir_index = dir_combo.findData(direction)
        if dir_index >= 0:
            dir_combo.setCurrentIndex(dir_index)

        self._sync_objective_row(entry)
        self._update_objective_placeholder()
        self._update_algorithm_availability()
        return entry

    def _sync_objective_row(self, entry: dict):
        """Refresh a row's tooltips to match its selected column and direction."""
        column = entry["column"].currentData()
        if column is None:
            return
        tooltip = self.column_tooltip(column)
        label = self.display_name(column).lower()
        direction = entry["direction"].currentData()

        entry["column"].setToolTip(tooltip)
        entry["check"].setToolTip(f"Include this objective in the optimisation.\n\n{tooltip}")
        entry["direction"].setToolTip(
            f"Search for solutions with the highest {label}."
            if direction == "maximize"
            else f"Search for solutions with the lowest {label}."
        )

    def _remove_objective_row(self, entry: dict):
        """Delete an objective row from the dialog."""
        if entry not in self.objective_rows:
            return
        self.objective_rows.remove(entry)
        self.objective_layout.removeWidget(entry["holder"])
        entry["holder"].deleteLater()
        self._update_objective_placeholder()
        self._update_algorithm_availability()

    def _clear_objective_rows(self):
        """Remove every objective row."""
        for entry in list(self.objective_rows):
            self._remove_objective_row(entry)

    def _update_objective_placeholder(self):
        """Show the hint only while there are no objective rows."""
        self.no_objectives_label.setVisible(not self.objective_rows)

    # ---------------------------------------------------------- constraints

    def _add_constraint_row(self, column: str | None = None, op: str = "<=",
                            value: float | None = None):
        """Append a constraint row (column / operator / value) to the dialog."""
        if not self.columns:
            return

        holder = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(holder)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        chk = QtWidgets.QCheckBox()
        chk.setChecked(True)
        chk.setToolTip("Apply this constraint to the optimisation.")

        combo = QtWidgets.QComboBox()
        for col in self.columns:
            combo.addItem(self.display_name(col), col)
        combo.setMinimumWidth(220)

        op_combo = QtWidgets.QComboBox()
        for key in self.OPERATORS:
            op_combo.addItem(self.OPERATOR_LABELS[key], key)
        op_combo.setToolTip(
            "≤ keeps slices at or below the value; ≥ keeps slices at or above it."
        )

        edit = QtWidgets.QDoubleSpinBox()
        edit.setDecimals(3)
        edit.setMaximumWidth(140)

        btn_remove = QtWidgets.QToolButton()
        btn_remove.setText("✕")
        btn_remove.setToolTip("Remove this constraint.")

        row.addWidget(chk)
        row.addWidget(combo)
        row.addWidget(op_combo)
        row.addWidget(edit)
        row.addWidget(btn_remove)
        row.addStretch(1)

        entry = {"holder": holder, "check": chk, "column": combo,
                 "op": op_combo, "value": edit}
        self.constraint_rows.append(entry)
        self.constraint_layout.addWidget(holder)

        combo.currentIndexChanged.connect(lambda _i, e=entry: self._sync_constraint_row(e))
        btn_remove.clicked.connect(lambda _checked=False, e=entry: self._remove_constraint_row(e))

        if column in self.columns:
            combo.setCurrentIndex(self.columns.index(column))
        op_index = op_combo.findData(op)
        if op_index >= 0:
            op_combo.setCurrentIndex(op_index)
        self._sync_constraint_row(entry)
        if value is not None:
            edit.setValue(float(value))

        self._update_constraint_placeholder()
        return entry

    def _sync_constraint_row(self, entry: dict):
        """Re-bound a row's spin box and tooltips to its selected column."""
        column = entry["column"].currentData()
        if column is None:
            return
        tooltip = self.column_tooltip(column)
        entry["column"].setToolTip(tooltip)
        entry["check"].setToolTip(
            f"Apply this constraint to the optimisation.\n\n{tooltip}"
        )

        edit = entry["value"]
        span = self.column_ranges.get(column)
        if span is None:
            edit.setRange(0.0, 10.0 ** 9)
            edit.setToolTip(tooltip)
            return

        low, high = span
        # A bound outside the observed range can only be vacuous (≤ above the
        # max) or unsatisfiable (≥ above the max), so the spin box is held to
        # what the data can actually answer.
        edit.setRange(low, high)
        step = (high - low) / 20.0
        edit.setSingleStep(step if step > 0 else 1.0)
        # Integer-valued metrics (sulci counts) read better without decimals.
        edit.setDecimals(0 if float(low).is_integer() and float(high).is_integer() else 3)
        edit.setToolTip(tooltip)

    def _remove_constraint_row(self, entry: dict):
        """Delete a constraint row from the dialog."""
        if entry not in self.constraint_rows:
            return
        self.constraint_rows.remove(entry)
        self.constraint_layout.removeWidget(entry["holder"])
        entry["holder"].deleteLater()
        self._update_constraint_placeholder()

    def _clear_constraint_rows(self):
        """Remove every constraint row."""
        for entry in list(self.constraint_rows):
            self._remove_constraint_row(entry)

    def _update_constraint_placeholder(self):
        """Show the 'no constraints' hint only when there are none."""
        self.no_constraints_label.setVisible(not self.constraint_rows)

    # -------------------------------------------------------------- defaults

    def _set_defaults(self):
        """Reset all UI controls to their factory-default state."""
        # Add a row per default objective the file provides. If it has none of
        # them, fall back to its first columns so the dialog always opens with
        # a usable selection rather than an empty box.
        self._clear_objective_rows()
        defaults = [c for c in self.DEFAULT_OBJECTIVES if c in self.columns]
        if len(defaults) < self.MIN_OBJECTIVES:
            defaults = self.columns[:self.MIN_OBJECTIVES]
        for column in defaults:
            self._add_objective_row(column=column, direction="maximize")

        self._clear_constraint_rows()
        for row in self.DEFAULT_CONSTRAINTS:
            column = row["column"]
            if column not in self.columns:
                continue
            value = float(row["value"])
            span = self.column_ranges.get(column)
            if span is not None:
                value = min(max(value, span[0]), span[1])
            self._add_constraint_row(column=column, op=row["op"], value=value)

        self.selected_algorithm.setCurrentText("NSGA-III")
        self.termination_slider.setValue(20)
        self.termination_value_label.setText("200")
        self._update_algorithm_availability()

    # ------------------------------------------------------------ validation

    def _validate_and_accept(self):
        """Validate selections and accept the dialog, or show a warning."""
        objectives = self.get_selected_objectives()
        if len(objectives) < self.MIN_OBJECTIVES:
            QtWidgets.QMessageBox.warning(
                self, "Invalid Selection",
                f"Select at least {self.MIN_OBJECTIVES} objectives — the "
                "optimiser trades them off against each other, so a single "
                f"objective is just a sort.\n\nCurrently selected: {len(objectives)}.",
            )
            return

        # The same column twice would be optimised twice over, distorting the
        # Pareto front (and, in opposite directions, cannot be satisfied).
        duplicates = {col for col in objectives if objectives.count(col) > 1}
        if duplicates:
            names = ", ".join(sorted(self.display_name(c) for c in duplicates))
            QtWidgets.QMessageBox.warning(
                self, "Invalid Selection",
                f"Each column can only be an objective once. Duplicated: {names}.",
            )
            return

        seen: set[tuple[str, str]] = set()
        for row in self.get_constraints():
            key = (row["column"], row["op"])
            if key in seen:
                QtWidgets.QMessageBox.warning(
                    self, "Invalid Constraint",
                    f"'{self.display_name(row['column'])}' has more than one "
                    f"'{self.OPERATOR_LABELS[row['op']].strip()}' constraint. "
                    "Remove the duplicate.",
                )
                return
            seen.add(key)

        self.accept()

    def _update_algorithm_tooltip(self, name: str):
        """Show the description of the currently selected algorithm on hover."""
        self.selected_algorithm.setToolTip(self.algo_to_description.get(name, name))

    def _update_algorithm_availability(self):
        """Disable NSGA-II when more than three objectives are selected."""
        selected_count = len(self.get_selected_objectives())
        model = self.selected_algorithm.model()
        item = model.item(0)  # NSGA-II
        allow_nsga2 = selected_count <= 3
        if item is not None:
            description = self.algo_to_description.get("NSGA-II", "")
            item.setEnabled(allow_nsga2)
            if allow_nsga2:
                item.setToolTip(description)
            else:
                item.setToolTip(
                    f"{description}\n\nUnavailable: more than 3 objectives are selected."
                )
        item_nsga3 = model.item(1)
        if item_nsga3 is not None:
            item_nsga3.setToolTip(self.algo_to_description.get("NSGA-III", ""))

        if not allow_nsga2 and self.selected_algorithm.currentText() == "NSGA-II":
            self.selected_algorithm.setCurrentText("NSGA-III")

    # --------------------------------------------------------------- getters

    def get_selected_objectives(self) -> list[str]:
        """Return the enabled objectives as DataFrame column names, in row order."""
        objectives: list[str] = []
        for entry in self.objective_rows:
            if not entry["check"].isChecked():
                continue
            column = entry["column"].currentData()
            if column is not None:
                objectives.append(column)
        return objectives

    def get_objective_directions(self) -> dict[str, str]:
        """Return a mapping of selected columns to 'maximize' or 'minimize'."""
        directions: dict[str, str] = {}
        for entry in self.objective_rows:
            if not entry["check"].isChecked():
                continue
            column = entry["column"].currentData()
            if column is not None:
                directions[column] = entry["direction"].currentData() or "maximize"
        return directions

    def get_constraints(self) -> list[dict]:
        """Return the enabled constraints as ``{column, op, value}`` rows."""
        constraints: list[dict] = []
        for entry in self.constraint_rows:
            if not entry["check"].isChecked():
                continue
            column = entry["column"].currentData()
            if column is None:
                continue
            constraints.append({
                "column": column,
                "op": entry["op"].currentData(),
                "value": float(entry["value"].value()),
            })
        return constraints

    def get_selected_algorithms(self) -> str:
        """Return the name of the selected solver algorithm."""
        return self.selected_algorithm.currentText()

    def get_termination_criterion(self) -> int:
        """Return the termination criterion as the number of generations."""
        return int(self.termination_slider.value() * 10)

    def get_settings(self) -> tuple[list[str], list[dict], str, int, dict[str, str]]:
        """Return all dialog settings as a single tuple.

        Returns:
            A tuple of (objectives, constraints, algorithm, n_gen, directions).
        """
        return (
            self.get_selected_objectives(),
            self.get_constraints(),
            self.get_selected_algorithms(),
            self.get_termination_criterion(),
            self.get_objective_directions(),
        )
