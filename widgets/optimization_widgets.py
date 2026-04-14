"""Multi-objective optimisation configuration dialog.

Presents checkboxes for objectives (gyrification index, sulci depth, etc.),
constraints with numeric limits, solver algorithm selection, and a
termination-criterion slider so the user can fully configure an
optimisation run before launching it.
"""

from deps import *


class OptimizationOptionsDialog(QtWidgets.QDialog):
    """Dialog for configuring multi-objective optimisation parameters.

    The user selects which objectives to optimise (and their direction),
    enables or disables constraints with numeric bounds, picks a solver
    algorithm (NSGA-II / NSGA-III), and sets the number of generations.
    """

    OBJECTIVE_DIRECTIONS = ["Maximize", "Minimize"]

    obj_to_name = {
        "perimeter_rate": "Local gyrification index",
        "cell_density": "The max cortical cell density",
        "min_d_value": "The minimum sulci depth",
        "mean_d_value": "The mean sulci depth",
        "max_d_value": "The maximum sulci depth",
        "area": "The area"
    }

    const_to_name = {
        "max_cell_density": "The max cortical cell density",
        "number_SulciCount": "Number of sulci",
    }


    OBJECTIVES = [
        "perimeter_rate",
        "max_d_value",
        "cell_density",
        "min_d_value",
        "mean_d_value",
        "area",
    ]

    CONSTRAINTS_DEFAULTS = {
        "max_cell_density": 1000,
        "number_SulciCount": 1,
    }

    ALGORITHMS = ["NSGA-II", "NSGA-III"]

    def __init__(self, parent=None, max_sulci_count: int | None = None, max_cell_density: float | None = None):
        """Initialise the optimisation options dialog.

        Args:
            parent: Parent widget.
            max_sulci_count: Upper bound for the sulci-count constraint
                (None for unlimited).
            max_cell_density: Upper bound for the cell-density constraint
                (None for unlimited).
        """
        super().__init__(parent)
        self.setWindowTitle("Optimization Options")
        self.setModal(True)
        self.resize(660, 560)

        self.objective_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.objective_direction_groups: dict[str, QtWidgets.QButtonGroup] = {}
        self.constraint_checks: dict[str, QtWidgets.QCheckBox] = {}
        self.constraint_edits: dict[str, QtWidgets.QSpinBox] = {}
        self.constraint_limits: dict[str, float | int | None] = {
            "max_cell_density": max_cell_density,
            "number_SulciCount": max_sulci_count,
            "max_MaxDepth": None,
        }
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

    def _build_ui(self):
        """Construct the full dialog layout: objectives, constraints, solver settings."""
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        obj_group = QtWidgets.QGroupBox("Objectives")
        obj_layout = QtWidgets.QGridLayout(obj_group)
        obj_layout.setContentsMargins(8, 8, 8, 8)
        obj_layout.setHorizontalSpacing(16)
        obj_layout.setVerticalSpacing(8)
        for obj in self.OBJECTIVES:
            display_name = self.obj_to_name.get(obj, obj)
            chk = QtWidgets.QCheckBox(display_name)
            chk.setToolTip(obj)
            chk.toggled.connect(self._update_algorithm_availability)
            chk.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
            rb_max = QtWidgets.QRadioButton("Maximize")
            rb_min = QtWidgets.QRadioButton("Minimize")
            rb_max.setChecked(True)
            direction_group = QtWidgets.QButtonGroup(self)
            direction_group.addButton(rb_max)
            direction_group.addButton(rb_min)

            direction_holder = QtWidgets.QWidget()
            direction_layout = QtWidgets.QHBoxLayout(direction_holder)
            direction_layout.setContentsMargins(0, 0, 0, 0)
            direction_layout.setSpacing(12)
            direction_layout.addWidget(rb_max)
            direction_layout.addWidget(rb_min)
            direction_layout.addStretch(1)
            direction_holder.setMinimumWidth(220)

            self.objective_checks[obj] = chk
            self.objective_direction_groups[obj] = direction_group
            row_idx = len(self.objective_checks) - 1
            obj_layout.addWidget(chk, row_idx, 0)
            obj_layout.addWidget(direction_holder, row_idx, 1, QtCore.Qt.AlignmentFlag.AlignRight)
            obj_layout.setRowMinimumHeight(row_idx, 30)
        obj_layout.setColumnStretch(0, 1)
        obj_layout.setColumnStretch(1, 0)
        root.addWidget(obj_group)

        con_group = QtWidgets.QGroupBox("Constraints")
        con_form = QtWidgets.QFormLayout(con_group)
        con_form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        con_form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        con_form.setVerticalSpacing(8)
        con_form.setHorizontalSpacing(12)
        for key, default in self.CONSTRAINTS_DEFAULTS.items():
            row = QtWidgets.QHBoxLayout()
            row.setSpacing(10)
            chk = QtWidgets.QCheckBox()
            edit = QtWidgets.QSpinBox()
            edit.setMaximumWidth(140)
            edit.setMinimum(0)
            limit = self.constraint_limits.get(key)
            if limit is not None:
                edit.setMaximum(int(limit))
                edit.setToolTip(f"Maximum allowed: {limit}")
            else:
                edit.setMaximum(10**9)

            row.addWidget(chk)
            row.addWidget(edit)
            row.addStretch(1)

            holder = QtWidgets.QWidget()
            holder.setLayout(row)
            display_name = self.const_to_name.get(key, key)
            lbl = QtWidgets.QLabel(display_name)
            lbl.setWordWrap(True)
            lbl.setMinimumWidth(220)
            lbl.setToolTip(key)
            con_form.addRow(lbl, holder)

            self.constraint_checks[key] = chk
            self.constraint_edits[key] = edit

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
        algo_layout.addRow(QtWidgets.QLabel("Algorithm:"), algo_holder)

        term_row = QtWidgets.QHBoxLayout()
        term_row.addWidget(self.termination_slider)
        self.termination_value_label.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.termination_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.termination_value_label.setMinimumWidth(52)
        term_row.addWidget(self.termination_value_label)
        term_holder = QtWidgets.QWidget()
        term_holder.setLayout(term_row)
        algo_layout.addRow(QtWidgets.QLabel("Termination Criterion (n_gen):"), term_holder)
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

    def _set_defaults(self):
        """Reset all UI controls to their factory-default state."""
        # Select all objectives by default.
        for obj, chk in self.objective_checks.items():
            chk.setChecked(True)
            buttons = self.objective_direction_groups[obj].buttons()
            rb_max = next((b for b in buttons if b.text() == "Maximize"), None)
            if rb_max is not None:
                rb_max.setChecked(True)

        # Enable all constraints with provided defaults.
        for key, chk in self.constraint_checks.items():
            chk.setChecked(True)
            val = self.CONSTRAINTS_DEFAULTS[key]
            limit = self.constraint_limits.get(key)
            if limit is not None:
                val = min(val, limit)
            self.constraint_edits[key].setValue(int(val))

        self.selected_algorithm.setCurrentText("NSGA-III")
        self.termination_slider.setValue(20)
        self.termination_value_label.setText("200")
        self._update_algorithm_availability()

    def _validate_and_accept(self):
        """Validate selections and accept the dialog, or show a warning."""
        if not any(chk.isChecked() for chk in self.objective_checks.values()):
            QtWidgets.QMessageBox.warning(
                self, "Invalid Selection", "Select at least one objective."
            )
            return

        for key, chk in self.constraint_checks.items():
            if not chk.isChecked():
                continue
            value = float(self.constraint_edits[key].value())
            limit = self.constraint_limits.get(key)
            if limit is not None and value > float(limit):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Constraint",
                    f"Constraint '{key}' must be <= {limit}.",
                )
                return

        self.accept()

    def _update_algorithm_availability(self):
        """Disable NSGA-II when more than three objectives are selected."""
        selected_count = sum(1 for chk in self.objective_checks.values() if chk.isChecked())
        model = self.selected_algorithm.model()
        item = model.item(0)  # NSGA-II
        allow_nsga2 = selected_count <= 3
        if item is not None:
            item.setEnabled(allow_nsga2)
            if allow_nsga2:
                item.setToolTip("")
            else:
                item.setToolTip("NSGA-II is available only when 3 or fewer objectives are selected.")

        if not allow_nsga2 and self.selected_algorithm.currentText() == "NSGA-II":
            self.selected_algorithm.setCurrentText("NSGA-III")

    def get_selected_objectives(self) -> list[str]:
        """Return the list of checked objective key strings."""
        return [obj for obj, chk in self.objective_checks.items() if chk.isChecked()]

    def get_objective_directions(self) -> dict[str, str]:
        """Return a mapping of selected objective keys to 'maximize' or 'minimize'."""
        directions: dict[str, str] = {}
        for obj, chk in self.objective_checks.items():
            if not chk.isChecked():
                continue
            checked = self.objective_direction_groups[obj].checkedButton()
            directions[obj] = checked.text().lower() if checked is not None else "maximize"
        return directions

    def get_constraints(self) -> dict[str, float]:
        """Return a dict of enabled constraint keys to their numeric limits."""
        constraints: dict[str, float] = {}
        for key, chk in self.constraint_checks.items():
            if chk.isChecked():
                value = float(self.constraint_edits[key].value())
                if key == "number_SulciCount":
                    value = int(round(value))
                constraints[key] = value
        return constraints

    def get_selected_algorithms(self) -> str:
        """Return the name of the selected solver algorithm."""
        return self.selected_algorithm.currentText()

    def get_termination_criterion(self) -> int:
        """Return the termination criterion as the number of generations."""
        return int(self.termination_slider.value() * 10)

    def get_settings(self) -> tuple[list[str], dict[str, float], str, int, dict[str, str]]:
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
