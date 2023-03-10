try:
    from matplotlib import pyplot as plt
except ImportError:
    # don't want a matplotlib dependency on Travis/Appveyor
    pass
import numpy as np


def plot_trajectory(
    prob, x_var, x_unit, y_vars, y_units, phases, x_label=None, y_labels=None, marker="o", plot_title="Trajectory"
):
    val_list = []
    for phase in phases:
        val_list.append(prob.get_val(phase + "." + x_var, units=x_unit))
    x_vec = np.concatenate(val_list)

    for i, y_var in enumerate(y_vars):
        val_list = []
        for phase in phases:
            val_list.append(prob.get_val(phase + "." + y_var, units=y_units[i]))
        y_vec = np.concatenate(val_list)
        plt.figure()
        plt.plot(x_vec, y_vec, marker)
        if x_label is None:
            plt.xlabel(x_var)
        else:
            plt.xlabel(x_label)
        if y_labels is not None:
            if y_labels[i] is not None:
                plt.ylabel(y_labels[i])
        else:
            plt.ylabel(y_var)
        plt.title(plot_title)
    plt.show()


def plot_trajectory_grid(
    cases,
    x_var,
    x_unit,
    y_vars,
    y_units,
    phases,
    x_label=None,
    y_labels=None,
    grid_layout=[5, 2],
    marker="o",
    savefig=None,
    figsize=None,
):
    """
    Plots multiple trajectories against each other
    Cases is a list of OpenMDAO CaseReader cases which act like OpenMDAO problems
    """
    x_vecs = []
    for case in cases:
        val_list = []
        for phase in phases:
            val_list.append(case.get_val(phase + "." + x_var, units=x_unit))
        x_vec = np.concatenate(val_list)
        x_vecs.append(x_vec)

    file_counter = -1
    counter_within_file = 0
    if figsize is None:
        figsize = (8.5, 11)
    for i, y_var in enumerate(y_vars):
        if counter_within_file % (grid_layout[0] * grid_layout[1]) == 0:
            if file_counter >= 0:
                # write the file
                if savefig is not None:
                    plt.savefig(savefig + "_" + str(file_counter) + ".pdf")
            fig, axs = plt.subplots(grid_layout[0], grid_layout[1], sharex=True, figsize=figsize)
            file_counter += 1
            counter_within_file = 0

        row_no = counter_within_file // grid_layout[1]
        col_no = counter_within_file % grid_layout[1]
        for j, case in enumerate(cases):
            val_list = []
            for phase in phases:
                val_list.append(case.get_val(phase + "." + y_var, units=y_units[i]))
            y_vec = np.concatenate(val_list)
            axs[row_no, col_no].plot(x_vecs[j], y_vec, marker)
        if row_no + 1 == grid_layout[0]:  # last row
            if x_label is None:
                axs[row_no, col_no].set(xlabel=x_var)
            else:
                axs[row_no, col_no].set(xlabel=x_label)
        if y_labels is not None:
            if y_labels[i] is not None:
                axs[row_no, col_no].set(ylabel=y_labels[i])
        else:
            axs[row_no, col_no].set(y_var)

        counter_within_file += 1

    if savefig is not None:
        fig.tight_layout()
        plt.savefig(savefig + "_" + str(file_counter) + ".pdf")
