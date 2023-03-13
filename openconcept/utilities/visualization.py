try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
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


def plot_OAS_mesh(OAS_mesh, ax=None, set_xlim=True, turn_off_axis=True):
    """
    Plots the wing planform mesh from OpenConcept's OpenAeroStruct interface.

    Parameters
    ----------
    OAS_mesh : ndarray
        The mesh numpy array pulled out of the aerodynamics model (the output
        of the mesh component).
    ax : matplotlib axis object (optional)
        Axis on which to plot the wingbox. If not specified, this function
        creates and returns a figure and axis.
    set_xlim : bool (optional)
        Set the x limits on the axis to just fit the span of the wing.
    turn_off_axis : bool (optional)
        Turn off the spines, ticks, etc.

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
        If ax is not specified, returns the created figure and axis.
    """
    # Duplicate the half mesh that is used by OAS
    mesh = np.hstack((OAS_mesh, OAS_mesh[:, -2::-1, :] * np.array([1, -1, 1])))
    chord_wing_front = mesh[0, :, 0]
    span_wing = mesh[0, :, 1]
    chord_wing_back = mesh[-1, :, 0]

    return_ax = False
    if ax is None:
        # Figure out the size the plot should be
        span = 2 * np.max(span_wing)
        y_range = abs(np.min(chord_wing_front) - np.max(chord_wing_back))
        x_size = 10
        y_size = y_range / span * x_size

        fig, ax = plt.subplots(figsize=(x_size, y_size))
        return_ax = True

    # Plot wing
    ax.fill_between(
        span_wing,
        -chord_wing_front,
        -chord_wing_back,
        facecolor="#d5e4f5",
        zorder=0,
        edgecolor="#919191",
        clip_on=False,
    )

    # Plot aerodynamic mesh
    x = mesh[:, :, 1]
    y = -mesh[:, :, 0]
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color="#919191", zorder=2, linewidth=0.3))
    ax.add_collection(LineCollection(segs2, color="#919191", zorder=2, linewidth=0.3))

    # Set final plot details
    ax.set_aspect("equal")
    if turn_off_axis:
        ax.set_axis_off()
    if set_xlim:
        ax.set_xlim((np.min(span_wing), np.max(span_wing)))

    if return_ax:
        return fig, ax
