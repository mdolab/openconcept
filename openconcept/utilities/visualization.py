from matplotlib import pyplot as plt
import numpy as np

def plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases, x_label=None, y_labels=None,  marker='o', plot_title='Trajectory'):

    val_list = []
    for phase in phases:
        val_list.append(prob.get_val(phase + '.' + x_var, units=x_unit))
    x_vec = np.concatenate(val_list)

    for i, y_var in enumerate(y_vars):
        val_list = []
        for phase in phases:
            val_list.append(prob.get_val(phase + '.' + y_var, units=y_units[i]))
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

