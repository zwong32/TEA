# figure_style.py

import matplotlib as mpl
import matplotlib.pyplot as plt


def set_figure_defaults():
    """
    Set global matplotlib defaults for publication-quality figures.
    """

    # Font settings
    mpl.rcParams["font.family"] = "Times New Roman"

    # mpl.rcParams["font.size"] = 12

    # mpl.rcParams["axes.labelsize"] = 12
    # mpl.rcParams["axes.titlesize"] = 12
    # mpl.rcParams["xtick.labelsize"] = 10
    # mpl.rcParams["ytick.labelsize"] = 10

    # Line thickness
    # mpl.rcParams["lines.linewidth"] = 6 # ideal for conference presentations

    mpl.rcParams["axes.labelweight"] = "bold"
    
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 9
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["lines.linewidth"] = 1.5
    
    # Axis appearance
    mpl.rcParams["axes.linewidth"]    = 1.0
    mpl.rcParams["xtick.major.width"] = 1.0
    mpl.rcParams["ytick.major.width"] = 1.0
    mpl.rcParams["xtick.direction"] = "out"
    mpl.rcParams["ytick.direction"] = "out"

    # Legend
    mpl.rcParams["legend.frameon"] = True
    mpl.rcParams["legend.fontsize"] = 12
    mpl.rcParams["legend.loc"] = "best"

    # LaTeX rendering
    mpl.rcParams["text.usetex"] = True

    # Figure appearance
    mpl.rcParams["figure.facecolor"] = "white"

    # Savefig settings
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["savefig.bbox"] = "tight"


def make_single_column_figure():
    """
    Create a journal-style single-column figure.

    Width = 3.4 in
    Height = 2.5 in
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    return fig, ax


# def set_fig_aesthetics():
#     markers = ["o", "+", "x", "s", "^", "v", "D", "p", "h", "*", ">", "<"]
#     colors = ["blue", "orange", "purple", "green", "red", "brown", "pink", "gray", "olive", "turquoise", "mediumslateblue", "lightgreen"]
    
#     return markers, colors