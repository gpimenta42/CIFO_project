
import scikit_posthocs as spx
from scipy.stats import kruskal
import scikit_posthocs as sp
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

#------------------------------------------------------------------------------
# NON-PARAMETRIC 3+ GROUPS
#------------------------------------------------------------------------------

def kruskal_wallis_test(
    groups: list[list], 
    alpha: float = 0.01
    ) -> None:
    """
    Performs the Kruskal-Wallis H-test to compare three or more groups.

    Args:
        groups (list[list]): List of lists, where each inner list contains numerical data (e.g., fitness scores) for a group.
        alpha (float, optional): Significance level for the test, between 0.0 and 1.0, defaults to 0.01.

    Returns:
        None: Prints test statistics, p-value, Kendall's W, and whether the null hypothesis is rejected.
    """
    
    kruskal_wallis_stats, p_value = kruskal(*groups)
    
    # Total sample size and number of groups
    N = sum(len(g) for g in groups)
    k = len(groups)

    # Epsilon-squared effect size
    eta_squared = (kruskal_wallis_stats - k + 1) / (N - k)
    
    print(f"Kruskal-Wallis H-test statistic: {kruskal_wallis_stats:.3f}")
    print(f"P-value: {p_value:.3e}")
    print(f"Eta-squared effect size (η²): {eta_squared:.3f}")
    print(100*'-')
    
    if p_value < alpha:
        print("REJECT H0: At least one configuration has significantly different fitness scores.")
    else:
        print("FAIL TO REJECT H0: No significant differences.")


def run_dunn_posthoc(
    df: pd.DataFrame, 
    value_col: str, 
    group_col: str, 
    p_adjust_method: str = "holm"
    ) -> pd.DataFrame:
    """
    Performs Dunn's post-hoc test for pairwise comparisons after a significant Kruskal-Wallis test.

    Args:
        df (pd.DataFrame): DataFrame containing the data with columns for values and group labels.
        value_col (str): Name of the column containing the numerical values (e.g., fitness scores).
        group_col (str): Name of the column containing group labels.
        p_adjust_method (str, optional): Method for p-value adjustment, defaults to "holm".

    Returns:
        pd.DataFrame: Matrix of p-values from Dunn's test.
    """
    
    dunn_results = sp.posthoc_dunn(
        df,
        val_col=value_col,
        group_col=group_col,
        p_adjust=p_adjust_method
    )
    return dunn_results


#------------------------------------------------------------------------------
# NON-PARAMETRIC 2 GROUPS
#------------------------------------------------------------------------------
def mann_whitney_u_test(
    group1: list, 
    group2: list, 
    alpha: float = 0.01
    ) -> None:
    """
    Performs the Mann-Whitney U test to compare two groups.

    Args:
        group1 (list[float]): List of numerical values for the first group (e.g., fitness scores).
        group2 (list[float]): List of numerical values for the second group (e.g., fitness scores).
        alpha (float, optional): Significance level for the test, between 0.0 and 1.0, defaults to 0.01.

    Returns:
        None: Prints test statistics, p-value, rank-biserial correlation (if significant), and whether the null hypothesis is rejected.
    """
    
    n1 = len(group1)
    n2 = len(group2)

    stat, p = mannwhitneyu(group1, group2, alternative='two-sided')

    print(f"Mann-Whitney U statistic: {stat:.3f}")
    print(f"P-value: {p:.3e}")
    print(100*'-')

    if p < alpha:
        print("REJECT H0: Groups differ significantly.")
        # Rank-biserial correlation: r_rb = (2U) / (n1*n2) - 1
        r_rb = (2 * stat) / (n1 * n2) - 1
        print(f"Rank-biserial correlation: {r_rb:.3f}")
    else:
        print("FAIL TO REJECT H0: No significant difference.")



#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

def plot_dunn_results(
    dunn_results: pd.DataFrame, 
    n_decimals: int = 4, 
    shrink: float = 0.7, 
    figsize: tuple[int, int] = (20, 20)
    ) -> None:
    """
    Plots a heatmap of p-values from Dunn's post-hoc test.

    Args:
        dunn_results (pd.DataFrame): Matrix of p-values from Dunn's test.
        n_decimals (int, optional): Number of decimal places for p-value annotations, defaults to 4.
        shrink (float, optional): Shrink factor for the colorbar, defaults to 0.7.
        figsize (tuple[int, int], optional): Figure size as (width, height), defaults to (20, 20).

    Returns:
        None: Displays the heatmap.
    """
    
    # Create mask for upper triangle to hide it in heatmap
    mask = np.triu(np.ones_like(dunn_results, dtype=bool))

    # Plot the heatmap with only lower triangle shown
    plt.figure(figsize=figsize)
    sns.heatmap(dunn_results, 
                mask=mask,
                cmap="GnBu", 
                annot=True, 
                fmt=f".{n_decimals}f", 
                cbar_kws={'label': 'p-value', 'shrink': shrink},
                linewidths=0.5, 
                linecolor='gray',
                square=True,
                center=0.1)

    plt.title("Dunn's Post-Hoc Test P-values (Holm-adjusted)", fontsize=16)
    plt.xlabel("Configuration")
    plt.ylabel("Configuration")

    plt.xticks(rotation=30, ha='right')  # Rotate bottom x-axis labels by 45 degrees
    plt.tight_layout()
    plt.show()


def plot_fitness_distribution(
    df: pd.DataFrame, 
    column: str, 
    palette: dict, 
    title: str = "Fitness Distribution",
    xlabel: str = "Fitness",
    ylabel: str = "Density", 
    figsize: tuple[int, int] = (12, 6),
    x: str = 'fitness'
    ) -> None:
    """
    Plots kernel density estimates of fitness distributions for different groups.

    Args:
        df (pd.DataFrame): DataFrame containing the data with a fitness column and group labels.
        column (str): Name of the column containing group labels.
        palette (dict): Dictionary mapping group names to color hex codes.
        title (str, optional): Figure title, defaults to "Fitness Distribution".
        xlabel (str, optional): X-axis label, defaults to "Fitness".
        ylabel (str, optional): Y-axis label, defaults to "Density".
        figsize (tuple[int, int], optional): Figure size as (width, height), defaults to (12, 6).
        x (str, optional): Name of the column containing fitness values, defaults to 'fitness'.

    Returns:
        None: Displays the plot.
    """
    plt.figure(figsize=figsize)

    for name, colour in palette.items():
        subset = df[df[column] == name]

        sns.kdeplot(
            data=subset,
            x=x,
            alpha=0.5,
            linewidth=1.5,
            fill=True,
            label=str(name).replace("_", " "),
            color=colour,
            )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title=column.replace("_", " ").title())
    plt.tight_layout()
    plt.show()
