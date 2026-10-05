"""
Helper functions for the advanced example notebook.

Only used by the notebook and nowhere in the Python library.
"""

import os
import sys

import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_common import conditional_background_gradient  # noqa: E402


def conditional_highlight(s: pd.Series, val: str) -> list[str]:
    """
    Returns single-colour background styling rows in the pandas series s, based on whether their values match val.

    :param s: Column to check if any entry is equal to val
    :param val: Value entries in s should be equal to
    """
    mask = s == val
    return ["background-color: " if i else "background-color: lightcoral;" for i in mask]


def get_similarity_table(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    top_n_ids: np.ndarray,
    top_n_similarities: np.ndarray,
    car_to_explain_id: int,
    top_n_features: list[str],
    formatting: dict[str, str],
) -> pd.DataFrame:
    """
    Generates a stylised table of the most similar cars for a chosen car to be explained (car_to_explain_id).

    Styling:
    - background gradient of numeric feature columns based on the percentage deviation from the car to be explained.
    - single colour background for categorical feature columns based on whether they are
      the same for the car to be explained.

    :param df_train: Dataframe containing training examples (from which the most similar cars are identified)
    :param df_test: Dataframe containing test examples (cars we want to generate explanations for)
    :param top_n_ids: The indices of the training cars that are most similar to a test car
    :param top_n_similarities: The LeafSim score of the most similar training cars (same order as top_n_ids)
    :param car_to_explain_id: Index of the test car to explain
    :param top_n_features: The list of feature columns to visualise in the stylised table
    :param formatting: Custom formatting of the feature columns passed on to pd.DataFrame().style.format(...)
    :return: Stylised dataframe holding the top N most similar cars
    """
    d = df_test.loc[car_to_explain_id, top_n_features + ["predicted_price"]].to_dict()

    df_to_show = (
        df_train.loc[top_n_ids[car_to_explain_id, :]]
        .assign(
            similarity=top_n_similarities[car_to_explain_id, :],
            # Measure the relative price difference between similar cars and the car to explain
            # This column is used to colour the price column shown in the table
            diff_price=lambda x: (x.price - d["predicted_price"]).abs() / d["predicted_price"],
            # Do the same as above for other numeric attributes
            diff_mileage=lambda x: (x.mileage - d["mileage"]).abs() / d["mileage"],
            diff_year=lambda x: (x.year - d["year"]).abs() / d["year"],
            diff_enginesize=lambda x: (x.enginesize - d["enginesize"]).abs() / d["enginesize"],
        )
        # Stable sort keeps LeafSim's tie order (lowest training index first)
        .sort_values(by="similarity", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    df_to_show = (
        df_to_show[top_n_features + ["similarity", "price"]]
        .style.format(formatting)
        # Add a background colour to the certain column based on how
        # far away they are from the car to explain (in relative terms)
        .apply(
            conditional_background_gradient,
            subset="price",
            cmap="Reds",
            df=df_to_show,
            col="diff_price",
            upper_threshold=1,
        )
        .apply(
            conditional_background_gradient,
            subset="mileage",
            cmap="Reds",
            df=df_to_show,
            col="diff_mileage",
            upper_threshold=1.5,
        )
        .apply(
            conditional_background_gradient,
            subset="year",
            cmap="Reds",
            df=df_to_show,
            col="diff_year",
            upper_threshold=1e-3,
        )
        .apply(
            conditional_background_gradient,
            subset="enginesize",
            cmap="Reds",
            df=df_to_show,
            col="diff_enginesize",
            upper_threshold=1,
        )
        # Add a background gradient for the similarity column
        .background_gradient(subset="similarity", cmap="Blues", vmax=1, vmin=0)
        # Highlight cells in categorical columns if they differ from the value of the car to explain
        .apply(conditional_highlight, val=d["brand"], subset=["brand"])
        .apply(conditional_highlight, val=d["fueltype"], subset=["fueltype"])
        .apply(conditional_highlight, val=d["brand"], subset=["brand"])
        .apply(conditional_highlight, val=d["transmission"], subset=["transmission"])
        .apply(conditional_highlight, val=d["model"], subset=["model"])
    )

    return df_to_show


def get_similarity_plots(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    similarities: np.ndarray,
    car_to_explain_id: int,
    test_avg_similarity: np.ndarray,
) -> None:
    """
    Visualises two matplotlib plots to summarise different similarity aspects.

    The first plot on the left plots the price versus LeafSim score for all the training samples given a specific
    car one wishes to explain (car_to_explain_id).
    The second plot on the right plots the distribution of the average LeafSim score of the top 50 most similar cars
    for each of the cars in df_test.

    An interpretation of these plots can be found in the notebook.

    :param df_train: Dataframe containing training examples
    :param df_test: Dataframe containing test examples
    :param similarities: LeafSim scores between every test car and every training car
    :param car_to_explain_id: Index of the test car to explain
    :param test_avg_similarity: Average LeafSim score of the top 50 training cars for each test car
    :return: Shows matplotlib figure
    """
    tmp = pd.DataFrame(
        {"similarity": similarities[car_to_explain_id, :], "price": df_train.price}
    )

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    axes = axes.flatten()
    ax = axes[0]
    ax.axhline(
        df_test.loc[car_to_explain_id, "predicted_price"],
        linewidth=1,
        c="darkviolet",
        linestyle="--",
        label="Car to explain - predicted price",
    )
    ax.axhline(
        df_test.loc[car_to_explain_id, "price"],
        linewidth=1,
        c="royalblue",
        linestyle="--",
        label="Car to explain - actual price",
    )
    sns.lineplot(
        data=tmp,
        x="similarity",
        y="price",
        linewidth=1,
        color="darkviolet",
        alpha=0.5,
        ci="sd",
        label="$\mu$ and $\sigma$ of prices of cars in training set",
        ax=ax,
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:,.0f}"))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 100:,.0f}"))
    ax.set_ylabel("Price [£]")
    ax.set_xlabel("LeafSim Score [%]")
    ax.set_title("Price vs LeafSim Score")
    ax.legend(loc="best")

    ax = axes[1]
    sns.histplot(
        test_avg_similarity,
        ax=ax,
        stat="probability",
        color="darkviolet",
        alpha=0.5,
        label="All cars in test set",
    )
    ax.axvline(
        test_avg_similarity[car_to_explain_id],
        linewidth=1,
        c="royalblue",
        label="Car to explain",
    )
    ax.set_title("Distribution of Average Top 50 LeafSim Scores", fontsize=22)
    ax.set_xlabel("Average Top 50 LeafSim Score [%]")
    ax.set_ylabel("Share [%]")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 100:,.0f}"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x * 100:,.0f}"))
    ax.legend()
    plt.tight_layout()
