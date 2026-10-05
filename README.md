# LeafSim

`leafsim` retrieves **the training examples that a tree-based model treats as most similar to a given input**. It is an example-based **explainable AI (XAI)** technique for decision tree based ensemble methods.

It measures similarity *as the model sees it*, not influence: the returned examples are the ones that follow the same decision paths through the ensemble, not the ones whose removal would change the prediction most.

LeafSim is most useful when you need to explain a prediction to someone who understands the training data. 
In general, when explaining predictions to a domain expert who can judge whether the retrieved examples are reasonable analogues for the case at hand.

It complements feature-attribution methods (SHAP, LIME) rather than replacing them. Where SHAP answers *"which features drove this prediction?"*, LeafSim answers *"which training examples does the model consider comparable to this case?"* — both perspectives are often needed.
The technique is:
- easy to interpret by non-technical domain experts
- complementary to feature-attribution methods like SHAP and LIME
- straightforward to implement and maintain in production
- computationally lightweight: memory stays bounded however many rows you explain

More details can be found in [this blog post](https://datascience.ch/leafsim/) and the version accompanied by code found [here](https://sdsc-innovation.github.io/leafsim/).

# How it works

<img src="resources/leafsim.svg" alt="drawing" width="1000"/>

**Summary**

LeafSim works by tracking which leaf node each sample lands in across every tree of the ensemble. Samples that consistently land in the same leaves are considered similar, because they followed the same sequence of decision rules through the forest. The similarity score is the fraction of trees in which two samples share a leaf (one minus the Hamming distance between their leaf indices), and the most similar training samples are returned as the explanation. This score is the classic random forest proximity of Breiman and Cutler; see [Related work](#related-work).

The result is a human-readable answer to the question: *"which past cases does the model consider most similar to this new input?"*

**Example**

As an example, consider explaining the prediction for this Iris flower observation (see [notebook](notebooks/Simple_Example/Example.ipynb)):

<img src="resources/to_explain.png" alt="drawing" width="600"/>

Using LeafSim, we identify the N training observations the model considers most similar to this one when making the prediction `predictedtarget`. The top 10 look like this:

<img src="resources/explanation.png" alt="drawing" width="600"/>

where `target` is the ground-truth label and `similarity` the LeafSim score ranging from 0 (the two samples share a leaf in no tree) to 1 (they share a leaf in every tree). A score of 1 does not mean the features or targets are identical, only that the model cannot tell the two samples apart.

In this example, the model makes an incorrect prediction because many of the most similar training observations carry a different target label — LeafSim makes this failure mode visible.

# Installation

> **Note:** the name `leafsim` on PyPI belongs to an unrelated project, so `pip install leafsim` installs the wrong package. Install from GitHub instead.

```commandline
pip install git+https://github.com/calyptis/leafsim
```

Or, for development:

```commandline
git clone https://github.com/calyptis/leafsim
uv sync                          # core library only
uv sync --group dev              # include test and lint dependencies
uv sync --group notebooks        # include notebook dependencies (pandas, matplotlib, seaborn)
```

# Supported models

LeafSim works out of the box with these models and any subclass of them:

- scikit-learn: `RandomForest`, `ExtraTrees` and `GradientBoosting` (classifier and regressor)
- XGBoost: `XGBClassifier`, `XGBRegressor`
- LightGBM: `LGBMClassifier`, `LGBMRegressor`
- CatBoost: `CatBoostClassifier`, `CatBoostRegressor`

Any other model can be used by giving it a `get_leaf_indices(X)` method that returns an array of shape `(n_samples, n_trees)`. scikit-learn's `HistGradientBoosting` models do not expose leaf indices and are not supported.

# Usage example

A complete example is available in this [notebook](notebooks/Simple_Example/Example.ipynb). Below is a quick-start:

```python
from leafsim import LeafSim
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris(as_frame=True)
X_train, X_to_explain, y_train, _ = train_test_split(
    data["data"], data["target"], test_size=0.2, random_state=46
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

leafsim_instance = LeafSim(model).fit(X_train)  # caches the training leaf indices
explanation_ids, explanation_similarities = leafsim_instance.explain(X_to_explain, top_n=10)
# explanation_ids:          shape (n_test, top_n) — indices into X_train, most similar first
# explanation_similarities: shape (n_test, top_n) — LeafSim score in [0, 1]

# Retrieve the 10 most similar training examples for the first test observation
X_train.iloc[explanation_ids[0]]

# And their corresponding similarities [0, 1]
# with 1 => landing in the same leaf as the test observation in every tree
explanation_similarities[0]
```

Further calls to `explain()` reuse the cached training leaf indices. Ties are broken by the lowest training index. To get the full `(n_test, n_train)` similarity matrix, use `leafsim_instance.pairwise_similarities(X_to_explain)`; it is dense, so prefer `explain()` for large training sets.

`generate_explanations(X_train, X_to_explain, top_n=10)` remains available as a one-call shorthand for `fit(...).explain(...)`.

# Limitations

- **Similarity, not influence.** LeafSim shows which training examples the model groups with the input. It does not estimate how much each example contributed to the prediction; for that, see the influence-based methods under [Related work](#related-work).
- **All trees count equally.** This suits random forests, whose trees are averaged with equal weight. In boosted models (XGBoost, LightGBM, CatBoost, GradientBoosting) trees contribute very unequally to the prediction, so the score is a rougher proxy there.
- **Cost grows with both data sizes.** Explaining `n_test` rows compares each one against every training row, so time scales with `n_test × n_train × n_trees`. Memory stays bounded because rows are scored in batches.

# Tests

```commandline
uv run pytest tests/
```

# Related work

LeafSim belongs to the family of **example-based (case-based) explanations** — methods that explain a prediction by reference to similar known cases rather than by attributing importance to individual features. This paradigm traces back to Case-Based Reasoning (Aamodt & Plaza, 1994).

**The method is not new.** The LeafSim score is the random forest *proximity* defined by Breiman and Cutler. What LeafSim adds is a small, uniform interface across scikit-learn, XGBoost, LightGBM and CatBoost, aimed at explaining individual predictions to domain experts.

**Closest relatives: tree-ensemble proximities and training-example explanations**

| Work | Relation to LeafSim |
|------|---------------------|
| [Random forest proximities](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm) (Breiman & Cutler) | The same score: fraction of trees in which two samples share a terminal node |
| [RF-GAP](https://arxiv.org/abs/2201.12682) (Rhodes, Cutler & Moon, 2023) | Proximities redefined so that they reproduce the forest's out-of-bag predictions |
| [TREX](https://arxiv.org/abs/2009.05530) (Brophy & Lowd, 2020) | Training-example attributions for tree ensembles via a tree-ensemble kernel |
| [LeafInfluence](https://arxiv.org/abs/1802.06640) (Sharchilev et al., 2018) | Influence of training samples on predictions of gradient boosted trees |
| [Tree space prototypes](https://arxiv.org/abs/1611.07115) (Tan et al., 2020) | A distance for gradient boosted trees, used to select representative prototypes |

**Influence-based approaches for arbitrary models**

For gradient-based training-data influence, see [influence functions](https://arxiv.org/abs/1703.04730) (Koh & Liang, 2017) and [TracIn](https://arxiv.org/abs/2002.08484) (Pruthi et al., 2020). These estimate how much each training example *influenced* a prediction, a different question from the similarity LeafSim measures.

**Tools following similar/complementary approaches**

| Tool | Explanation type | Key question answered |
|------|------------------|-----------------------|
| [SHAP / TreeSHAP](https://github.com/shap/shap) | Feature attribution | Which features drove this prediction? |
| [LIME](https://github.com/marcotcr/lime) | Local linear surrogate | Which features matter locally? |
| [ELI5](https://github.com/eli5-org/eli5) | Feature weights | How does this model use each feature? |
| LeafSim | Example-based | Which training examples does the model consider most similar? |

# Structure of repo

- `leafsim/` — the Python library
- `notebooks/` — usage examples (simple Iris classification and advanced car-price regression)

# Further resources

For a more comprehensive usage example, refer to this [blog post](https://sdsc-innovation.github.io/leafsim/) and the corresponding [notebook](notebooks/Advanced_Example/Example.ipynb).

# Citation

If you use this software in your work, it would be appreciated if you would cite it using the following BibTeX reference:

```
@software{leafsim,
  author = {Lucas Chizzali},
  title = {LeafSim: Example based XAI for decision tree ensembles},
  url = {https://github.com/calyptis/LeafSim},
  date = {2022-11-14},
}
```
