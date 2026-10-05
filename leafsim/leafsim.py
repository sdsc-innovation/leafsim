"""
LeafSim — example-based explanations for tree-based ensemble models.

For a given prediction, LeafSim identifies the training samples most similar to
the sample being explained. Similarity is the fraction of trees in the ensemble that
assign both samples to the same leaf node (one minus the Hamming distance between
their leaf indices); this is the random forest proximity of Breiman and Cutler.
A high score means the two samples follow the same decision paths through the
ensemble, making them naturally comparable for explanation purposes.
"""

import logging
import warnings
from typing import Optional, Union

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.metrics import DistanceMetric

logger = logging.getLogger("leafsim")
logger.addHandler(logging.NullHandler())

# Functions that get leaf indices for the different models supported by LeafSim
# E.g. catboost models use calc_leaf_indexes() while sklearn models use apply().
# Models are matched by class name anywhere in their MRO, so subclasses are supported too.
LEAF_INDEX_FUNC = {
    "CatBoostRegressor": "calc_leaf_indexes",
    "CatBoostClassifier": "calc_leaf_indexes",
    "ExtraTreesRegressor": "apply",
    "ExtraTreesClassifier": "apply",
    "GradientBoostingRegressor": "apply",
    "GradientBoostingClassifier": "apply",
    "LGBMRegressor": "predict",
    "LGBMClassifier": "predict",
    "RandomForestRegressor": "apply",
    "RandomForestClassifier": "apply",
    "XGBRegressor": "apply",
    "XGBClassifier": "apply",
}
# Default parameters to use for the models supported by LeafSim
LEAF_INDEX_DEFAULT_PARAMS = {
    "CatBoostRegressor": {
        "ntree_start": 0,
        "ntree_end": 0,
        "thread_count": -1,
        "verbose": False,
    },
    "CatBoostClassifier": {
        "ntree_start": 0,
        "ntree_end": 0,
        "thread_count": -1,
        "verbose": False,
    },
    "LGBMRegressor": {"pred_leaf": True},
    "LGBMClassifier": {"pred_leaf": True},
}

SUPPORTED_MODELS = sorted(list(LEAF_INDEX_FUNC.keys()))

# Maximum number of (test, train) pairs scored at once. Each batch holds a few 8-byte
# arrays of this size, so peak working memory stays around 100 MB.
_BATCH_ELEMENTS = 2**22


def _supported_base_name(model) -> Optional[str]:
    """Return the first class name in the model's MRO that LeafSim supports."""
    for cls in type(model).__mro__:
        if cls.__name__ in LEAF_INDEX_FUNC:
            return cls.__name__
    return None


class LeafSim:
    """LeafSim class."""

    def __init__(self, model, index_func_params: Optional[dict] = None):
        """
        Initialise the LeafSim instance.

        This defines the ML model that we want to explain.
        It also specifies the function to identify what
        observations fall into which leaves.

        :param model: Tree-based ensemble model, one from LEAF_INDEX_FUNC.keys()
        :param index_func_params: Parameters passed onto the leaf indexing function
        """
        # Set model
        self.model = model
        self.model_name = str(self.model.__class__.__name__)

        # Get leaf indexing function
        base_name = _supported_base_name(self.model)
        if base_name is None:
            # If providing a model that is not supported by LeafSim out of the box
            # This new model needs to have an attribute "get_leaf_indices"
            try:
                index_func = self.model.get_leaf_indices
            except AttributeError:
                supported = "\n".join(SUPPORTED_MODELS)
                error_msg = (
                    f"Provide one of the following currently supported models:\n\n"
                    f"{supported}\n\n"
                    f"or provide a custom model instance with a get_leaf_indices attribute.\n"
                    f"This must be a function that returns leaf indices as a matrix of shape"
                    f" [n_samples, n_predictors]."
                )
                raise TypeError(error_msg)
        else:
            index_func = getattr(self.model, LEAF_INDEX_FUNC[base_name])
        self.index_func = index_func

        # Get leaf indexing function parameters
        # Copy, so that changing one instance's params affects neither the
        # module-level defaults nor the dict the caller passed in
        if index_func_params is None:
            self.index_func_params = dict(LEAF_INDEX_DEFAULT_PARAMS.get(base_name, {}))
        else:
            self.index_func_params = dict(index_func_params)

    def get_leaf_indices(self, X: np.ndarray, params: Optional[dict] = None):
        """
        Get the indices of leaves for every observation in the feature matrix X.

        The function gets one index for every observation and tree in the ensemble model.

        :param X: feature matrix
        :param params:
            Parameters passed onto the function that gets the indices, for this call only.
            Defaults to the instance's index_func_params.
            Supported values depend on the model one wishes to generate explanations for.
        :return leaf_indices: Indices of the leaves in the shape of (X.shape[0], # trees)
        """
        if params is None:
            params = self.index_func_params

        # Get a matrix with each row containing the leaf indices
        # across all trees for a given instance
        leaf_indices = np.asarray(self.index_func(X, **params))

        # Some models (e.g. multiclass GradientBoosting) return one tree per class
        # and iteration as a 3D array: flatten so that every tree is a column
        return leaf_indices.reshape(leaf_indices.shape[0], -1)

    def fit(self, X_train: np.ndarray, params: Optional[dict] = None) -> "LeafSim":
        """
        Compute and cache the leaf indices of the training data.

        Call this once, then call explain() as often as needed without
        recomputing the training leaf indices.

        :param X_train: Data the model to explain was trained on.
        :param params: Parameters for the leaf indexing function, for this call only.
        :return self: The fitted LeafSim instance.
        """
        logger.info("Getting leaf indices of samples in training data")
        self.train_leaf_indices_ = self.get_leaf_indices(X_train, params)
        return self

    def _check_fitted(self) -> np.ndarray:
        try:
            return self.train_leaf_indices_
        except AttributeError:
            raise NotFittedError("Call fit(X_train) before explaining predictions.") from None

    def explain(
        self,
        X_to_explain: np.ndarray,
        top_n: int = 10,
        params: Optional[dict] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Identify the training examples most similar to each row of X_to_explain.

        Similarities are computed in batches of test rows, so memory stays bounded
        regardless of the number of rows to explain, and only the top_n per row are
        sorted. Ties are broken by the lowest training index.

        :param X_to_explain: Data points one wishes to generate explanations for.
        :param top_n: The number of explanations to provide per data point.
        :param params: Parameters for the leaf indexing function, for this call only.
        :return top_n_ids: Integer location for the observations in X_train that are among
                           the top_n, most similar first. Shape: (n_to_explain, top_n).
        :return top_n_similarity: The corresponding similarity, i.e. the fraction of trees
                                  in which the observation in top_n_ids and the observation
                                  one wishes to generate an explanation for share a leaf.
        """
        train_leaf_indices = self._check_fitted()
        n_train, n_trees = train_leaf_indices.shape
        if not 1 <= top_n <= n_train:
            raise ValueError(
                f"top_n ({top_n}) must be between 1 and the number of training samples ({n_train})"
            )
        logger.info("Getting leaf indices of samples in test data")
        test_leaf_indices = self.get_leaf_indices(X_to_explain, params)
        logger.info(
            f"Identifying top {top_n} most similar training data points for each test data point"
        )
        n_test = test_leaf_indices.shape[0]
        top_n_ids = np.empty((n_test, top_n), dtype=np.intp)
        top_n_mismatches = np.empty((n_test, top_n), dtype=np.int64)
        train_idx = np.arange(n_train, dtype=np.int64)
        for start, stop in self._batches(n_test, n_train):
            mismatches = self._mismatches(test_leaf_indices[start:stop], train_leaf_indices)
            # A unique integer key that orders by mismatches, then by training index,
            # so selecting the top_n is exact and deterministic under ties
            key = mismatches * n_train + train_idx
            ids = np.argpartition(key, top_n - 1, axis=1)[:, :top_n]
            top_keys = np.take_along_axis(key, ids, axis=1)
            ids = np.take_along_axis(ids, np.argsort(top_keys, axis=1), axis=1)
            top_n_ids[start:stop] = ids
            top_n_mismatches[start:stop] = np.take_along_axis(mismatches, ids, axis=1)

        top_n_similarity = 1 - top_n_mismatches / n_trees
        return top_n_ids, top_n_similarity

    def pairwise_similarities(
        self, X_to_explain: np.ndarray, params: Optional[dict] = None
    ) -> np.ndarray:
        """
        Compute the similarity between every row of X_to_explain and every training row.

        This builds a dense matrix of shape (n_to_explain, n_train): prefer explain()
        when only the most similar training rows are needed.

        :param X_to_explain: Data points one wishes to compare to the training data.
        :param params: Parameters for the leaf indexing function, for this call only.
        :return similarities: Fraction of trees in which each pair shares a leaf.
        """
        train_leaf_indices = self._check_fitted()
        test_leaf_indices = self.get_leaf_indices(X_to_explain, params)
        logger.info("Measuring similarities between every train and test data point")
        return (
            1
            - self._mismatches(test_leaf_indices, train_leaf_indices)
            / (train_leaf_indices.shape[1])
        )

    @staticmethod
    def _batches(n_test: int, n_train: int):
        """Yield (start, stop) ranges of test rows whose score matrix fits a fixed budget."""
        batch_size = max(1, _BATCH_ELEMENTS // max(n_train, 1))
        for start in range(0, n_test, batch_size):
            yield start, min(start + batch_size, n_test)

    @staticmethod
    def _mismatches(test_leaf_indices: np.ndarray, train_leaf_indices: np.ndarray) -> np.ndarray:
        """Count, for every pair of rows, the trees in which they land in different leaves."""
        distances = DistanceMetric.get_metric("hamming").pairwise(
            X=test_leaf_indices, Y=train_leaf_indices
        )
        return np.rint(distances * train_leaf_indices.shape[1]).astype(np.int64)

    def generate_explanations(
        self,
        X_train: np.ndarray,
        X_to_explain: np.ndarray,
        params: Optional[dict] = None,
        top_n: int = 10,
        return_all_similarities: bool = False,
    ) -> Union[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Identify the training examples most similar to each row of X_to_explain.

        Shorthand for fit(X_train, params).explain(X_to_explain, top_n, params).
        To explain several batches against the same training data, call fit() once
        and explain() per batch instead, so the training leaf indices are reused.

        :param X_train: Data the model to explain was trained on.
        :param X_to_explain: Data points one wished to generate explanations for.
        :param params: Parameters for the function that returns
                       indices of leaves for every observation.
                       See official documentation of the LEAF_INDEX_FUNC functions
                       supported by LeafSim.
        :param top_n: The number of explanations to provide.
                      By default, provide the 10 closest matches to every
                      observation in X_to_explain.
        :param return_all_similarities: Deprecated, use pairwise_similarities() instead.
                                        Whether to also return the similarities for
                                        all training observations.
        :return top_n_ids: See explain().
        :return top_n_similarity: See explain().
        """
        top_n_ids, top_n_similarity = self.fit(X_train, params).explain(
            X_to_explain, top_n, params
        )
        if return_all_similarities:
            warnings.warn(
                "return_all_similarities is deprecated and will be removed in a future"
                " release; use fit(X_train).pairwise_similarities(X_to_explain) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            similarities = self.pairwise_similarities(X_to_explain, params)
            return top_n_ids, top_n_similarity, similarities
        return top_n_ids, top_n_similarity
