"""Tests for leafsim.leafsim."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split

import leafsim.leafsim
from leafsim import SUPPORTED_MODELS, LeafSim


@pytest.fixture
def iris():
    data = load_iris()
    X, y = data.data, data.target
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train


@pytest.fixture
def fitted_classifier(iris):
    X_train, _, y_train = iris
    return RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)


@pytest.fixture
def fitted_regressor(iris):
    X_train, _, y_train = iris
    return RandomForestRegressor(n_estimators=10, random_state=42).fit(X_train, y_train)


# --- Construction ---


def test_construction_classifier(fitted_classifier):
    ls = LeafSim(fitted_classifier)
    assert ls.model is fitted_classifier
    assert ls.model_name == "RandomForestClassifier"
    assert ls.index_func_params == {}


def test_construction_regressor(fitted_regressor):
    ls = LeafSim(fitted_regressor)
    assert ls.model_name == "RandomForestRegressor"


def test_construction_custom_index_func_params_are_forwarded():
    class TruncatingModel:
        def get_leaf_indices(self, X, n_trees=3):
            return np.asarray(X)[:, :n_trees]

    X_train = np.array([[0, 0, 0], [0, 1, 1]])
    X_test = np.array([[0, 1, 1]])
    ls = LeafSim(TruncatingModel(), index_func_params={"n_trees": 1})
    _, sims = ls.generate_explanations(X_train, X_test, top_n=2)
    # Only the first "tree" is compared, so both training rows match fully
    np.testing.assert_allclose(sims, [[1.0, 1.0]])


def test_construction_unsupported_model_raises():
    class UnsupportedModel:
        pass

    with pytest.raises(TypeError, match="currently supported models"):
        LeafSim(UnsupportedModel())


def test_construction_custom_model_with_get_leaf_indices(iris, fitted_classifier):
    X_train, X_test, _ = iris

    class CustomModel:
        def get_leaf_indices(self, X, **kwargs):
            return fitted_classifier.apply(X)

    ls = LeafSim(CustomModel())
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=5)
    assert ids.shape == (X_test.shape[0], 5)


def test_supported_models_exported():
    assert "RandomForestClassifier" in SUPPORTED_MODELS
    assert "RandomForestRegressor" in SUPPORTED_MODELS


# --- generate_explanations ---


def test_output_shapes(fitted_classifier, iris):
    X_train, X_test, _ = iris
    top_n = 5
    ls = LeafSim(fitted_classifier)
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=top_n)
    assert ids.shape == (X_test.shape[0], top_n)
    assert sims.shape == (X_test.shape[0], top_n)


def test_similarity_range(fitted_classifier, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_classifier)
    _, sims = ls.generate_explanations(X_train, X_test, top_n=10)
    assert np.all(sims >= 0)
    assert np.all(sims <= 1)


def test_return_all_similarities_is_deprecated(fitted_classifier, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_classifier)
    with pytest.warns(DeprecationWarning, match="pairwise_similarities"):
        result = ls.generate_explanations(X_train, X_test, top_n=5, return_all_similarities=True)
    assert len(result) == 3
    ids, top_sims, all_sims = result
    assert all_sims.shape == (X_test.shape[0], X_train.shape[0])


def test_top_n_exceeds_train_size_raises(fitted_classifier, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_classifier)
    with pytest.raises(ValueError, match="top_n"):
        ls.generate_explanations(X_train, X_test, top_n=X_train.shape[0] + 1)


def test_regressor_output_shapes(fitted_regressor, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_regressor)
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=3)
    assert ids.shape == (X_test.shape[0], 3)
    assert sims.shape == (X_test.shape[0], 3)


# --- XGBoost ---


def test_xgb_regressor(iris):
    xgb = pytest.importorskip("xgboost")
    X_train, X_test, y_train = iris
    model = xgb.XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    ls = LeafSim(model)
    assert ls.model_name == "XGBRegressor"
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=5)
    assert ids.shape == (X_test.shape[0], 5)
    assert np.all(sims >= 0) and np.all(sims <= 1)


def test_xgb_classifier(iris):
    xgb = pytest.importorskip("xgboost")
    X_train, X_test, y_train = iris
    model = xgb.XGBClassifier(n_estimators=10, random_state=42, eval_metric="mlogloss")
    model.fit(X_train, y_train)
    ls = LeafSim(model)
    assert ls.model_name == "XGBClassifier"
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=5)
    assert ids.shape == (X_test.shape[0], 5)
    assert np.all(sims >= 0) and np.all(sims <= 1)


# --- Correctness ---


class LeavesAsFeatures:
    """Custom model whose leaf indices are the input rows themselves."""

    def get_leaf_indices(self, X, **kwargs):
        return np.asarray(X)


def test_neighbours_match_hand_computed_similarities():
    X_train = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1]])
    X_test = np.array([[0, 0, 0]])
    ls = LeafSim(LeavesAsFeatures())
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=4)
    np.testing.assert_array_equal(ids, [[0, 1, 3, 2]])
    np.testing.assert_allclose(sims, [[1, 2 / 3, 1 / 3, 0]])


def test_neighbours_match_brute_force_proximity(fitted_classifier, iris):
    X_train, X_test, _ = iris
    train_leaves = fitted_classifier.apply(X_train)
    test_leaves = fitted_classifier.apply(X_test)
    expected = (test_leaves[:, None, :] == train_leaves[None, :, :]).mean(axis=2)

    ls = LeafSim(fitted_classifier)
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=10)

    row_idx = np.arange(X_test.shape[0])[:, None]
    np.testing.assert_allclose(sims, expected[row_idx, ids])
    np.testing.assert_allclose(sims, -np.sort(-expected, axis=1)[:, :10])


def test_training_sample_is_its_own_nearest_neighbour(fitted_classifier, iris):
    X_train, _, _ = iris
    ls = LeafSim(fitted_classifier)
    _, sims = ls.generate_explanations(X_train, X_train[:5], top_n=1)
    np.testing.assert_allclose(sims[:, 0], 1.0)


# --- CatBoost ---


def test_catboost_regressor(iris):
    catboost = pytest.importorskip("catboost")
    X_train, X_test, y_train = iris
    model = catboost.CatBoostRegressor(
        iterations=10, random_seed=42, verbose=False, allow_writing_files=False
    )
    model.fit(X_train, y_train)
    ls = LeafSim(model)
    ids, sims = ls.generate_explanations(X_train, X_test, top_n=5)

    leaves_train = model.calc_leaf_indexes(X_train)
    leaves_test = model.calc_leaf_indexes(X_test)
    expected = (leaves_test[:, None, :] == leaves_train[None, :, :]).mean(axis=2)
    row_idx = np.arange(X_test.shape[0])[:, None]
    np.testing.assert_allclose(sims, expected[row_idx, ids])
    np.testing.assert_allclose(sims, -np.sort(-expected, axis=1)[:, :5])


def test_catboost_classifier(iris):
    catboost = pytest.importorskip("catboost")
    X_train, X_test, y_train = iris
    model = catboost.CatBoostClassifier(
        iterations=10, random_seed=42, verbose=False, allow_writing_files=False
    )
    model.fit(X_train, y_train)
    ids, sims = LeafSim(model).generate_explanations(X_train, X_test, top_n=5)
    assert ids.shape == (X_test.shape[0], 5)
    assert np.all(sims >= 0) and np.all(sims <= 1)


# --- Model detection ---


def _assert_matches_reference(model, leaves_train, leaves_test, X_train, X_test, top_n=5):
    leaves_train = np.asarray(leaves_train).reshape(len(X_train), -1)
    leaves_test = np.asarray(leaves_test).reshape(len(X_test), -1)
    expected = (leaves_test[:, None, :] == leaves_train[None, :, :]).mean(axis=2)
    ids, sims = LeafSim(model).generate_explanations(X_train, X_test, top_n=top_n)
    row_idx = np.arange(len(X_test))[:, None]
    np.testing.assert_allclose(sims, expected[row_idx, ids])
    np.testing.assert_allclose(sims, -np.sort(-expected, axis=1)[:, :top_n])


def test_subclass_of_supported_model_is_accepted(iris):
    X_train, X_test, y_train = iris

    class MyForest(RandomForestClassifier):
        pass

    model = MyForest(n_estimators=10, random_state=42).fit(X_train, y_train)
    assert LeafSim(model).model_name == "MyForest"
    _assert_matches_reference(model, model.apply(X_train), model.apply(X_test), X_train, X_test)


@pytest.mark.parametrize(
    "model_cls",
    [
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    ],
)
def test_other_sklearn_tree_ensembles(iris, model_cls):
    X_train, X_test, y_train = iris
    model = model_cls(n_estimators=10, random_state=42).fit(X_train, y_train)
    _assert_matches_reference(model, model.apply(X_train), model.apply(X_test), X_train, X_test)


@pytest.mark.parametrize("model_name", ["LGBMClassifier", "LGBMRegressor"])
def test_lightgbm(iris, model_name):
    lightgbm = pytest.importorskip("lightgbm")
    X_train, X_test, y_train = iris
    model = getattr(lightgbm, model_name)(n_estimators=10, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    _assert_matches_reference(
        model,
        model.predict(X_train, pred_leaf=True),
        model.predict(X_test, pred_leaf=True),
        X_train,
        X_test,
    )


def test_model_without_leaf_indices_is_rejected(iris):
    X_train, _, y_train = iris
    model = HistGradientBoostingClassifier(max_iter=5).fit(X_train, y_train)
    with pytest.raises(TypeError, match="currently supported models"):
        LeafSim(model)


# --- API behaviour ---


class RecordingModel:
    """Custom model that records the keyword arguments of every call."""

    def __init__(self):
        self.calls = []

    def get_leaf_indices(self, X, **kwargs):
        self.calls.append(kwargs)
        return np.asarray(X)


def test_per_call_params_do_not_replace_instance_defaults():
    model = RecordingModel()
    ls = LeafSim(model, index_func_params={"a": 1})
    X = np.zeros((3, 2))
    ls.generate_explanations(X, X, params={"a": 2}, top_n=1)
    model.calls.clear()
    ls.generate_explanations(X, X, top_n=1)
    assert model.calls == [{"a": 1}, {"a": 1}]
    assert ls.index_func_params == {"a": 1}


def test_default_params_are_not_shared_between_instances(iris):
    catboost = pytest.importorskip("catboost")
    X_train, _, y_train = iris
    model = catboost.CatBoostRegressor(iterations=2, verbose=False, allow_writing_files=False)
    model.fit(X_train, y_train)
    first, second = LeafSim(model), LeafSim(model)
    first.index_func_params["ntree_end"] = 1
    assert second.index_func_params["ntree_end"] == 0


def test_ties_are_broken_by_lowest_training_index():
    rng = np.random.default_rng(0)
    X_train = rng.integers(0, 2, size=(500, 3))
    X_test = np.array([[0, 1, 0]])
    ids, sims = LeafSim(LeavesAsFeatures()).generate_explanations(X_train, X_test, top_n=500)
    # Order: similarity descending, then training index ascending
    expected = np.lexsort((np.arange(500), -sims[0][np.argsort(ids[0])]))
    np.testing.assert_array_equal(ids[0], expected)


# --- fit / explain ---


def test_fit_caches_training_leaves():
    model = RecordingModel()
    X_train = np.array([[0, 0], [0, 1], [1, 1]])
    ls = LeafSim(model).fit(X_train)
    ls.explain(X_train[:1], top_n=2)
    ls.explain(X_train[1:], top_n=2)
    # One call for the training data, one per explain call
    assert len(model.calls) == 3


def test_explain_matches_generate_explanations(fitted_classifier, iris):
    X_train, X_test, _ = iris
    ls = LeafSim(fitted_classifier)
    expected_ids, expected_sims = ls.generate_explanations(X_train, X_test, top_n=7)
    ids, sims = ls.fit(X_train).explain(X_test, top_n=7)
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_allclose(sims, expected_sims)


def test_explain_before_fit_raises(fitted_classifier, iris):
    _, X_test, _ = iris
    with pytest.raises(NotFittedError):
        LeafSim(fitted_classifier).explain(X_test)


def test_pairwise_similarities_match_brute_force(fitted_classifier, iris):
    X_train, X_test, _ = iris
    train_leaves = fitted_classifier.apply(X_train)
    test_leaves = fitted_classifier.apply(X_test)
    expected = (test_leaves[:, None, :] == train_leaves[None, :, :]).mean(axis=2)
    sims = LeafSim(fitted_classifier).fit(X_train).pairwise_similarities(X_test)
    np.testing.assert_allclose(sims, expected)


@pytest.mark.parametrize("top_n", [0, -1])
def test_non_positive_top_n_raises(fitted_classifier, iris, top_n):
    X_train, X_test, _ = iris
    with pytest.raises(ValueError, match="top_n"):
        LeafSim(fitted_classifier).fit(X_train).explain(X_test, top_n=top_n)


def test_batched_scoring_matches_brute_force(monkeypatch, fitted_classifier, iris):
    # Force tiny batches so that results must be stitched across many of them
    monkeypatch.setattr(leafsim.leafsim, "_BATCH_ELEMENTS", 250)
    X_train, X_test, _ = iris
    train_leaves = fitted_classifier.apply(X_train)
    test_leaves = fitted_classifier.apply(X_test)
    expected = (test_leaves[:, None, :] == train_leaves[None, :, :]).mean(axis=2)
    expected_ids = np.lexsort(
        (np.broadcast_to(np.arange(len(X_train)), expected.shape), -expected)
    )

    ids, sims = LeafSim(fitted_classifier).fit(X_train).explain(X_test, top_n=10)
    np.testing.assert_array_equal(ids, expected_ids[:, :10])
    np.testing.assert_allclose(sims, np.take_along_axis(expected, ids, axis=1))
