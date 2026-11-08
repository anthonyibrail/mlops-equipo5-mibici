"""Pruebas rápidas para la etapa de selección, sin acceso a archivos grandes."""

import numpy as np
import pytest

from mibici.modeling.tune import parameter_candidates, sample_index


def _params():
    return {
        "seed": 42,
        "modelos": {
            "hist_gbr": {
                "fijos": {"loss": "poisson", "max_iter": 100, "learning_rate": 0.05},
                "busqueda": {
                    "espacio": {
                        "learning_rate": [0.03, 0.05, 0.1],
                        "max_leaf_nodes": [15, 31, 63],
                    }
                },
            }
        },
    }


def test_parameter_candidates_include_original_and_are_reproducible():
    c1 = parameter_candidates(_params(), 4)
    c2 = parameter_candidates(_params(), 4)
    assert c1 == c2
    assert c1[0] == _params()["modelos"]["hist_gbr"]["fijos"]
    assert len(c1) >= 2
    assert len(c1) <= 5


def test_sampling_without_replacement_and_seed():
    idx = np.arange(1000)
    a = sample_index(idx, 35, np.random.default_rng(42))
    b = sample_index(idx, 35, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert len(np.unique(a)) == 35
    assert np.all(np.diff(a) > 0)


def test_sampling_small_fold_keeps_all_rows():
    idx = np.arange(5)
    np.testing.assert_array_equal(
        sample_index(idx, 15, np.random.default_rng(42)), idx
    )


def test_sampling_rejects_zero_rows():
    with pytest.raises(ValueError):
        sample_index(np.arange(10), 0, np.random.default_rng(42))
