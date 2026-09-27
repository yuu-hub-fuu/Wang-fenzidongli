import numpy as np
import pytest
import scipy.linalg

from atlas_bench.metrics import core


def test_sqrtm_matches_scipy_for_psd_products():
    rng = np.random.RandomState(0)
    A = rng.randn(5, 3, 3)
    B = rng.randn(5, 3, 3)
    S1 = A @ A.transpose(0, 2, 1) + 0.1 * np.eye(3)
    S2 = B @ B.transpose(0, 2, 1) + 0.1 * np.eye(3)
    ours = core.sqrtm(S1 @ S2)
    ref = np.stack([scipy.linalg.sqrtm(a @ b) for a, b in zip(S1, S2)])
    np.testing.assert_allclose(ours, np.real(ref), atol=1e-8)


def test_gaussian_w2_zero_for_identical_and_translation_only_for_shift():
    rng = np.random.RandomState(1)
    x = rng.randn(400, 7, 3) * 0.05
    emd_mean, emd_var = core.gaussian_w2_terms(x, x)
    np.testing.assert_allclose(emd_mean, 0, atol=1e-10)
    np.testing.assert_allclose(emd_var, 0, atol=1e-5)

    shift = np.array([0.3, 0.0, 0.4])  # nm -> 5 A
    emd_mean, emd_var = core.gaussian_w2_terms(x, x + shift)
    np.testing.assert_allclose(emd_mean, 5.0, atol=1e-8)
    np.testing.assert_allclose(emd_var, 0, atol=1e-5)
    d = core.rmwd_decomposition(emd_mean, emd_var)
    assert d["rmwd"] == pytest.approx(np.hypot(d["rmwd_trans"], d["rmwd_var"]))


def test_gaussian_w2_variance_term_isotropic_closed_form():
    # N(0, s1^2 I) vs N(0, s2^2 I) in 3D: W2^2 = 3 (s1 - s2)^2
    rng = np.random.RandomState(2)
    s1, s2 = 0.1, 0.25
    x = rng.randn(200000, 1, 3) * s1
    y = rng.randn(200000, 1, 3) * s2
    _, emd_var = core.gaussian_w2_terms(x, y)
    assert emd_var[0] == pytest.approx(np.sqrt(3) * (s2 - s1) * 10, rel=0.02)


def test_empirical_wasserstein():
    rng = np.random.RandomState(3)
    a = rng.randn(50, 2)
    dist = np.linalg.norm(a[:, None] - a[None], axis=-1)
    assert core.get_wasserstein(dist) == pytest.approx(0.0)
    perm = rng.permutation(50)
    dist = np.linalg.norm(a[:, None] - (a[perm] + 1.0)[None], axis=-1)
    assert core.get_wasserstein(dist) == pytest.approx(np.sqrt(2.0))


def test_sasa_mi_properties():
    rng = np.random.RandomState(4)
    x = rng.rand(5000, 3) > 0.5
    x[:, 2] = x[:, 0]  # residue 2 copies residue 0
    mi = core.sasa_mi(x)
    assert np.allclose(np.diag(mi), 0)
    assert mi[0, 2] == pytest.approx(np.log(2), rel=0.01)
    assert mi[0, 1] < 1e-3
    np.testing.assert_allclose(mi, mi.T)


def test_condense_sidechain_sasas():
    names = ["N", "CA", "C", "O", "CB", "N", "CA", "C", "O", "CB", "OG", "OXT"]
    res = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    sasa = np.arange(12, dtype=float)[None]
    out = core.condense_sidechain_sasas(sasa, names, res)
    np.testing.assert_allclose(out, [[4, 9 + 10]])


def test_rmsd_and_contacts():
    rng = np.random.RandomState(5)
    x = rng.randn(4, 10, 3)
    d = core.get_rmsds(x, x, broadcast=True)
    np.testing.assert_allclose(np.diag(d), 0, atol=1e-12)
    probs = core.contact_prob(x * 0.1, chunk=3)
    ref = (np.linalg.norm(x[:, None] - x[:, :, None], axis=-1) * 0.1 < core.CONTACT_CUTOFF_NM).mean(0)
    np.testing.assert_allclose(probs, ref)


def test_jaccard():
    a = np.array([1, 1, 0, 0], bool)
    b = np.array([1, 0, 1, 0], bool)
    assert core.jaccard(a, b) == pytest.approx(1 / 3)
    assert np.isnan(core.jaccard(np.zeros(3, bool), np.zeros(3, bool)))
