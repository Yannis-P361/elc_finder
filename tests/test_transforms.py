"""Tests for coordinate space transforms."""

import numpy as np
import pytest

from electrode_coverage.coords.transforms import (
    LANCASTER_TAL2MNI,
    flag_suspicious_units,
    transform_to_mni,
)


class TestFlagSuspiciousUnits:
    def test_normal_coords(self):
        coords = np.array([[30.0, -20.0, 50.0]])
        assert flag_suspicious_units(coords) is False

    def test_meter_scale(self):
        coords = np.array([[0.03, -0.02, 0.05]])
        assert flag_suspicious_units(coords) is True

    def test_unrealistically_large(self):
        coords = np.array([[500.0, -400.0, 300.0]])
        assert flag_suspicious_units(coords) is True

    def test_boundary_normal(self):
        coords = np.array([[1.0, 0.0, 0.0]])
        assert flag_suspicious_units(coords) is False

    def test_boundary_large(self):
        coords = np.array([[300.0, 0.0, 0.0]])
        assert flag_suspicious_units(coords) is False  # exactly 300 is ok


class TestTransformToMni:
    def test_mni_native(self):
        coords = np.array([[10.0, -20.0, 30.0]])
        result, status, suspicious = transform_to_mni(coords, "MNI152NLin2009cAsym")
        assert status == "mni_native"
        assert suspicious is False
        np.testing.assert_array_equal(result, coords)

    def test_acpc_approximate(self):
        coords = np.array([[10.0, -20.0, 30.0]])
        result, status, suspicious = transform_to_mni(coords, "ACPC")
        assert status == "acpc_approximate"
        np.testing.assert_array_equal(result, coords)

    def test_talairach_transform(self):
        coords = np.array([[10.0, -20.0, 30.0], [0.0, 0.0, 0.0]])
        result, status, suspicious = transform_to_mni(coords, "Talairach")
        assert status == "mni_from_talairach"
        assert suspicious is False
        # Origin should stay at origin
        np.testing.assert_array_almost_equal(result[1], [0.0, 0.0, 0.0])
        # Non-origin should change
        assert not np.allclose(result[0], coords[0])

    def test_talairach_matches_manual(self):
        coords = np.array([[10.0, -20.0, 30.0]])
        result, _, _ = transform_to_mni(coords, "Talairach")
        ones = np.ones((1, 1))
        manual = (np.hstack([coords, ones]) @ LANCASTER_TAL2MNI.T)[:, :3]
        np.testing.assert_array_almost_equal(result, manual)

    def test_surface_space(self):
        coords = np.array([[1.0, 2.0, 3.0]])
        _, status, _ = transform_to_mni(coords, "fsaverage")
        assert status == "surface_space"

    def test_native_space(self):
        coords = np.array([[1.0, 2.0, 3.0]])
        _, status, _ = transform_to_mni(coords, "ScanRAS")
        assert status == "native_space"

    def test_unknown_space(self):
        coords = np.array([[1.0, 2.0, 3.0]])
        _, status, _ = transform_to_mni(coords, "SomethingWeird")
        assert status == "no_transform"

    def test_suspicious_units_in_mni(self):
        coords = np.array([[0.01, -0.02, 0.03]])
        _, status, suspicious = transform_to_mni(coords, "MNI152Lin")
        assert status == "mni_native"
        assert suspicious is True
