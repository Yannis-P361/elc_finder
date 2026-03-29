"""Tests for atlas validation."""

import nibabel as nib
import numpy as np
import pytest

from electrode_coverage.atlas.validation import Severity, validate_mni_atlas


def _make_nifti(shape=(182, 218, 182), affine=None, data=None, sform_code=4):
    """Create a test NIfTI image."""
    if affine is None:
        # Standard MNI152 1mm affine
        affine = np.diag([-1, 1, 1, 1]).astype(float)
        affine[0, 3] = 90   # x offset
        affine[1, 3] = -126  # y offset
        affine[2, 3] = -72   # z offset
    if data is None:
        data = np.zeros(shape, dtype=np.uint8)
        # Put some non-zero voxels in the middle
        data[80:100, 100:120, 80:100] = 1
    img = nib.Nifti1Image(data, affine)
    img.header["sform_code"] = sform_code
    return img


class TestValidateMniAtlas:
    def test_valid_mni_atlas(self):
        img = _make_nifti(sform_code=4)
        messages = validate_mni_atlas(img)
        errors = [m for m in messages if m.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_empty_mask(self):
        data = np.zeros((182, 218, 182), dtype=np.uint8)
        img = _make_nifti(data=data)
        messages = validate_mni_atlas(img)
        errors = [m for m in messages if m.severity == Severity.ERROR]
        assert any("no non-zero" in m.message.lower() for m in errors)

    def test_4d_single_volume(self):
        data = np.zeros((182, 218, 182, 1), dtype=np.uint8)
        data[80:100, 100:120, 80:100, 0] = 1
        img = _make_nifti(data=data)
        messages = validate_mni_atlas(img)
        info = [m for m in messages if m.severity == Severity.INFO]
        assert any("4D" in m.message for m in info)

    def test_wrong_dimensions(self):
        data = np.zeros((10, 10, 10, 5), dtype=np.uint8)
        img = _make_nifti(data=data)
        messages = validate_mni_atlas(img)
        errors = [m for m in messages if m.severity == Severity.ERROR]
        assert any("3D" in m.message for m in errors)

    def test_scanner_space_warning(self):
        img = _make_nifti(sform_code=1)
        messages = validate_mni_atlas(img)
        warnings = [m for m in messages if m.severity == Severity.WARNING]
        assert any("scanner" in m.message.lower() for m in warnings)

    def test_labeled_atlas(self):
        data = np.zeros((182, 218, 182), dtype=np.uint8)
        data[80:90, 100:110, 80:90] = 1
        data[90:100, 110:120, 90:100] = 2
        data[100:110, 120:130, 100:110] = 3
        img = _make_nifti(data=data)
        messages = validate_mni_atlas(img)
        info = [m for m in messages if m.severity == Severity.INFO]
        assert any("label" in m.message.lower() for m in info)
