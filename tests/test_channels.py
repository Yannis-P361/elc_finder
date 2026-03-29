"""Tests for channel annotation parsing."""

import pandas as pd
import pytest

from electrode_coverage.io.channels import (
    extract_annotations,
    has_annotations,
    has_soz_annotations,
)


def _make_channels_df(**kwargs):
    """Helper to create a channels DataFrame."""
    defaults = {"name": ["ch1", "ch2", "ch3"]}
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


class TestExtractAnnotations:
    def test_soz_detection(self):
        df = _make_channels_df(
            status_description=["seizure onset zone", "normal", "soz"],
        )
        result = extract_annotations(df)
        assert result["is_soz"].iloc[0] == True
        assert result["is_soz"].iloc[1] == False
        assert result["is_soz"].iloc[2] == True

    def test_irritative_zone(self):
        df = _make_channels_df(
            status_description=["irritative zone", "normal", "irz"],
        )
        result = extract_annotations(df)
        assert result["is_irritative_zone"].iloc[0] == True
        assert result["is_irritative_zone"].iloc[1] == False
        assert result["is_irritative_zone"].iloc[2] == True

    def test_resected(self):
        df = _make_channels_df(
            description=["resected", "normal", "in resection zone"],
        )
        result = extract_annotations(df)
        assert result["is_resected"].iloc[0] == True
        assert result["is_resected"].iloc[1] == False
        assert result["is_resected"].iloc[2] == True

    def test_bad_channel(self):
        df = _make_channels_df(
            status=["good", "bad", "good"],
        )
        result = extract_annotations(df)
        assert result["is_bad"].iloc[0] == False
        assert result["is_bad"].iloc[1] == True
        assert result["is_bad"].iloc[2] == False

    def test_no_annotation_columns(self):
        df = _make_channels_df()
        result = extract_annotations(df)
        assert result["is_soz"].iloc[0] == False
        assert result["clinical_annotation"].iloc[0] == None

    def test_case_insensitive(self):
        df = _make_channels_df(
            status_description=["SOZ", "Seizure Onset Zone", "IRRITATIVE"],
        )
        result = extract_annotations(df)
        assert result["is_soz"].iloc[0] == True
        assert result["is_soz"].iloc[1] == True
        assert result["is_irritative_zone"].iloc[2] == True


class TestHasAnnotations:
    def test_with_annotations(self):
        df = _make_channels_df(
            status_description=["soz", "normal", ""],
        )
        assert has_annotations(df) == True

    def test_without_annotations(self):
        df = _make_channels_df()
        assert has_annotations(df) == False

    def test_empty_annotations(self):
        df = _make_channels_df(
            status_description=["", "", ""],
        )
        assert has_annotations(df) == False


class TestHasSozAnnotations:
    def test_with_soz(self):
        df = _make_channels_df(
            status_description=["soz", "normal", ""],
        )
        assert has_soz_annotations(df) == True

    def test_without_soz(self):
        df = _make_channels_df(
            status_description=["irritative", "normal", ""],
        )
        assert has_soz_annotations(df) == False
