"""Tests del split congelado de desarrollo/evaluación."""

from __future__ import annotations

from sources.phase2.development_split import (
    ALL_SUBJECTS,
    DEV_SUBJECTS,
    TEST_SUBJECTS,
    contains_development_subjects,
    evaluation_subjects,
    development_subjects,
    is_development_subject,
    split_subjects,
    split_warning,
)


def test_split_is_disjoint_and_covers_the_corpus():
    assert set(DEV_SUBJECTS).isdisjoint(TEST_SUBJECTS)
    assert len(ALL_SUBJECTS) == 15
    assert "S12" not in ALL_SUBJECTS  # no existe en WESAD
    assert set(ALL_SUBJECTS) == set(DEV_SUBJECTS) | set(TEST_SUBJECTS)


def test_split_subjects_classifies_known_subjects():
    dev, test = split_subjects(["S2", "S5", "S17"])
    assert dev == ["S2"]
    assert test == ["S5", "S17"]
    assert development_subjects() == list(DEV_SUBJECTS)
    assert evaluation_subjects() == list(TEST_SUBJECTS)


def test_split_normalizes_case_and_spaces():
    dev, test = split_subjects([" s2 ", "s5"])
    assert dev == ["S2"]
    assert test == ["S5"]


def test_unknown_subjects_are_ignored():
    dev, test = split_subjects(["S99"])
    assert dev == [] and test == []


def test_split_warning_only_for_development_subjects():
    assert split_warning(["S5", "S6"]) is None
    warning = split_warning(["S5", "S3"])
    assert warning is not None
    assert "S3" in warning
    assert "DESARROLLO" in warning


def test_helpers():
    assert is_development_subject("s3") is True
    assert is_development_subject("S5") is False
    assert contains_development_subjects(["S5", "S2"]) is True
    assert contains_development_subjects(["S5", "S6"]) is False
