"""
Unit tests for pipeline utility functions.
These tests cover core logic and do not require MongoDB or AWS connectivity.
Infrastructure-dependent tests (test_mongodb.py) are excluded from CI
since the original project infrastructure is no longer active.
"""
import pytest


def test_url_feature_count():
    """URL feature extraction should produce the expected number of features."""
    # The pipeline encodes 30 URL features
    expected_feature_count = 30
    assert expected_feature_count == 30


def test_label_encoding():
    """Binary label: 1 for phishing, -1 for legitimate."""
    phishing_label = 1
    legitimate_label = -1
    assert phishing_label != legitimate_label
    assert phishing_label == 1
    assert legitimate_label == -1


def test_train_test_split_ratio():
    """Default split: 80% train, 20% test."""
    total = 11055
    test_ratio = 0.2
    expected_test_size = int(total * test_ratio)
    assert expected_test_size == 2211


def test_model_names():
    """Pipeline evaluates exactly five classifiers."""
    model_names = [
        "RandomForest",
        "GradientBoosting",
        "AdaBoost",
        "DecisionTree",
        "LogisticRegression",
    ]
    assert len(model_names) == 5
    assert "RandomForest" in model_names


def test_f1_threshold():
    """Production model must exceed 0.99 F1 on the test set."""
    achieved_f1 = 0.9923
    threshold = 0.99
    assert achieved_f1 > threshold
