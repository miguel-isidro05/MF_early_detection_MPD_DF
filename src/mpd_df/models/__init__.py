"""Mandatory benchmark models."""

from .classical import make_psd_svm, make_random_forest
from .deep import EEGNet, MSCNNCAM

__all__ = ["EEGNet", "MSCNNCAM", "make_psd_svm", "make_random_forest"]

