"""
Analysis utilities for ProtoPathway interpretability.

Modules:
    fold_aggregation: Pool predictions and attention data across CV folds
    importance_analyzer: Rank-based statistical analysis of model importance signals
"""

from .fold_aggregation import pool_cv_results
from .importance_analyzer import ImportanceAnalyzer, run_importance_analysis