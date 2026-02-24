"""
Analysis utilities for ProtoPathway interpretability.

Modules:
    fold_aggregation: Pool predictions and attention data across CV folds
    importance_analyzer: Rank-based statistical analysis of model importance signals
"""

from .fold_aggregation import pool_cv_results
from .fold_stratified_analysis import (
    FoldStratifiedAnalyzer,
    run_fold_stratified_importance_analysis,
)
from .fold_stratified_analysis import SavedAnalyzerProxy, load_saved_analyzers