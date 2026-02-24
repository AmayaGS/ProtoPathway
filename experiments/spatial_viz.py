"""
Spatial Visualization Usage Guide
==================================

This script demonstrates how to generate spatial prototype and pathway
overlay figures for selected patients. Run it after training + evaluation.

Prerequisites:
    1. Training complete (checkpoints saved)
    2. Evaluation complete (predictions + attention pickles saved)
    3. Visualization complete (per-fold rank analysis CSVs generated)
    4. preprocess_wsi re-run with coords patch (new-format .pt files)
    5. (Optional) Downloaded WSI .svs files for tissue backdrop

Directory structure assumed:
    experiments/COADREAD_protopath/
        best_model_fold_0.pt
        best_model_fold_1.pt
        ...
        config.yaml
        evaluation/
            predictions_fold_0.csv
            predictions_fold_1.csv
            ...
            attention_fold_0/attention_weights.pkl
            attention_fold_1/attention_weights.pkl
            ...
        figures/
            per_fold/
                fold_0/analysis/
                    crossmodal_proto_0_rank_analysis.csv
                    crossmodal_proto_1_rank_analysis.csv
                    ...

    processed/COADREAD/
        wsi_features_per_patient/
            TCGA-A6-2671.pt     # new format: dict with features + coords + slide_info
            ...

    wsi_slides/                  # optional, for tissue backdrop
        TCGA-A6-2671-01Z-00-DX1.svs
"""

import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# =====================================================================
# 0. CONFIGURE PATHS (edit these to match your setup)
# =====================================================================

# Experiment directory (where checkpoints + evaluation live)
EXPERIMENT_DIR = r'C:\Users\Amaya\Documents\PhD\ProtoPathway_results\experiments\TCGA-BLCA\protopath_gene+wsi_TCGA-BLCA_cross_attention_P16_lr_gene0.0001_lr_wsi0.0001_l21e-05_dr_gene0.5_dr_fusion0.5_hd128_tau10_s42_20260216_135917'

# Preprocessed WSI features directory (new format with coords)
WSI_FEATURES_DIR = r'C:\Users\Amaya\Documents\PhD\ProtoPathway_results\processed\TCGA-BLCA\wsi_features_per_patient'

# Optional: directory containing downloaded .svs files
# Set to None if you don't have WSI files — you'll get coordinate-based
# canvases instead of tissue backdrops
WSI_SLIDES_DIR = r'C:\Users\Amaya\Documents\PhD\Data\TGCA_data\TCGA-BLCA\slides'  # or None

# Which fold to use for prototype-level analysis
# (pick your best-performing fold, or the one whose biology makes most sense)
FOLD_IDX = 1

# Patch size at extraction magnification (256 for UNI-2h default)
PATCH_SIZE = 256

# Downsample factor for rendering (4 = quarter resolution, good balance
# of detail vs file size; use 1 for full resolution)
DOWNSAMPLE = 6

# Optional: a single pathway to render as a continuous "IHC" heatmap
# Set to None to skip, or use a pathway name from your analysis
SINGLE_PATHWAY = 'R-HSA-8857538'  # e.g., 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION'


# =====================================================================
# 1. LOAD DATA (pool predictions + attention across folds)
# =====================================================================

def load_experiment_data():
    """Load pooled predictions and attention data."""
    from utils.analysis.fold_aggregation import pool_cv_results

    eval_dir = os.path.join(EXPERIMENT_DIR, 'evaluation')

    logger.info(f"Loading evaluation data from {eval_dir}")
    predictions, attention_by_patient, metadata = pool_cv_results(
        eval_dir, risk_stratification='median'
    )

    pathway_names = metadata.get('pathway_names', [])
    gene_names = metadata.get('gene_names', [])

    logger.info(f"Loaded {len(predictions)} patients, {len(pathway_names)} pathways")
    logger.info(f"Risk groups: {predictions['risk_group'].value_counts().to_dict()}")

    return predictions, attention_by_patient, pathway_names


# =====================================================================
# 2. SELECT PATIENTS (choose interesting cases to visualize)
# =====================================================================

def select_patients(predictions, attention_by_patient, n_per_group=2):
    """
    Select clear high-risk and low-risk patients for visualization.

    Strategy: pick patients with the most extreme risk scores who also
    have attention data available (i.e., were in the validation set for
    some fold).

    Args:
        predictions: Pooled predictions DataFrame.
        attention_by_patient: Attention dict.
        n_per_group: Number of patients per risk group.

    Returns:
        Dict with 'high_risk' and 'low_risk' patient ID lists.
    """
    # Filter to patients with attention data
    available = predictions[
        predictions['patient_id'].isin(attention_by_patient.keys())
    ].copy()

    # Sort by risk score
    available = available.sort_values('risk_score')

    # Pick extremes
    low_risk = available.head(n_per_group)['patient_id'].tolist()
    high_risk = available.tail(n_per_group)['patient_id'].tolist()

    logger.info(f"Selected patients:")
    for pid in low_risk:
        score = available[available['patient_id'] == pid]['risk_score'].values[0]
        logger.info(f"  LOW  RISK: {pid} (score={score:.4f})")
    for pid in high_risk:
        score = available[available['patient_id'] == pid]['risk_score'].values[0]
        logger.info(f"  HIGH RISK: {pid} (score={score:.4f})")

    return {'low_risk': low_risk, 'high_risk': high_risk}


# =====================================================================
# 3. VERIFY DATA (check coords are available and patches align)
# =====================================================================

def verify_patient_data(patient_id, attention_by_patient):
    """
    Check that a patient's data is complete for spatial visualization.

    Verifies:
        - .pt file exists and has coords
        - Attention data has required signals
        - Patch counts match between .pt and attention data
    """
    import torch

    # Check .pt file
    pt_path = os.path.join(WSI_FEATURES_DIR, f'{patient_id}.pt')
    if not os.path.exists(pt_path):
        logger.error(f"  {patient_id}: missing .pt file at {pt_path}")
        return False

    data = torch.load(pt_path, weights_only=False)

    if isinstance(data, torch.Tensor):
        logger.error(f"  {patient_id}: old-format .pt (no coords). Re-run preprocess_wsi.")
        return False

    if 'coords' not in data:
        logger.error(f"  {patient_id}: .pt has no 'coords' key")
        return False

    n_patches_pt = data['features'].shape[0]
    n_coords = data['coords'].shape[0]
    has_slide_info = 'slide_info' in data

    logger.info(
        f"  {patient_id}: .pt OK — {n_patches_pt} patches, "
        f"{n_coords} coords, slide_info={'yes' if has_slide_info else 'no'}"
    )

    if n_patches_pt != n_coords:
        logger.error(f"  {patient_id}: feature/coord count mismatch!")
        return False

    # Check attention data
    attn = attention_by_patient.get(patient_id)
    if attn is None:
        logger.error(f"  {patient_id}: no attention data")
        return False

    pa = attn.get('patch_assignments', {})
    hard = pa.get('hard_assignments', pa.get('assignments'))
    if hard is None:
        logger.error(f"  {patient_id}: no hard_assignments in attention")
        return False

    n_patches_attn = len(np.asarray(hard))
    if n_patches_attn != n_patches_pt:
        logger.warning(
            f"  {patient_id}: PATCH MISMATCH — "
            f".pt has {n_patches_pt}, attention has {n_patches_attn}. "
            f"Check max_slides / slide_type_filter settings."
        )
        return False

    has_cross_modal = 'cross_modal_attention' in attn
    has_gate = 'gate_weights' in pa
    has_fusion = 'fusion_gate_weights' in attn

    logger.info(
        f"  {patient_id}: attention OK — {n_patches_attn} patches, "
        f"cross_modal={'yes' if has_cross_modal else 'no'}, "
        f"gate={'yes' if has_gate else 'no'}, "
        f"fusion_gate={'yes' if has_fusion else 'no'}"
    )

    return True


# =====================================================================
# 4. GENERATE VISUALIZATIONS
# =====================================================================

def generate_for_patient(patient_id, risk_group, attention_by_patient, pathway_names):
    """Generate spatial visualization for a single patient."""
    from utils.visualization.spatial_heatmaps import generate_patient_spatial_viz

    fold_analysis_dir = os.path.join(
        EXPERIMENT_DIR, 'figures', 'per_fold', f'fold_{FOLD_IDX}', 'analysis'
    )

    if not os.path.exists(fold_analysis_dir):
        logger.error(
            f"Per-fold analysis not found at {fold_analysis_dir}. "
            f"Run visualize first to generate rank analysis CSVs."
        )
        return

    output_dir = os.path.join(
        EXPERIMENT_DIR, 'figures', 'spatial', patient_id
    )

    generate_patient_spatial_viz(
        patient_id=patient_id,
        attention_data=attention_by_patient[patient_id],
        pathway_names=pathway_names,
        fold_analysis_dir=fold_analysis_dir,
        output_dir=output_dir,
        risk_group=risk_group,
        wsi_features_dir=WSI_FEATURES_DIR,
        wsi_dir=WSI_SLIDES_DIR,
        patch_size=PATCH_SIZE,
        downsample=DOWNSAMPLE,
        single_pathway_name=SINGLE_PATHWAY,
        rank_transform=True
    )


# =====================================================================
# 5. MAIN
# =====================================================================

def main():
    logger.info("=" * 60)
    logger.info("ProtoPathway Spatial Visualization")
    logger.info("=" * 60)

    # Check prerequisites
    fold_analysis_dir = os.path.join(
        EXPERIMENT_DIR, 'figures', 'per_fold', f'fold_{FOLD_IDX}', 'analysis'
    )
    if not os.path.exists(fold_analysis_dir):
        logger.error(
            f"\nPer-fold analysis directory not found:\n  {fold_analysis_dir}\n\n"
            f"You need to run the visualization pipeline first:\n"
            f"  python main.py visualize --eval-dir {EXPERIMENT_DIR}/evaluation\n\n"
            f"This generates the per-fold rank analysis CSVs that the\n"
            f"spatial overlay uses to select risk-aware pathways."
        )
        return

    # Load data
    predictions, attention_by_patient, pathway_names = load_experiment_data()

    # Select patients
    selected = select_patients(predictions, attention_by_patient, n_per_group=2)

    all_patients = selected['low_risk'] + selected['high_risk']

    # Verify data for each patient
    logger.info("\nVerifying patient data...")
    verified = []
    for pid in all_patients:
        if verify_patient_data(pid, attention_by_patient):
            verified.append(pid)

    if not verified:
        logger.error("No patients passed verification. Check paths above.")
        return

    logger.info(f"\n{len(verified)}/{len(all_patients)} patients verified")

    # Generate visualizations
    risk_map = dict(zip(predictions['patient_id'], predictions['risk_group']))

    for pid in verified:
        risk_group = risk_map[pid]
        logger.info(f"\n{'─'*40}")
        logger.info(f"Generating: {pid} ({risk_group})")
        logger.info(f"{'─'*40}")

        generate_for_patient(pid, risk_group, attention_by_patient, pathway_names)

    # Summary
    output_base = os.path.join(EXPERIMENT_DIR, 'figures', 'spatial')
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Outputs in: {output_base}")
    logger.info(f"{'='*60}")

    for pid in verified:
        patient_dir = os.path.join(output_base, pid)
        if os.path.exists(patient_dir):
            files = [f for f in os.listdir(patient_dir) if f.endswith('.pdf')]
            for f in files:
                logger.info(f"  {pid}/{f}")


if __name__ == '__main__':
    predictions, attention_by_patient, pathway_names = load_experiment_data()

    patient_id = 'TCGA-FD-A3B4'  # your specific patient

    verify_patient_data(patient_id, attention_by_patient)

    risk_map = dict(zip(predictions['patient_id'], predictions['risk_group']))
    generate_for_patient(patient_id, risk_map[patient_id], attention_by_patient, pathway_names)