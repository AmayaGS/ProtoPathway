"""
Spatial Visualization for ProtoPathway.

Generates spatial prototype and pathway overlay figures for selected
patients. Run after training + evaluation + visualization pipeline.

Prerequisites:
    1. Training complete (checkpoints saved)
    2. Evaluation complete (predictions + attention pickles saved)
    3. Visualization complete (per-fold rank analysis CSVs generated)
    4. preprocess_wsi re-run with coords patch (new-format .pt files)
    5. (Optional) Downloaded WSI .svs files for tissue backdrop

Directory structure assumed:
    experiments/COADREAD_protopath/
        best_model_fold_0.pt
        ...
        config.yaml
        evaluation/
            predictions_fold_0.csv
            attention_fold_0/attention_weights.pkl
            ...
        figures/
            per_fold/
                fold_0/analysis/
                    crossmodal_proto_0_rank_analysis.csv
                    ...

    processed/COADREAD/
        wsi_features_per_patient/
            TCGA-A6-2671.pt

    wsi_slides/                  # optional
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
WSI_SLIDES_DIR = r'C:\Users\Amaya\Documents\PhD\Data\TGCA_data\TCGA-BLCA\slides'  # or None

# Which fold to use for prototype-level analysis
FOLD_IDX = 1

# Patch size at extraction magnification (256 for UNI-2h default)
PATCH_SIZE = 256

# Downsample factor for rendering
DOWNSAMPLE = 6

# Optional: a single pathway to render as a continuous "IHC" heatmap
SINGLE_PATHWAY = 'R-HSA-8857538'


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
# 2. SELECT PATIENTS
# =====================================================================

def select_patients(predictions, attention_by_patient, n_per_group=2):
    """
    Select clear high-risk and low-risk patients for visualization.

    Picks patients with the most extreme risk scores who also have
    attention data available.
    """
    available = predictions[
        predictions['patient_id'].isin(attention_by_patient.keys())
    ].copy()

    available = available.sort_values('risk_score')

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
# 3. VERIFY DATA
# =====================================================================

def verify_patient_data(patient_id, attention_by_patient):
    """Check that a patient's data is complete for spatial visualization."""
    import torch

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
            f".pt has {n_patches_pt}, attention has {n_patches_attn}."
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
        rank_transform=True,
    )


# =====================================================================
# 5. PROTOTYPE PANELS (cohort-level)
# =====================================================================

def generate_prototype_panels(attention_by_patient):
    """
    Generate cohort-level prototype importance and exemplar figures.

    Produces:
        - WSI gate (Signal E) importance bars: overall, high-risk, low-risk
        - Fusion gate (Signal H) importance bars: overall, high-risk, low-risk
        - Top-by-risk-level comparison figure
        - Cohort exemplar patches for overall, high-risk, low-risk
    """
    from utils.visualization.prototype_panels import (
        plot_prototype_importance,
        plot_cohort_prototype_exemplars,
    )

    proto_output = os.path.join(
        EXPERIMENT_DIR, 'figures', 'spatial', 'prototype_panels'
    )

    # Prototype importance bar charts — both gates
    logger.info("  Generating prototype importance bars (E + H)...")
    try:
        plot_prototype_importance(
            attention_by_patient=attention_by_patient,
            output_dir=proto_output,
            top_k=5,
            dpi=300,
        )
    except Exception as e:
        logger.error(f"  Failed prototype importance: {e}")

    # Cohort-level exemplar patches
    logger.info("  Generating cohort exemplar patches...")
    try:
        plot_cohort_prototype_exemplars(
            attention_by_patient=attention_by_patient,
            wsi_features_dir=WSI_FEATURES_DIR,
            output_dir=proto_output,
            top_k_protos=5,
            n_patches_per_proto=8,
            wsi_dir=WSI_SLIDES_DIR,
            downsample=DOWNSAMPLE,
            patch_size=PATCH_SIZE,
        )
    except Exception as e:
        logger.error(f"  Failed cohort exemplars: {e}")


# =====================================================================
# 6. MAIN
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

    # Generate per-patient spatial visualizations
    risk_map = dict(zip(predictions['patient_id'], predictions['risk_group']))

    for pid in verified:
        risk_group = risk_map[pid]
        logger.info(f"\n{'─'*40}")
        logger.info(f"Generating: {pid} ({risk_group})")
        logger.info(f"{'─'*40}")
        generate_for_patient(pid, risk_group, attention_by_patient, pathway_names)

    # ── Cohort-level prototype panels ────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("Generating cohort-level prototype panels...")
    logger.info("=" * 60)

    generate_prototype_panels(attention_by_patient)

    # Summary
    output_base = os.path.join(EXPERIMENT_DIR, 'figures', 'spatial')
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Outputs in: {output_base}")
    logger.info(f"{'='*60}")

    for pid in verified:
        patient_dir = os.path.join(output_base, pid)
        if os.path.exists(patient_dir):
            files = [f for f in os.listdir(patient_dir)
                     if f.endswith(('.pdf', '.svg', '.png'))]
            for f in sorted(files):
                logger.info(f"  {pid}/{f}")

    proto_dir = os.path.join(output_base, 'prototype_panels')
    if os.path.exists(proto_dir):
        logger.info(f"\n  Prototype panels:")
        for root, dirs, files in os.walk(proto_dir):
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), output_base)
                logger.info(f"  {rel}")


def run_single_patient(patient_id):
    """
    Run spatial viz + prototype panels for a single patient.

    Use this when you have one WSI and want all outputs.
    """
    logger.info("=" * 60)
    logger.info(f"ProtoPathway Spatial Viz — Single Patient: {patient_id}")
    logger.info("=" * 60)

    # Load data
    predictions, attention_by_patient, pathway_names = load_experiment_data()

    # Verify
    if not verify_patient_data(patient_id, attention_by_patient):
        logger.error(f"Verification failed for {patient_id}")
        return

    # Risk group
    risk_map = dict(zip(predictions['patient_id'], predictions['risk_group']))
    risk_group = risk_map.get(patient_id, 'Unknown')
    logger.info(f"Risk group: {risk_group}")

    # Generate spatial overlays
    logger.info(f"\n{'─'*40}")
    logger.info(f"Generating spatial overlays...")
    logger.info(f"{'─'*40}")
    generate_for_patient(patient_id, risk_group, attention_by_patient, pathway_names)

    # Generate prototype panels (uses all patients for cohort context)
    logger.info(f"\n{'─'*40}")
    logger.info(f"Generating prototype panels (cohort-level)...")
    logger.info(f"{'─'*40}")
    generate_prototype_panels(attention_by_patient)

    # Summary
    output_base = os.path.join(EXPERIMENT_DIR, 'figures', 'spatial')
    logger.info(f"\n{'='*60}")
    logger.info(f"Done! Outputs in: {output_base}")
    logger.info(f"{'='*60}")

    patient_dir = os.path.join(output_base, patient_id)
    if os.path.exists(patient_dir):
        files = [f for f in os.listdir(patient_dir)
                 if f.endswith(('.pdf', '.svg', '.png'))]
        for f in sorted(files):
            logger.info(f"  {patient_id}/{f}")

    proto_dir = os.path.join(output_base, 'prototype_panels')
    if os.path.exists(proto_dir):
        logger.info(f"\n  Prototype panels:")
        for root, dirs, files in os.walk(proto_dir):
            for f in sorted(files):
                rel = os.path.relpath(os.path.join(root, f), output_base)
                logger.info(f"  {rel}")


if __name__ == '__main__':
    # ── Choose one: ──────────────────────────────────────────────────
    #
    # Option A: Single patient (when you have one WSI downloaded)
    run_single_patient('TCGA-FD-A3B4')
    #
    # Option B: Full cohort mode (auto-selects extreme patients)
    # main()