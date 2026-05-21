"""
Upload ProtoPathway checkpoints, preprocessed cohort data, and the curated
pathway graph to the Hugging Face Hub.

Prereqs:
    pip install huggingface_hub
    huggingface-cli login   # paste a token with WRITE access from
                            # https://huggingface.co/settings/tokens

Layout on the Hub:
    AmayaGS/ProtoPathway/
        README.md                              (model card)
        pathways/pathways_base_*.pkl           (curated pathway graph)
        raw_inputs/                            (raw files for re-running preprocessing)
            Reactome/
            Hallmark/
            {cohort}/                          (rna_clean.csv, clinical, splits)
        cohorts/{cohort}/                      (preprocessed + checkpoints)
            gene_expression.csv
            bipartite_graph.pt
            labels.csv
            data_splits.pkl
            checkpoints/best_fold_{0..4}.pt

Usage:
    # Push the model card and the shared pathway file
    python utils/upload_to_hf.py \
        --repo-id AmayaGS/ProtoPathway \
        --pathways /path/to/processed/pathways/pathways_base_d5_g3-200_j100.pkl

    # Push preprocessed data for every cohort
    python utils/upload_to_hf.py \
        --repo-id AmayaGS/ProtoPathway \
        --processed-root /path/to/processed

    # Push checkpoints for every cohort
    python utils/upload_to_hf.py \
        --repo-id AmayaGS/ProtoPathway \
        --checkpoints-root /path/to/results/checkpoints

    # Or just one cohort at a time
    python utils/upload_to_hf.py \
        --repo-id AmayaGS/ProtoPathway \
        --processed-root /path/to/processed \
        --checkpoints-root /path/to/results/checkpoints \
        --cohorts TCGA-BLCA
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import create_repo, upload_file, upload_folder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DEFAULT_COHORTS = ["TCGA-BRCA", "TCGA-BLCA", "TCGA-COADREAD", "TCGA-HNSC", "TCGA-STAD"]

# Files expected under {processed_root}/{cohort}/
PREPROCESSED_FILES = [
    "gene_expression.csv",
    "bipartite_graph.pt",
    "labels.csv",
    "data_splits.pkl",
]

# Shared reference files expected under {raw_data_root}/Reactome/ and /Hallmark/
REFERENCE_DATA_FILES = {
    "Reactome": [
        "ReactomePathways.gmt",
        "ReactomePathwaysRelation.txt",
        "ReactomePathways.txt",
    ],
    "Hallmark": [
        "hallmarks_signatures.csv",
    ],
}


MODEL_CARD = """\
---
license: mit
tags:
  - medical-imaging
  - computational-pathology
  - survival-analysis
  - multimodal
  - tcga
datasets:
  - TCGA
library_name: pytorch
---

# ProtoPathway

Pretrained checkpoints, preprocessed cohort data, and the curated pathway
graph for **ProtoPathway**, an interpretable-by-design multimodal framework
for cancer survival prediction.

See the [code repository](https://github.com/AmayaGS/ProtoPathway) for usage,
training, and evaluation instructions.

## Layout

```
pathways/pathways_base_*.pkl           curated Reactome + Hallmark pathway graph
raw_inputs/                            raw files for re-running preprocessing from scratch
    Reactome/                          Reactome hierarchy files (GMT, relations, names)
    Hallmark/                          MSigDB Hallmark gene sets
    {cohort}/                          rna_clean.csv, clinical CSV, SurvPath splits
cohorts/{cohort}/                      preprocessed cohort data and trained models
    gene_expression.csv                preprocessed expression matrix
    bipartite_graph.pt                 cohort-specific gene-pathway graph
    labels.csv                         survival times, events, and bins
    data_splits.pkl                    5-fold CV splits (SurvPath-compatible)
    checkpoints/best_fold_{0..4}.pt    trained model weights
```

## Cohorts

Five TCGA cohorts: BRCA (N=714), BLCA (N=359), COADREAD (N=227),
HNSC (N=392), STAD (N=318). Gene expression is the preprocessed
SurvPath release. WSI patch features (UNI2-h) are not redistributed
here and should be obtained from the
[Mahmood Lab](https://huggingface.co/MahmoodLab/UNI2-h) directly.

## Quick load

```python
from huggingface_hub import snapshot_download

# Everything for one cohort plus the shared pathway file
snapshot_download(
    repo_id="AmayaGS/ProtoPathway",
    local_dir="./assets",
    allow_patterns=["cohorts/TCGA-BLCA/*", "pathways/*"],
)
```

## Citation

```bibtex
@article{protopathway2026,
  title   = {ProtoPathway: Interpretable Multimodal Cancer Survival Prediction
             via Prototype-Pathway Cross-Modal Attention},
  author  = {...},
  year    = {2026}
}
```
"""


def create_repo_if_needed(repo_id: str, private: bool = False):
    """Create the model repo on the Hub. Safe to call repeatedly."""
    logger.info(f"Creating (or reusing) repo: {repo_id}")
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )


def push_model_card(repo_id: str):
    """Write the model card README.md to the Hub root."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(MODEL_CARD)
        tmp_path = f.name

    upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Update model card",
    )
    Path(tmp_path).unlink()
    logger.info("  Uploaded README.md (model card)")


def push_pathways(repo_id: str, pathways_pkl: Path):
    """Upload the shared curated pathway file."""
    if not pathways_pkl.exists():
        logger.warning(f"  Pathways file not found: {pathways_pkl}")
        return
    target = f"pathways/{pathways_pkl.name}"
    upload_file(
        path_or_fileobj=str(pathways_pkl),
        path_in_repo=target,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload curated pathways: {pathways_pkl.name}",
    )
    logger.info(f"  Uploaded {target}")


def push_cohort_preprocessed(repo_id: str, processed_root: Path, cohorts):
    """
    Stage and upload the four preprocessed files per cohort:
    gene_expression.csv, bipartite_graph.pt, labels.csv, data_splits.pkl.

    One commit per cohort.
    """
    for cohort in cohorts:
        cohort_src = processed_root / cohort
        if not cohort_src.exists():
            logger.warning(f"  Preprocessed dir not found: {cohort_src}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / cohort
            staging.mkdir(parents=True)

            uploaded = []
            for filename in PREPROCESSED_FILES:
                src = cohort_src / filename
                if src.exists():
                    shutil.copy2(src, staging / filename)
                    uploaded.append(filename)
                else:
                    logger.warning(f"  Missing for {cohort}: {filename}")

            if not uploaded:
                logger.warning(f"  No preprocessed files for {cohort}, skipping")
                continue

            upload_folder(
                folder_path=str(staging),
                path_in_repo=f"cohorts/{cohort}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload preprocessed data for {cohort}",
            )
            logger.info(f"  Uploaded {len(uploaded)} preprocessed files for {cohort}: {uploaded}")


def push_cohort_checkpoints(repo_id: str, checkpoints_root: Path, cohorts):
    """
    Stage and upload best_fold_*.pt for each cohort.
    Lands in cohorts/{cohort}/checkpoints/.

    Expects layout: {checkpoints_root}/{cohort}/best_fold_*.pt
    """
    for cohort in cohorts:
        cohort_dir = checkpoints_root / cohort
        if not cohort_dir.exists():
            logger.warning(f"  Checkpoint dir not found: {cohort_dir}")
            continue

        ckpts = sorted(cohort_dir.glob("best_fold_*.pt"))
        if not ckpts:
            logger.warning(f"  No checkpoints in {cohort_dir}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / "checkpoints"
            staging.mkdir(parents=True)
            for ckpt in ckpts:
                shutil.copy2(ckpt, staging / ckpt.name)

            upload_folder(
                folder_path=str(staging),
                path_in_repo=f"cohorts/{cohort}/checkpoints",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {len(ckpts)} fold checkpoints for {cohort}",
            )
            logger.info(f"  Uploaded {len(ckpts)} checkpoints for {cohort}")


# Auxiliary files copied from an experiment dir alongside its checkpoints
EXPERIMENT_METADATA_FILES = ["config.yaml"]


def push_experiment_dir(repo_id: str, experiment_dir: Path, cohort: str):
    """
    Upload best_fold_*.pt from a flat experiment directory.

    Also picks up config.yaml, summary.json, and training_log.txt
    if they exist, so the uploaded checkpoint folder fully documents
    how the model was trained.

    Lands in cohorts/{cohort}/checkpoints/.
    """
    if not experiment_dir.exists():
        logger.warning(f"  Experiment dir not found: {experiment_dir}")
        return

    ckpts = sorted(experiment_dir.glob("best_fold_*.pt"))
    if not ckpts:
        logger.warning(f"  No best_fold_*.pt files in {experiment_dir}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "checkpoints"
        staging.mkdir(parents=True)

        for ckpt in ckpts:
            shutil.copy2(ckpt, staging / ckpt.name)

        extras = []
        for filename in EXPERIMENT_METADATA_FILES:
            src = experiment_dir / filename
            if src.exists():
                shutil.copy2(src, staging / filename)
                extras.append(filename)

        upload_folder(
            folder_path=str(staging),
            path_in_repo=f"cohorts/{cohort}/checkpoints",
            repo_id=repo_id,
            repo_type="model",
            commit_message=(
                f"Upload {len(ckpts)} fold checkpoints for {cohort} "
                f"(from {experiment_dir.name})"
            ),
        )
        logger.info(
            f"  Uploaded {len(ckpts)} checkpoints for {cohort} "
            f"from {experiment_dir.name}"
            + (f" (+ metadata: {extras})" if extras else "")
        )


def push_reference_data(repo_id: str, raw_data_root: Path):
    """
    Upload shared Reactome and Hallmark files to raw_inputs/ on the Hub.

    Expects layout:
        {raw_data_root}/Reactome/{ReactomePathways.gmt, ReactomePathwaysRelation.txt, ReactomePathways.txt}
        {raw_data_root}/Hallmark/hallmarks_signatures.csv

    Lands at:
        raw_inputs/Reactome/...
        raw_inputs/Hallmark/...
    """
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        uploaded = {}

        for subdir, files in REFERENCE_DATA_FILES.items():
            src_dir = raw_data_root / subdir
            if not src_dir.exists():
                logger.warning(f"  Reference dir not found: {src_dir}")
                continue

            dst_dir = staging / subdir
            dst_dir.mkdir(parents=True)

            uploaded[subdir] = []
            for filename in files:
                src = src_dir / filename
                if src.exists():
                    shutil.copy2(src, dst_dir / filename)
                    uploaded[subdir].append(filename)
                else:
                    logger.warning(f"  Missing reference file: {src}")

        if not uploaded:
            logger.warning("  No reference data files found, skipping")
            return

        upload_folder(
            folder_path=str(staging),
            path_in_repo="raw_inputs",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Reactome and Hallmark reference data",
        )
        for subdir, files in uploaded.items():
            logger.info(f"  Uploaded raw_inputs/{subdir}/: {files}")


def push_cohort_raw_inputs(repo_id: str, raw_data_root: Path, cohorts):
    """
    Upload per-cohort raw inputs (gene expression, clinical, SurvPath splits)
    to raw_inputs/{cohort}/ on the Hub.

    For each cohort, expects under {raw_data_root}/{cohort}/:
        rna_clean.csv          gene expression matrix
        {cohort}.csv           clinical CSV (named to match the cohort)
        splits/splits_*.csv    SurvPath predefined 5-fold splits

    One commit per cohort.
    """
    for cohort in cohorts:
        cohort_src = raw_data_root / cohort
        if not cohort_src.exists():
            logger.warning(f"  Cohort raw dir not found: {cohort_src}")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp) / cohort
            staging.mkdir(parents=True)

            uploaded = []

            rna = cohort_src / "rna_clean.csv"
            if rna.exists():
                shutil.copy2(rna, staging / "rna_clean.csv")
                uploaded.append("rna_clean.csv")
            else:
                logger.warning(f"  Missing for {cohort}: rna_clean.csv")

            clinical = cohort_src / f"{cohort}.csv"
            if clinical.exists():
                shutil.copy2(clinical, staging / f"{cohort}.csv")
                uploaded.append(f"{cohort}.csv")
            else:
                logger.warning(f"  Missing for {cohort}: {cohort}.csv (clinical)")

            splits_dir = cohort_src / "splits"
            if splits_dir.exists():
                staging_splits = staging / "splits"
                staging_splits.mkdir()
                for splits_file in sorted(splits_dir.glob("splits_*.csv")):
                    shutil.copy2(splits_file, staging_splits / splits_file.name)
                    uploaded.append(f"splits/{splits_file.name}")
            else:
                logger.warning(f"  Missing for {cohort}: splits/")

            if not uploaded:
                logger.warning(f"  No raw files for {cohort}, skipping")
                continue

            upload_folder(
                folder_path=str(staging),
                path_in_repo=f"raw_inputs/{cohort}",
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload raw inputs for {cohort}",
            )
            logger.info(f"  Uploaded {len(uploaded)} raw files for {cohort}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload ProtoPathway assets to the Hugging Face Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo-id",
        default="AmayaGS/ProtoPathway",
        help="Hub repo id (default: AmayaGS/ProtoPathway).",
    )
    parser.add_argument("--create-repo", action="store_true", help="Create the repo if it does not exist.")
    parser.add_argument("--private", action="store_true", help="Mark the repo as private (only with --create-repo).")
    parser.add_argument(
        "--pathways",
        type=Path,
        default=None,
        help="Path to the curated pathways pkl. Goes to pathways/ on the Hub.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=None,
        help="Root containing {cohort}/ subdirs with gene_expression.csv, "
             "bipartite_graph.pt, labels.csv, data_splits.pkl.",
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=None,
        help="Root containing Reactome/, Hallmark/, and per-cohort raw subdirs "
             "(rna_clean.csv, {cohort}.csv, splits/). When given, uploads "
             "Reactome + Hallmark to raw_inputs/ and per-cohort raw inputs "
             "(scoped by --cohorts) to raw_inputs/{cohort}/.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=None,
        help="Root containing {cohort}/best_fold_*.pt files.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Flat experiment directory containing best_fold_*.pt directly "
             "(plus optional config.yaml, summary.json, training_log.txt). "
             "Requires exactly one cohort name via --cohorts.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=DEFAULT_COHORTS,
        help=f"Cohort names to process (default: {DEFAULT_COHORTS}).",
    )
    parser.add_argument("--no-model-card", action="store_true", help="Skip uploading the model card.")

    args = parser.parse_args()

    if args.create_repo:
        create_repo_if_needed(args.repo_id, private=args.private)

    if not args.no_model_card:
        push_model_card(args.repo_id)

    if args.pathways:
        push_pathways(args.repo_id, args.pathways)

    if args.processed_root:
        push_cohort_preprocessed(args.repo_id, args.processed_root, args.cohorts)

    if args.raw_data_root:
        push_reference_data(args.repo_id, args.raw_data_root)
        push_cohort_raw_inputs(args.repo_id, args.raw_data_root, args.cohorts)

    if args.checkpoints_root:
        push_cohort_checkpoints(args.repo_id, args.checkpoints_root, args.cohorts)

    if args.experiment_dir:
        if len(args.cohorts) != 1:
            parser.error(
                "--experiment-dir requires exactly one cohort name via --cohorts "
                f"(got {len(args.cohorts)}: {args.cohorts})."
            )
        push_experiment_dir(args.repo_id, args.experiment_dir, args.cohorts[0])

    logger.info(f"\nDone. View your repo at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()