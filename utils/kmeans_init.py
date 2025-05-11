
# utils/kmeans_init.py

import numpy as np
import random
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
# from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch


def sample_embeddings(emb_dict, max_samples: int = 100_000, seed: int = 42) -> torch.Tensor:
    """
    Return ≤ `max_samples` patch-level embeddings from a patient-embedding dict.
    Sampling is proportional to each patient’s share of patches.
    """
    random.seed(seed)
    np.random.seed(seed)

    # — Gather every patient tensor on CPU —
    tensors = [d[0]
               for d in emb_dict.values()
                ]

    # — How many patches exist in total? —
    patch_counts = torch.tensor([t.shape[0] for t in tensors])
    total_patches = int(patch_counts.sum())

    # — Decide how many to keep from each patient —
    if total_patches <= max_samples:                    # small dataset → keep everything
        keep_counts = patch_counts
    else:                                               # large dataset → scale down
        ratio = max_samples / total_patches
        keep_counts = torch.clamp((patch_counts * ratio).long(), min=1)

    # — Actually sample —
    sampled = []
    for t, k in zip(tensors, keep_counts.tolist()):
        idx = np.random.choice(len(t), size=k, replace=False)
        sampled.append(t[idx])

    out = torch.cat(sampled)

    # A final trim (rarely needed because of clamp-to-min-1 rounding)
    if len(out) > max_samples:
        out = out[np.random.choice(len(out), max_samples, replace=False)]

    print(f"Sampled {len(out)} embeddings from {len(tensors)} patients")

    return out



def init_prototypes(X,
                    n_proto: int,
                    seed: int = 42,
                    cosine: bool = True,
                    centroid_path: str | Path | None = None) -> torch.Tensor:
    """
    Return `n_proto` centroids for a prototype network.

    * Draw ≤ `max_samples` patch embeddings (proportional per-patient).
    * Run (mini-batch) k-means in Euclidean or cosine space.
    * Optionally cache / reload the result.
    """
    # -------- optional cache ----------

    if centroid_path is not None:
        f = Path(centroid_path).expanduser()

    else:
        f = None

    if f is not None and f.exists():  # <- early-exit load
        print(f"Loading cached centroids from {f}")
        return torch.load(f, weights_only=True)

    # ---------- sample & cluster -------------
    if cosine:
        X = normalize(X)

    km = MiniBatchKMeans(n_clusters=n_proto, batch_size=1024,
                         random_state=seed, n_init=10).fit(X)
    C = km.cluster_centers_
    if cosine:
        C = normalize(C)

    C = torch.as_tensor(C, dtype=torch.float32)

    # ---------- save if desired --------------
    if f is not None:
        torch.save(C, f)
        print(f"Centroids saved to {f}")

    return C
