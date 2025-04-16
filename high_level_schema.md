[Input]
Whole Slide Image (WSI) + Gene Expression Vector (per sample)

↓
[1. Prototype Discovery Module]
- Learnable superpixels from WSI via contrastive or clustering-based prototype learning
- Output: N_p prototype vectors (patch-level latent features)

↓
[2. Biological Hypergraph Module]
- Nodes: Genes (with expression features)
- Hyperedges: Biological pathways (sets of genes)
- Use Hypergraph Attention Network (HGAN or similar)
- Output: Pathway embeddings (N_pathway × d), attention scores gene↔pathway

↓
[3. Latent Harmonization Module]
- Project both proto-superpixels and pathways to same latent space:
    - Proto: W_v(p_i), Pathway: W_p(h_j)
- Output: Latent-aligned proto and pathway embeddings

↓
[4. Cross-Attention Module]
- For each proto-superpixel, attend over pathways:
    - α_ij = Attention(p_i, pathway_j)
- Output: attention matrix (N_proto × N_pathways)

↓
[5. Fusion + Classification Head]
- Fuse attention-weighted pathway info into proto features
- Global aggregation → classification (e.g. slide-level label)

↓
[6. Interpretability]
- Overlay heatmaps of pathway attention on WSI
- Gene-level saliency from gene→pathway attention and/or gradient
- Per-superpixel pathway ranking
