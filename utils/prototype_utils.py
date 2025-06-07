import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from collections import defaultdict


def generate_prototype_heatmap(patient_id, patch_assignments, patch_names, patch_coordinates,
                               extracted_patches_path, output_dir, fold, patch_size=224,
                               use_binning=True):
    """
    Generate prototype assignment heatmap for a patient

    Args:
        patient_id: Patient identifier
        patch_assignments: List of prototype assignments [18, 18, 22, ...]
        patch_names: List of patch filename tuples
        patch_coordinates: List of coordinate strings ['[row1, row2, col1, col2]', ...]
        extracted_patches_path: Path to extracted_patches.csv
        output_dir: Output directory for heatmaps
        fold: Fold identifier
        patch_size: Size of patches (default 224)
        use_binning: Whether to bin prototypes into groups (default True)
    """

    # Apply binning if requested
    if use_binning:
        binned_assignments = bin_prototypes(patch_assignments)
        assignments_to_use = binned_assignments
        assignment_type = "binned"
    else:
        assignments_to_use = patch_assignments
        assignment_type = "original"

    # Load extracted patches CSV to get file locations
    patches_df = pd.read_csv(extracted_patches_path)

    # Create mapping from patch name to file location
    patch_to_location = {}
    for _, row in patches_df.iterrows():
        patch_to_location[row['Patch_name']] = row['File_location']

    # Process patch data
    patch_data = []
    for i, (patch_name_tuple, coord_str, proto_id) in enumerate(
            zip(patch_names, patch_coordinates, assignments_to_use)):
        patch_name = patch_name_tuple[0]  # Extract from tuple

        # Parse coordinates
        coords = eval(coord_str)  # [row1, row2, col1, col2]
        row1, row2, col1, col2 = coords

        # Get file location
        file_location = patch_to_location.get(patch_name, None)
        if file_location is None:
            continue

        patch_data.append({
            'patch_name': patch_name,
            'row1': row1, 'row2': row2, 'col1': col1, 'col2': col2,
            'prototype_id': proto_id,
            'original_prototype_id': patch_assignments[i],
            'file_location': file_location
        })

    if not patch_data:
        print(f"No valid patches found for patient {patient_id}")
        return

    # Group by slide (extract slide name from patch name)
    slides = defaultdict(list)
    for patch in patch_data:
        slide_name = extract_slide_name(patch['patch_name'])
        slides[slide_name].append(patch)

    # Generate heatmap for each slide
    for slide_name, slide_patches in slides.items():
        create_prototype_slide_heatmap(patient_id, slide_name, slide_patches, output_dir, fold,
                                       patch_size, use_binning, assignment_type)


def extract_slide_name(patch_name):
    """Extract slide name from patch filename"""
    # Remove the coordinate part to get slide name
    parts = patch_name.split('_row1=')
    return parts[0] if len(parts) > 1 else patch_name


def create_prototype_slide_heatmap(patient_id, slide_name, patches, output_dir, fold, patch_size,
                                   use_binning, assignment_type):
    """Create prototype heatmap for a single slide"""

    # Calculate canvas dimensions
    max_row = max(p['row2'] for p in patches)
    max_col = max(p['col2'] for p in patches)

    canvas = np.zeros((max_row + patch_size, max_col + patch_size, 3), dtype=np.uint8)
    prototype_map = np.full((max_row + patch_size, max_col + patch_size), -1, dtype=np.int32)

    # Get unique prototypes and create consistent colormap
    unique_prototypes = sorted(set(p['prototype_id'] for p in patches))

    if use_binning:
        # Use consistent bin colors
        bin_colors = get_consistent_colors()
        prototype_to_color = {proto_id: bin_colors[proto_id] for proto_id in unique_prototypes}
    else:
        # Use tab20 but make it consistent by using prototype ID as index
        max_proto_id = max(unique_prototypes)
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, max_proto_id + 1)))
        prototype_to_color = {proto_id: colors[proto_id % 20] for proto_id in unique_prototypes}

    # Load patches and fill canvas
    valid_patches = []
    for patch in patches:
        if not os.path.exists(patch['file_location']):
            continue

        try:
            # Load patch image
            patch_img = np.array(Image.open(patch['file_location']))
            if len(patch_img.shape) == 3 and patch_img.shape[2] == 4:
                patch_img = patch_img[:, :, :3]

            # Place on canvas
            r1, r2, c1, c2 = patch['row1'], patch['row2'], patch['col1'], patch['col2']
            canvas[r1:r2, c1:c2] = patch_img
            prototype_map[r1:r2, c1:c2] = patch['prototype_id']

            valid_patches.append(patch)

        except Exception as e:
            print(f"Error loading patch {patch['patch_name']}: {e}")
            continue

    if not valid_patches:
        print(f"No valid patches loaded for slide {slide_name}")
        return

    # Create prototype overlay
    prototype_overlay = np.zeros((prototype_map.shape[0], prototype_map.shape[1], 3))
    for proto_id in unique_prototypes:
        mask = prototype_map == proto_id
        color = mcolors.to_rgb(prototype_to_color[proto_id]) if isinstance(prototype_to_color[proto_id], str) else \
        prototype_to_color[proto_id][:3]
        prototype_overlay[mask] = color

    # Create the plot
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[4, 1], hspace=0.1, wspace=0.02)

    # Original slide
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Original Slide', size=16, pad=10)
    ax1.imshow(canvas)
    ax1.axis('off')

    # Prototype heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    title_suffix = " (Binned)" if use_binning else ""
    ax2.set_title(f'Prototype Assignment Heatmap{title_suffix}', size=16, pad=10)
    ax2.imshow(canvas)
    ax2.imshow(prototype_overlay, alpha=0.6)
    ax2.axis('off')

    # Prototype legend and sample patches
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # Create legend with sample patches
    create_prototype_legend(ax_legend, valid_patches, prototype_to_color, unique_prototypes, use_binning)

    title_suffix = " (Binned Prototypes)" if use_binning else " (Original Prototypes)"
    plt.suptitle(f'Patient {patient_id} - {slide_name}{title_suffix}', size=18)

    # Save
    output_folder = os.path.join(output_dir, patient_id)
    os.makedirs(output_folder, exist_ok=True)
    filename_suffix = "_binned" if use_binning else "_original"
    output_path = os.path.join(output_folder, f"{slide_name}_prototype_heatmap{filename_suffix}_fold_{fold}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved prototype heatmap: {output_path}")


def create_prototype_legend(ax, patches, prototype_to_color, unique_prototypes, use_binning, max_samples=3):
    """Create legend showing sample patches for each prototype"""

    # Group patches by prototype
    proto_patches = defaultdict(list)
    for patch in patches:
        proto_patches[patch['prototype_id']].append(patch)

    n_prototypes = len(unique_prototypes)
    patch_width = 0.8 / n_prototypes

    for i, proto_id in enumerate(unique_prototypes):
        # Sample patches for this prototype
        sample_patches = proto_patches[proto_id][:max_samples]

        # Position for this prototype group
        x_start = i * patch_width + 0.1

        # Plot sample patches
        for j, patch in enumerate(sample_patches):
            if not os.path.exists(patch['file_location']):
                continue

            try:
                patch_img = np.array(Image.open(patch['file_location']))
                if len(patch_img.shape) == 3 and patch_img.shape[2] == 4:
                    patch_img = patch_img[:, :, :3]

                # Create small subplot for patch
                x_pos = x_start + j * (patch_width / max_samples)
                patch_ax = ax.inset_axes([x_pos, 0.3, patch_width / max_samples * 0.8, 0.6])

                # Add colored border
                border_width = 4
                h, w = patch_img.shape[:2]
                color = mcolors.to_rgb(prototype_to_color[proto_id]) if isinstance(prototype_to_color[proto_id],
                                                                                   str) else prototype_to_color[
                                                                                                 proto_id][:3]
                border_color = (np.array(color) * 255).astype(np.uint8)
                bordered_patch = np.full((h + 2 * border_width, w + 2 * border_width, 3),
                                         border_color, dtype=np.uint8)
                bordered_patch[border_width:-border_width, border_width:-border_width] = patch_img

                patch_ax.imshow(bordered_patch)
                patch_ax.axis('off')

            except Exception as e:
                print(f"Error loading sample patch: {e}")
                continue

        # Add prototype label
        label_x = x_start + patch_width / 2
        if use_binning:
            # Show bin range
            bin_ranges = {0: "0-10", 1: "11-20", 2: "21-30", 3: "31-40", 4: "41-50", 5: "51-64", 6: "65+"}
            label_text = f'Bin {proto_id} ({bin_ranges.get(proto_id, "?")})\n({len(proto_patches[proto_id])} patches)'
        else:
            label_text = f'Prototype {proto_id}\n({len(proto_patches[proto_id])} patches)'

        ax.text(label_x, 0.1, label_text,
                ha='center', va='center', fontsize=10, weight='bold',
                transform=ax.transAxes)


def analyze_prototype_distribution(patient_id, patch_assignments, output_dir, use_binning=True):
    """Analyze and visualize prototype distribution for a patient"""

    from collections import Counter

    # Apply binning if requested
    if use_binning:
        assignments_to_analyze = bin_prototypes(patch_assignments)
        assignment_type = "Binned Prototypes"
        filename_suffix = "_binned"
    else:
        assignments_to_analyze = patch_assignments
        assignment_type = "Original Prototypes"
        filename_suffix = "_original"

    # Count prototype assignments
    proto_counts = Counter(assignments_to_analyze)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    prototypes = sorted(proto_counts.keys())
    counts = [proto_counts[p] for p in prototypes]

    # Use consistent colors
    if use_binning:
        bin_colors = get_consistent_colors()
        colors = [bin_colors.get(p, '#cccccc') for p in prototypes]
    else:
        colors = plt.cm.tab20(np.array(prototypes) % 20)

    bars = plt.bar(prototypes, counts, alpha=0.7, color=colors)

    if use_binning:
        bin_ranges = {0: "0-10", 1: "11-20", 2: "21-30", 3: "31-40", 4: "41-50", 5: "51-64", 6: "65+"}
        plt.xlabel('Prototype Bin')
        # Add bin range labels
        ax = plt.gca()
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(prototypes)
        ax2.set_xticklabels([bin_ranges.get(p, "?") for p in prototypes], rotation=45)
        ax2.set_xlabel('Prototype Range')
    else:
        plt.xlabel('Prototype ID')

    plt.ylabel('Number of Patches')
    plt.title(f'{assignment_type} Distribution - Patient {patient_id}')
    plt.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom')

    # Save
    output_folder = os.path.join(output_dir, patient_id)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'{patient_id}_prototype_distribution{filename_suffix}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def generate_top_pathway_prototype_pairs(patient_id, cross_modal_attn, pathway_names, output_dir, top_k=20):
    """Generate bar chart of top pathway-prototype attention pairs"""

    attn_matrix = cross_modal_attn.squeeze(0).cpu().numpy()
    n_prototypes, n_pathways = attn_matrix.shape

    # Get top attention pairs
    flat_indices = np.argsort(attn_matrix.ravel())[-top_k:][::-1]
    top_pairs = []

    for idx in flat_indices:
        proto_idx, pathway_idx = np.unravel_index(idx, attn_matrix.shape)
        attention_val = attn_matrix[proto_idx, pathway_idx]
        pathway_name = pathway_names[pathway_idx]

        top_pairs.append({
            'prototype': proto_idx,
            'pathway': pathway_name[:40] + '...' if len(pathway_name) > 40 else pathway_name,
            'attention': attention_val,
            'label': f"P{proto_idx}: {pathway_name[:25]}{'...' if len(pathway_name) > 25 else ''}"
        })

    # Create bar plot
    plt.figure(figsize=(12, max(8, top_k * 0.4)))

    labels = [pair['label'] for pair in top_pairs]
    values = [pair['attention'] for pair in top_pairs]

    bars = plt.barh(range(len(labels)), values, alpha=0.7, color='steelblue')
    plt.yticks(range(len(labels)), labels)
    plt.xlabel('Attention Weight')
    plt.title(f'Top {top_k} Prototype-Pathway Attention Pairs - Patient {patient_id}')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()

    # Save
    output_folder = os.path.join(output_dir, patient_id)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'{patient_id}_top_pathway_prototype_pairs.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def generate_max_pathway_attention_heatmap(patient_id, patch_assignments, patch_names, patch_coordinates,
                                           cross_modal_attn, pathway_names, extracted_patches_path,
                                           output_dir, fold, patch_size=224, show_values=True, use_bin=False):
    """
    Generate spatial heatmap showing the most highly attended pathway per patch

    Args:
        show_values: If True, color by attention value; if False, color by pathway identity
    """

    # Load extracted patches CSV
    patches_df = pd.read_csv(extracted_patches_path)
    patch_to_location = {}
    for _, row in patches_df.iterrows():
        patch_to_location[row['Patch_name']] = row['File_location']

    # Get attention matrix [n_prototypes, n_pathways]
    attn_matrix = cross_modal_attn.squeeze(0).cpu().numpy()

    if use_bin:
        # Create binned attention matrix by averaging within bins
        bin_to_prototypes = defaultdict(list)
        for proto_id in range(attn_matrix.shape[0]):
            bin_id = bin_prototypes([proto_id])[0]
            bin_to_prototypes[bin_id].append(proto_id)

        # Aggregate attention by bins using mean
        n_bins = len(bin_to_prototypes)
        binned_attn_matrix = np.zeros((n_bins, attn_matrix.shape[1]))

        for bin_id, proto_ids in bin_to_prototypes.items():
            binned_attn_matrix[bin_id] = attn_matrix[proto_ids].mean(axis=0)

        effective_attn_matrix = binned_attn_matrix
        get_attention_unit = lambda proto_id: bin_prototypes([proto_id])[0]
        unit_type = "bin"
    else:
        effective_attn_matrix = attn_matrix
        get_attention_unit = lambda proto_id: proto_id
        unit_type = "prototype"

    # Process patch data with max pathway info
    patch_data = []
    for i, (patch_name_tuple, coord_str, proto_id) in enumerate(zip(patch_names, patch_coordinates, patch_assignments)):
        patch_name = patch_name_tuple[0]
        coords = eval(coord_str)
        row1, row2, col1, col2 = coords

        file_location = patch_to_location.get(patch_name)
        if file_location is None:
            continue

        # Get attention values for this prototype across all pathways
        attention_unit = get_attention_unit(proto_id)
        proto_attention = attn_matrix[proto_id, :]
        max_pathway_idx = proto_attention.argmax()
        max_attention_val = proto_attention[max_pathway_idx]
        max_pathway_name = pathway_names[max_pathway_idx]

        patch_data.append({
            'patch_name': patch_name,
            'row1': row1, 'row2': row2, 'col1': col1, 'col2': col2,
            'prototype_id': proto_id,
            'attention_unit': attention_unit,
            'max_pathway_idx': max_pathway_idx,
            'max_pathway_name': max_pathway_name,
            'max_attention_val': max_attention_val,
            'file_location': file_location
        })

    if not patch_data:
        return

    # Group by slide
    slides = defaultdict(list)
    for patch in patch_data:
        slide_name = extract_slide_name(patch['patch_name'])
        slides[slide_name].append(patch)

    # Generate heatmap for each slide
    for slide_name, slide_patches in slides.items():
        create_max_pathway_attention_slide(patient_id, slide_name, slide_patches, output_dir,
                                           fold, patch_size, pathway_names, show_values, use_bin, unit_type)


def create_max_pathway_attention_slide(patient_id, slide_name, patches, output_dir, fold,
                                       patch_size, pathway_names, show_values, use_bin, unit_type):
    """Create max pathway attention heatmap for a single slide"""

    # Calculate canvas dimensions
    max_row = max(p['row2'] for p in patches)
    max_col = max(p['col2'] for p in patches)

    canvas = np.zeros((max_row + patch_size, max_col + patch_size, 3), dtype=np.uint8)

    if show_values:
        # Color by attention value (continuous)
        value_map = np.zeros((max_row + patch_size, max_col + patch_size), dtype=np.float32)
        map_type = f"attention values ({unit_type}-aggregated)" if use_bin else "attention values"
        cmap = 'viridis'
    else:
        # Color by pathway identity (discrete)
        pathway_map = np.full((max_row + patch_size, max_col + patch_size), -1, dtype=np.int32)
        unique_pathways = sorted(set(p['max_pathway_idx'] for p in patches))
        pathway_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_pathways)))
        pathway_to_color = {pid: pathway_colors[i] for i, pid in enumerate(unique_pathways)}
        map_type = f"pathway identity ({unit_type}-aggregated)" if use_bin else "pathway identity"
        cmap = None

    # Load patches and fill canvas
    valid_patches = []
    values = []

    for patch in patches:
        if not os.path.exists(patch['file_location']):
            continue

        try:
            # Load patch image
            patch_img = np.array(Image.open(patch['file_location']))
            if len(patch_img.shape) == 3 and patch_img.shape[2] == 4:
                patch_img = patch_img[:, :, :3]

            # Place on canvas
            r1, r2, c1, c2 = patch['row1'], patch['row2'], patch['col1'], patch['col2']
            canvas[r1:r2, c1:c2] = patch_img

            if show_values:
                value_map[r1:r2, c1:c2] = patch['max_attention_val']
                values.append(patch['max_attention_val'])
            else:
                pathway_map[r1:r2, c1:c2] = patch['max_pathway_idx']

            valid_patches.append(patch)

        except Exception as e:
            continue

    if not valid_patches:
        return

    # Create the plot
    fig = plt.figure(figsize=(20, 10))
    gs = plt.GridSpec(2, 2, height_ratios=[4, 1], hspace=0.1, wspace=0.02)

    # Original slide
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Original Slide', size=16, pad=10)
    ax1.imshow(canvas)
    ax1.axis('off')

    # Max pathway heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title(f'Max Pathway Attention Heatmap ({map_type})', size=16, pad=10)
    ax2.imshow(canvas)

    if show_values:
        im = ax2.imshow(value_map, cmap=cmap, alpha=0.6,
                        vmin=np.min(values), vmax=np.max(values))
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('Max Attention Value', rotation=270, labelpad=15)
    else:
        # Create overlay for pathway colors
        overlay = np.zeros((pathway_map.shape[0], pathway_map.shape[1], 3))
        for pathway_idx in unique_pathways:
            mask = pathway_map == pathway_idx
            overlay[mask] = pathway_to_color[pathway_idx][:3]
        ax2.imshow(overlay, alpha=0.6)

    ax2.axis('off')

    # Add pathway statistics in bottom panel
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')

    from collections import Counter
    pathway_counts = Counter(p['max_pathway_idx'] for p in valid_patches)
    top_pathways = pathway_counts.most_common(5)

    aggregation_text = f" (aggregated by {unit_type})" if use_bin else ""
    stats_text = f"Top Attended Pathways{aggregation_text}:\n"
    for pathway_idx, count in top_pathways:
        pathway_name = pathway_names[pathway_idx][:40] + ('...' if len(pathway_names[pathway_idx]) > 40 else '')
        stats_text += f"{pathway_name}: {count} patches\n"

    if show_values:
        stats_text += f"\nAttention Range: {np.min(values):.4f} - {np.max(values):.4f}"

    ax_stats.text(0.1, 0.5, stats_text, ha='left', va='center',
                  fontsize=11, transform=ax_stats.transAxes,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

    title_suffix = f" ({unit_type.title()}-Aggregated)" if use_bin else ""
    plt.suptitle(f'Patient {patient_id} - {slide_name} - Max Pathway Per Patch{title_suffix}', size=18)

    # Save
    output_folder = os.path.join(output_dir, patient_id)
    os.makedirs(output_folder, exist_ok=True)

    suffix = "values" if show_values else "identity"
    bin_suffix = f"_{unit_type}" if use_bin else ""
    filename = f"{slide_name}_max_pathway_{suffix}{bin_suffix}_heatmap_fold_{fold}.png"
    output_path = os.path.join(output_folder, filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved max pathway heatmap: {output_path}")

def bin_prototypes(prototype_ids):
    """
    Bin prototype IDs into groups:
    0-10 -> 0, 11-20 -> 1, 21-30 -> 2, 31-40 -> 3, 41-50 -> 4, 51-64 -> 5
    """
    binned = []
    for proto_id in prototype_ids:
        if 0 <= proto_id <= 10:
            binned.append(0)
        elif 11 <= proto_id <= 20:
            binned.append(1)
        elif 21 <= proto_id <= 30:
            binned.append(2)
        elif 31 <= proto_id <= 40:
            binned.append(3)
        elif 41 <= proto_id <= 50:
            binned.append(4)
        elif 51 <= proto_id <= 64:
            binned.append(5)
        else:
            binned.append(6)  # fallback for any unexpected values
    return binned

def get_consistent_colors():
    """Get consistent color mapping for prototype bins"""
    bin_colors = {
        0: '#1f77b4',  # blue
        1: '#ff7f0e',  # orange
        2: '#2ca02c',  # green
        3: '#d62728',  # red
        4: '#9467bd',  # purple
        5: '#8c564b',  # brown
        6: '#e377c2'   # pink (fallback)
    }
    return bin_colors


#
# def generate_prototype_heatmap(patient_id, patch_assignments, patch_names, patch_coordinates,
#                                extracted_patches_path, output_dir, fold, patch_size=224):
#     """
#     Generate prototype assignment heatmap for a patient
#
#     Args:
#         patient_id: Patient identifier
#         patch_assignments: List of prototype assignments [18, 18, 22, ...]
#         patch_names: List of patch filename tuples
#         patch_coordinates: List of coordinate strings ['[row1, row2, col1, col2]', ...]
#         extracted_patches_path: Path to extracted_patches.csv
#         output_dir: Output directory for heatmaps
#         fold: Fold identifier
#         patch_size: Size of patches (default 224)
#     """
#
#     # Load extracted patches CSV to get file locations
#     patches_df = pd.read_csv(extracted_patches_path)
#
#     # Create mapping from patch name to file location
#     patch_to_location = {}
#     for _, row in patches_df.iterrows():
#         patch_to_location[row['Patch_name']] = row['File_location']
#
#     # Process patch data
#     patch_data = []
#     for i, (patch_name_tuple, coord_str, proto_id) in enumerate(zip(patch_names, patch_coordinates, patch_assignments)):
#         patch_name = patch_name_tuple[0]  # Extract from tuple
#
#         # Parse coordinates
#         coords = eval(coord_str)  # [row1, row2, col1, col2]
#         row1, row2, col1, col2 = coords
#
#         # Get file location
#         file_location = patch_to_location.get(patch_name, None)
#         if file_location is None:
#             continue
#
#         patch_data.append({
#             'patch_name': patch_name,
#             'row1': row1, 'row2': row2, 'col1': col1, 'col2': col2,
#             'prototype_id': proto_id,
#             'file_location': file_location
#         })
#
#     if not patch_data:
#         print(f"No valid patches found for patient {patient_id}")
#         return
#
#     # Group by slide (extract slide name from patch name)
#     slides = defaultdict(list)
#     for patch in patch_data:
#         slide_name = extract_slide_name(patch['patch_name'])
#         slides[slide_name].append(patch)
#
#     # Generate heatmap for each slide
#     for slide_name, slide_patches in slides.items():
#         create_prototype_slide_heatmap(patient_id, slide_name, slide_patches, output_dir, fold, patch_size)
#
#
# def extract_slide_name(patch_name):
#     """Extract slide name from patch filename"""
#     # Remove the coordinate part to get slide name
#     parts = patch_name.split('_row1=')
#     return parts[0] if len(parts) > 1 else patch_name
#
#
# def create_prototype_slide_heatmap(patient_id, slide_name, patches, output_dir, fold, patch_size):
#     """Create prototype heatmap for a single slide"""
#
#     # Calculate canvas dimensions
#     max_row = max(p['row2'] for p in patches)
#     max_col = max(p['col2'] for p in patches)
#
#     canvas = np.zeros((max_row + patch_size, max_col + patch_size, 3), dtype=np.uint8)
#     prototype_map = np.full((max_row + patch_size, max_col + patch_size), -1, dtype=np.int32)
#
#     # Get unique prototypes and create colormap
#     unique_prototypes = sorted(set(p['prototype_id'] for p in patches))
#     n_prototypes = len(unique_prototypes)
#     colors = plt.cm.tab20(np.linspace(0, 1, n_prototypes))
#     prototype_to_color = {proto_id: colors[i] for i, proto_id in enumerate(unique_prototypes)}
#
#     # Load patches and fill canvas
#     valid_patches = []
#     for patch in patches:
#         if not os.path.exists(patch['file_location']):
#             continue
#
#         try:
#             # Load patch image
#             patch_img = np.array(Image.open(patch['file_location']))
#             if len(patch_img.shape) == 3 and patch_img.shape[2] == 4:
#                 patch_img = patch_img[:, :, :3]
#
#             # Place on canvas
#             r1, r2, c1, c2 = patch['row1'], patch['row2'], patch['col1'], patch['col2']
#             canvas[r1:r2, c1:c2] = patch_img
#             prototype_map[r1:r2, c1:c2] = patch['prototype_id']
#
#             valid_patches.append(patch)
#
#         except Exception as e:
#             print(f"Error loading patch {patch['patch_name']}: {e}")
#             continue
#
#     if not valid_patches:
#         print(f"No valid patches loaded for slide {slide_name}")
#         return
#
#     # Create prototype overlay
#     prototype_overlay = np.zeros((prototype_map.shape[0], prototype_map.shape[1], 3))
#     for proto_id in unique_prototypes:
#         mask = prototype_map == proto_id
#         prototype_overlay[mask] = prototype_to_color[proto_id][:3]
#
#     # Create the plot
#     fig = plt.figure(figsize=(20, 10))
#     gs = plt.GridSpec(2, 2, height_ratios=[4, 1], hspace=0.1, wspace=0.02)
#
#     # Original slide
#     ax1 = fig.add_subplot(gs[0, 0])
#     ax1.set_title('Original Slide', size=16, pad=10)
#     ax1.imshow(canvas)
#     ax1.axis('off')
#
#     # Prototype heatmap
#     ax2 = fig.add_subplot(gs[0, 1])
#     ax2.set_title('Prototype Assignment Heatmap', size=16, pad=10)
#     ax2.imshow(canvas)
#     ax2.imshow(prototype_overlay, alpha=0.6)
#     ax2.axis('off')
#
#     # Prototype legend and sample patches
#     ax_legend = fig.add_subplot(gs[1, :])
#     ax_legend.axis('off')
#
#     # Create legend with sample patches
#     create_prototype_legend(ax_legend, valid_patches, prototype_to_color, unique_prototypes)
#
#     plt.suptitle(f'Patient {patient_id} - {slide_name} - Prototype Assignments', size=18)
#
#     # Save
#     output_folder = os.path.join(output_dir, patient_id)
#     os.makedirs(output_folder, exist_ok=True)
#     output_path = os.path.join(output_folder, f"{slide_name}_prototype_heatmap_fold_{fold}.png")
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close()
#
#     print(f"Saved prototype heatmap: {output_path}")
#
#
# def create_prototype_legend(ax, patches, prototype_to_color, unique_prototypes, max_samples=3):
#     """Create legend showing sample patches for each prototype"""
#
#     # Group patches by prototype
#     proto_patches = defaultdict(list)
#     for patch in patches:
#         proto_patches[patch['prototype_id']].append(patch)
#
#     n_prototypes = len(unique_prototypes)
#     patch_width = 0.8 / n_prototypes
#
#     for i, proto_id in enumerate(unique_prototypes):
#         # Sample patches for this prototype
#         sample_patches = proto_patches[proto_id][:max_samples]
#
#         # Position for this prototype group
#         x_start = i * patch_width + 0.1
#
#         # Plot sample patches
#         for j, patch in enumerate(sample_patches):
#             if not os.path.exists(patch['file_location']):
#                 continue
#
#             try:
#                 patch_img = np.array(Image.open(patch['file_location']))
#                 if len(patch_img.shape) == 3 and patch_img.shape[2] == 4:
#                     patch_img = patch_img[:, :, :3]
#
#                 # Create small subplot for patch
#                 x_pos = x_start + j * (patch_width / max_samples)
#                 patch_ax = ax.inset_axes([x_pos, 0.3, patch_width / max_samples * 0.8, 0.6])
#
#                 # Add colored border
#                 border_width = 4
#                 h, w = patch_img.shape[:2]
#                 border_color = (np.array(prototype_to_color[proto_id][:3]) * 255).astype(np.uint8)
#                 bordered_patch = np.full((h + 2 * border_width, w + 2 * border_width, 3),
#                                          border_color, dtype=np.uint8)
#                 bordered_patch[border_width:-border_width, border_width:-border_width] = patch_img
#
#                 patch_ax.imshow(bordered_patch)
#                 patch_ax.axis('off')
#
#             except Exception as e:
#                 print(f"Error loading sample patch: {e}")
#                 continue
#
#         # Add prototype label
#         label_x = x_start + patch_width / 2
#         ax.text(label_x, 0.1, f'Prototype {proto_id}\n({len(proto_patches[proto_id])} patches)',
#                 ha='center', va='center', fontsize=10, weight='bold',
#                 transform=ax.transAxes)
#
#
# def analyze_prototype_distribution(patient_id, patch_assignments, output_dir):
#     """Analyze and visualize prototype distribution for a patient"""
#
#     from collections import Counter
#
#     # Count prototype assignments
#     proto_counts = Counter(patch_assignments)
#
#     # Create bar plot
#     plt.figure(figsize=(12, 6))
#     prototypes = sorted(proto_counts.keys())
#     counts = [proto_counts[p] for p in prototypes]
#
#     bars = plt.bar(prototypes, counts, alpha=0.7, color='steelblue')
#     plt.xlabel('Prototype ID')
#     plt.ylabel('Number of Patches')
#     plt.title(f'Prototype Distribution - Patient {patient_id}')
#     plt.grid(axis='y', alpha=0.3)
#
#     # Add count labels on bars
#     for bar, count in zip(bars, counts):
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
#                  str(count), ha='center', va='bottom')
#
#     # Save
#     output_folder = os.path.join(output_dir, patient_id)
#     os.makedirs(output_folder, exist_ok=True)
#     plt.savefig(os.path.join(output_folder, f'{patient_id}_prototype_distribution.png'),
#                 dpi=300, bbox_inches='tight')
#     plt.close()