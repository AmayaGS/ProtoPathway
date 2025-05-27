import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict


def generate_prototype_heatmap(patient_id, patch_assignments, patch_names, patch_coordinates,
                               extracted_patches_path, output_dir, fold, patch_size=224):
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
    """

    # Load extracted patches CSV to get file locations
    patches_df = pd.read_csv(extracted_patches_path)

    # Create mapping from patch name to file location
    patch_to_location = {}
    for _, row in patches_df.iterrows():
        patch_to_location[row['Patch_name']] = row['File_location']

    # Process patch data
    patch_data = []
    for i, (patch_name_tuple, coord_str, proto_id) in enumerate(zip(patch_names, patch_coordinates, patch_assignments)):
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
        create_prototype_slide_heatmap(patient_id, slide_name, slide_patches, output_dir, fold, patch_size)


def extract_slide_name(patch_name):
    """Extract slide name from patch filename"""
    # Remove the coordinate part to get slide name
    parts = patch_name.split('_row1=')
    return parts[0] if len(parts) > 1 else patch_name


def create_prototype_slide_heatmap(patient_id, slide_name, patches, output_dir, fold, patch_size):
    """Create prototype heatmap for a single slide"""

    # Calculate canvas dimensions
    max_row = max(p['row2'] for p in patches)
    max_col = max(p['col2'] for p in patches)

    canvas = np.zeros((max_row + patch_size, max_col + patch_size, 3), dtype=np.uint8)
    prototype_map = np.full((max_row + patch_size, max_col + patch_size), -1, dtype=np.int32)

    # Get unique prototypes and create colormap
    unique_prototypes = sorted(set(p['prototype_id'] for p in patches))
    n_prototypes = len(unique_prototypes)
    colors = plt.cm.tab20(np.linspace(0, 1, n_prototypes))
    prototype_to_color = {proto_id: colors[i] for i, proto_id in enumerate(unique_prototypes)}

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
        prototype_overlay[mask] = prototype_to_color[proto_id][:3]

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
    ax2.set_title('Prototype Assignment Heatmap', size=16, pad=10)
    ax2.imshow(canvas)
    ax2.imshow(prototype_overlay, alpha=0.6)
    ax2.axis('off')

    # Prototype legend and sample patches
    ax_legend = fig.add_subplot(gs[1, :])
    ax_legend.axis('off')

    # Create legend with sample patches
    create_prototype_legend(ax_legend, valid_patches, prototype_to_color, unique_prototypes)

    plt.suptitle(f'Patient {patient_id} - {slide_name} - Prototype Assignments', size=18)

    # Save
    output_folder = os.path.join(output_dir, patient_id)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{slide_name}_prototype_heatmap_fold_{fold}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved prototype heatmap: {output_path}")


def create_prototype_legend(ax, patches, prototype_to_color, unique_prototypes, max_samples=3):
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
                border_color = (np.array(prototype_to_color[proto_id][:3]) * 255).astype(np.uint8)
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
        ax.text(label_x, 0.1, f'Prototype {proto_id}\n({len(proto_patches[proto_id])} patches)',
                ha='center', va='center', fontsize=10, weight='bold',
                transform=ax.transAxes)


def analyze_prototype_distribution(patient_id, patch_assignments, output_dir):
    """Analyze and visualize prototype distribution for a patient"""

    from collections import Counter

    # Count prototype assignments
    proto_counts = Counter(patch_assignments)

    # Create bar plot
    plt.figure(figsize=(12, 6))
    prototypes = sorted(proto_counts.keys())
    counts = [proto_counts[p] for p in prototypes]

    bars = plt.bar(prototypes, counts, alpha=0.7, color='steelblue')
    plt.xlabel('Prototype ID')
    plt.ylabel('Number of Patches')
    plt.title(f'Prototype Distribution - Patient {patient_id}')
    plt.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom')

    # Save
    output_folder = os.path.join(output_dir, patient_id)
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, f'{patient_id}_prototype_distribution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()