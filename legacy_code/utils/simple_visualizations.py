# utils/simple_visualizations.py

from adjustText import adjust_text

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from legacy_code.utils.helpers import ensure_directory


def volcano_plot(df, title, output_path, config=None):
    """Simple volcano plot with labels and legend"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get class names
    class_names = config.get('label_dict', {'0': 'Class 0', '1': 'Class 1'}) if config else {'0': 'Class 0',
                                                                                             '1': 'Class 1'}

    # Data
    x = df['cohens_d']
    y = -np.log10(df['p_value'] + 1e-300)

    # Colors: blue for negative Cohen's d, red for positive, gray for non-sig
    colors = np.where(~df['significant'], 'gray',
                      np.where(df['cohens_d'] < 0, '#3498DB', '#E74C3C'))

    # Plot
    ax.scatter(x, y, c=colors, alpha=0.6, s=20)

    # Thresholds
    ax.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.3)
    ax.axvline(0.2, color='black', linestyle='--', alpha=0.3)
    ax.axvline(-0.2, color='black', linestyle='--', alpha=0.3)

    # Label top features
    entity_col = 'gene' if 'gene' in df.columns else 'pathway'
    sig_df = df[df['significant'] & (df['abs_cohens_d'] > 0.2)]

    # Top 5 from each side
    left_top = sig_df[sig_df['cohens_d'] < 0].head(5)
    right_top = sig_df[sig_df['cohens_d'] > 0].head(5)

    # Label with smart positioning
    texts = []

    for _, row in pd.concat([left_top, right_top]).iterrows():
        # Truncate long names
        name = row[entity_col]
        if len(name) > 25:
            name = name[:25] + '...'

        text = plt.annotate(
            name,
            (row['cohens_d'], -np.log10(row['p_value'] + 1e-300)),
            fontsize=9, ha='center'
        )
        texts.append(text)

    # Adjust text positions to avoid overlap (if adjustText is available)
    try:
        adjust_text(texts)
    except:
        pass  # If adjustText fails, just use default positioning

    # for _, row in pd.concat([left_top, right_top]).iterrows():
    #     name = row[entity_col][:20] + '...' if len(row[entity_col]) > 20 else row[entity_col]
    #     ax.annotate(name, (row['cohens_d'], -np.log10(row['p_value'] + 1e-300)),
    #                 fontsize=8, ha='center')

    # Legend
    legend_elements = [
        plt.scatter([], [], c='#3498DB', s=50, label=f"Higher in {class_names['0']}"),
        plt.scatter([], [], c='#E74C3C', s=50, label=f"Higher in {class_names['1']}"),
        plt.scatter([], [], c='gray', s=50, label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    # Labels
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()


def bar_plot(df, x_col, y_col, title, output_path, color='steelblue', n=45, config=None):
    """Simple horizontal bar plot with value labels"""
    data = df.head(n)

    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.3)))

    class_names = config.get('label_dict', {'0': 'Class 0', '1': 'Class 1'}) if config else {'0': 'Class 0',
                                                                                             '1': 'Class 1'}

    # Color bars based on direction (for Cohen's d plots)
    if x_col in ['cohens_d', 'rank_difference'] and color == 'steelblue':
        colors = ['#3498DB' if val < 0 else '#E74C3C' for val in data[x_col]]
        show_legend = True
    else:
        colors = color
        show_legend = False

    bars = ax.barh(range(len(data)), data[x_col], color=colors, alpha=0.7)

    # Y-axis (pathway names)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([name[:50] + '...' if len(name) > 50 else name
                        for name in data[y_col]])
    ax.invert_yaxis()

    # Add vertical line at x=0 for directional plots
    if x_col in ['cohens_d', 'rank_difference']:
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    if show_legend:
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#3498DB', alpha=0.7, label=class_names['0']),
            plt.Rectangle((0,0),1,1, facecolor='#E74C3C', alpha=0.7, label=class_names['1'])
        ]
        ax.legend(handles=legend_elements)

    # Axis labels
    metric_names = {
        'abs_cohens_d': "Cohen's d (Effect Size)",
        'cohens_d': f"Cohen's d (Class 0 ← | → Class 1)",
        'rank_difference': 'Rank Difference (Class 0 ← | → Class 1)',
        'mean_importance': 'Mean Importance'
    }

    ax.set_xlabel(metric_names.get(x_col, x_col.replace('_', ' ').title()))
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf')
    plt.close()

def _get_entity_column(df):
    """Helper function to determine the entity column name"""
    if 'gene_sum' in df.columns:
        return 'gene_sum'
    elif 'gene_average' in df.columns:
        return 'gene_average'
    elif 'pathway' in df.columns:
        return 'pathway'
    else:
        raise ValueError(f"Could not find entity column. Available columns: {list(df.columns)}")

def create_plots(results_dir, output_dir, config=None):
    """Create all plots"""
    ensure_directory(output_dir)

    if config and 'label_dict' in config:
        class_names = config['label_dict']
        class_0_name = class_names.get('0', 'Class 0')
        class_1_name = class_names.get('1', 'Class 1')
    else:
        class_0_name, class_1_name = 'Class 0', 'Class 1'

    # # Volcano plots
    # plots = [
    #     ('gene_differences.csv', 'Gene Differences'),
    #     ('pathway_differences.csv', 'Pathway Differences'),
    #     ('crossmodal_differences.csv', 'Cross-modal Pathway Differences')
    # ]
    #
    # for filename, title in plots:
    #     path = f"{results_dir}/{filename}"
    #     if not os.path.exists(path):
    #         continue
    #
    #     df = pd.read_csv(path)
    #     output = f"{output_dir}/volcano_{filename.split('_')[0]}.pdf"
    #     volcano_plot(df, title, output, config)

    # Pathway ranking plots
    ranking_files = [
        ('pathway_differences.csv', 'cohens_d', 'Pathways with Highest Class Differences'),
        ('class_0_drivers_pathway_pathways.csv', 'cohens_d', f'{class_0_name} Signature Pathways'),
        ('class_1_drivers_pathway_pathways.csv', 'cohens_d', f'{class_1_name} Signature Pathways'),
        ('gene_sum_ranks.csv', 'rank_difference', 'Relative Ranking of Genes - Sum'),
        ('gene_average_ranks.csv', 'rank_difference', 'Relative Ranking of Genes - Average'),
        ('gene_sum_differences.csv', 'cohens_d', 'Highest Class Differences between Genes - sum'),
        ('gene_average_differences.csv', 'cohens_d', 'Highest Class Differences between Genes - average'),
        ('pathway_ranks.csv', 'rank_difference', 'Relative Ranking of Pathways'),
        ('class_0_drivers_crossmodal_pathways.csv', 'cohens_d', f'{class_0_name} Signature Crossmodal Pathways'),
        ('class_1_drivers_crossmodal_pathways.csv', 'cohens_d', f'{class_1_name} Signature Crossmodal Pathways'),
        ('crossmodal_ranks.csv', 'rank_difference', 'Crossmodal Relative Ranking of Pathways')
    ]

    for filename, metric_col, plot_title in ranking_files:
        path = f"{results_dir}/{filename}"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        # if 'significant' in df.columns:
        #     df = df[df['significant'] == True]

        try:
            entity_col = _get_entity_column(df)
        except ValueError as e:
            print(f"Warning: {e} in {filename}")
            continue

        output = f"{output_dir}/{filename.split('.')[0]}.pdf"

        if 'class_0' in filename:
            bar_plot(df, metric_col, entity_col, f'{plot_title}', output, color='#3498DB', config=config)
        elif 'class_1' in filename:
            bar_plot(df, metric_col, entity_col, f'{plot_title}', output, color='#E74C3C', config=config)
        else:
            bar_plot(df, metric_col, entity_col, f'{plot_title}', output, config=config)

    print(f"Plots saved to {output_dir}")

#
# def create_volcano_plot(diff_results, analysis_type, output_path, config=None, top_k=8):
#     """
#     Create clean volcano plot for differential analysis results
#
#     Args:
#         diff_results: DataFrame from class_differences() method
#         analysis_type: 'gene', 'pathway', or 'crossmodal'
#         output_path: Path to save figure
#         config: Config dict with label_dict for class names
#         top_k: Number of top features to label
#     """
#
#     plt.figure(figsize=(12, 8))
#
#     # Get class names
#     if config and 'label_dict' in config:
#         class_names = config['label_dict']
#         class_0_name = class_names.get('0', 'Class 0')
#         class_1_name = class_names.get('1', 'Class 1')
#     else:
#         class_0_name, class_1_name = 'Class 0', 'Class 1'
#
#     # Calculate values
#     neg_log_p = -np.log10(diff_results['p_value'] + 1e-300)
#     effect_size = diff_results['cohens_d']
#
#     print(f"Total genes: {len(diff_results)}")
#     print(f"Significant genes: {len(diff_results[diff_results['significant'] == True])}")
#     print(f"Non-significant genes: {len(diff_results[diff_results['significant'] == False])}")
#     print(f"P-value range: {diff_results['p_value'].min()} to {diff_results['p_value'].max()}")
#     print(f"-log10(p) range: {neg_log_p.min()} to {neg_log_p.max()}")
#
#     # Define thresholds
#     p_threshold = 0.05
#     effect_threshold = 0.15
#
#     # Create color mapping by class and significance
#     colors = []
#     labels = []
#
#     for _, row in diff_results.iterrows():
#         is_sig = row['significant'] and abs(row['cohens_d']) > effect_threshold
#
#         if not is_sig:
#             colors.append('#CCCCCC')  # Gray for non-significant
#             labels.append('Not significant')
#         elif row['cohens_d'] > 0:  # Favors class 1
#             colors.append('#E74C3C')  # Red for class 1
#             labels.append(f'Higher in {class_1_name}')
#         else:  # Favors class 0
#             colors.append('#3498DB')  # Blue for class 0
#             labels.append(f'Higher in {class_0_name}')
#
#     # Create scatter plot
#     plt.scatter(effect_size, neg_log_p, c=colors, alpha=0.7, s=35, edgecolors='white', linewidth=0.5)
#     plt.ylim(bottom=0)
#
#     color_counts = pd.Series(colors).value_counts()
#     print(f"Color distribution: {color_counts}")
#
#     # Add threshold lines
#     plt.axhline(y=-np.log10(p_threshold), color='black', linestyle='--', alpha=0.3, linewidth=1)
#     plt.axvline(x=effect_threshold, color='black', linestyle='--', alpha=0.3, linewidth=1)
#     plt.axvline(x=-effect_threshold, color='black', linestyle='--', alpha=0.3, linewidth=1)
#
#     # Smart label positioning for top features
#     entity_col = 'gene' if analysis_type == 'gene' else 'pathway'
#
#     # Get top features from each side
#     sig_results = diff_results[
#         (diff_results['significant'] == True) &
#         (diff_results['abs_cohens_d'] > effect_threshold)
#         ]
#
#     # Top features favoring each class
#     class_0_top = sig_results[sig_results['cohens_d'] < 0].head(top_k // 2)
#     class_1_top = sig_results[sig_results['cohens_d'] > 0].head(top_k // 2)
#
#     # Label with smart positioning
#     texts = []
#
#     for _, row in pd.concat([class_0_top, class_1_top]).iterrows():
#         # Truncate long names
#         name = row[entity_col]
#         if len(name) > 25:
#             name = name[:25] + '...'
#
#         text = plt.annotate(
#             name,
#             (row['cohens_d'], -np.log10(row['p_value'] + 1e-300)),
#             fontsize=9, ha='center'
#         )
#         texts.append(text)
#
#     # Adjust text positions to avoid overlap (if adjustText is available)
#     try:
#         adjust_text(texts)
#     except:
#         pass  # If adjustText fails, just use default positioning
#
#     # Custom legend
#     legend_elements = [
#         plt.scatter([], [], c='#3498DB', s=50, label=f'Higher in {class_0_name}'),
#         plt.scatter([], [], c='#E74C3C', s=50, label=f'Higher in {class_1_name}'),
#         plt.scatter([], [], c='#CCCCCC', s=50, label='Not significant')
#     ]
#     plt.legend(handles=legend_elements, loc='lower left', framealpha=0.9)
#
#     # Formatting
#     plt.xlabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
#     plt.ylabel("-log₁₀(p-value)", fontsize=12, fontweight='bold')
#     plt.title(f"Differential {analysis_type.title()} Analysis", fontsize=14, fontweight='bold', pad=20)
#
#     # Add summary box in bottom right
#     n_class_0 = len(sig_results[sig_results['cohens_d'] < 0])
#     n_class_1 = len(sig_results[sig_results['cohens_d'] > 0])
#
#     summary_text = f'{class_0_name}: {n_class_0}\n{class_1_name}: {n_class_1}'
#     plt.text(0.98, 0.02, summary_text, transform=plt.gca().transAxes,
#              fontsize=10, verticalalignment='bottom', horizontalalignment='right',
#              bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
#
#     # Clean up axes
#     plt.grid(True, alpha=0.2)
#     plt.tight_layout()
#
#     ensure_directory(output_path.parent if hasattr(output_path, 'parent') else output_path.rsplit('/', 1)[0])
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
#
#
# def create_summary_volcano_plots(results_dir, output_dir, config=None):
#     """
#     Create volcano plots for all analysis types
#
#     Args:
#         results_dir: Directory containing CSV results
#         output_dir: Directory to save plots
#         config: Configuration dict with label_dict
#     """
#
#     analysis_files = [
#         ('gene_differences.csv', 'gene'),
#         ('pathway_differences.csv', 'pathway'),
#         ('crossmodal_pathway_differences.csv', 'crossmodal pathways')
#     ]
#
#     ensure_directory(output_dir)
#
#     for filename, analysis_type in analysis_files:
#         file_path = f"{results_dir}/{filename}"
#
#         try:
#             diff_results = pd.read_csv(file_path)
#             output_path = f"{output_dir}/volcano_{analysis_type}.png"
#
#             create_volcano_plot(
#                 diff_results,
#                 analysis_type,
#                 output_path,
#                 config=config,
#                 top_k=20
#             )
#
#             print(f"Created volcano plot: {output_path}")
#
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")
#         except Exception as e:
#             print(f"Error creating {analysis_type} volcano plot: {e}")
#
# def create_simple_bar_chart(class_results, analysis_type, class_label, output_path, top_k=20):
#     """
#     Simple bar chart for top features by class
#
#     Args:
#         class_results: DataFrame from class_aggregation() method
#         analysis_type: 'gene', 'pathway', or 'crossmodal'
#         class_label: Class identifier
#         output_path: Path to save figure
#         top_k: Number of top features to show
#     """
#
#     plt.figure(figsize=(12, 8))
#
#     # Get top features
#     top_data = class_results.head(top_k)
#     entity_col = 'gene' if analysis_type == 'gene' else 'pathway'
#
#     # Create horizontal bar chart
#     bars = plt.barh(range(len(top_data)), top_data['mean_importance'], color='steelblue', alpha=0.7)
#
#     # Customize
#     plt.yticks(range(len(top_data)), [name[:50] + ('...' if len(name) > 50 else '')
#                                       for name in top_data[entity_col]])
#     plt.xlabel('Mean Importance Score', fontsize=12)
#     plt.title(f'Top {top_k} {analysis_type.title()}s - Class {class_label}', fontsize=14, fontweight='bold')
#
#     # Add value labels on bars
#     for i, bar in enumerate(bars):
#         width = bar.get_width()
#         plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2,
#                  f'{width:.3f}', ha='left', va='center', fontsize=9)
#
#     plt.gca().invert_yaxis()  # Top feature at top
#     plt.tight_layout()
#
#     ensure_directory(output_path.parent if hasattr(output_path, 'parent') else output_path.rsplit('/', 1)[0])
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()