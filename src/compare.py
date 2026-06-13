import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import pandas as pd
import seaborn as sns
from PIL import Image
import os
import pickle
from datetime import datetime
from itertools import product
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import copy
from scipy import ndimage
from PIL import Image
from itertools import product
from sklearn.metrics import mean_squared_error
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import entropy

from TCDGI import (tcdgi, normalize_image, simulate_ghost_imaging,
                   load_and_preprocess_image, calculate_ssim, calculate_quality_metrics,
                   enhance_edges)


def advanced_compare_imaging_methods(reference_frames, bucket_signals, object_mask,
                                     thresholds=[0.5], filter_sigmas=[1.0],
                                     use_weighted_avgs=[True], include_edge_enhancement=False,
                                     edge_factors=[1.0], edge_methods=['sobel'],
                                     include_iterative=False, iterations_list=[3],
                                     save_results=False, output_dir=None):
    """
    Provides a detailed comparative analysis of multiple ghost imaging methods.

    Parameters
    -----------
    reference_frames : ndarray
        Array of reference detector frames
    bucket_signals : ndarray
        Array of bucket detector signals
    object_mask : ndarray
        Original object mask
    thresholds : list
        List of TCDGI thresholds
    filter_sigmas : list
        List of Gaussian filter standard deviations
    use_weighted_avgs : list
        List of booleans indicating whether to use weighted averaging
    include_edge_enhancement : bool
        Whether to include edge enhancement methods
    edge_factors : list
        List of edge enhancement factors
    edge_methods : list
        List of edge detection methods ('sobel', 'laplacian', 'prewitt')
    include_iterative : bool
        Whether to include iterative optimization methods
    iterations_list : list
        List of iteration counts
    save_results : bool
        Whether to save the results
    output_dir : str
        Output directory path

    Returns
    --------
    results : dict
        Dictionary containing all reconstructed images and their quality metrics
    metrics_df : DataFrame
        Comparison table of quality metrics for all methods
    """
    print("Starting multi-algorithm comparison analysis...")

    num_frames = len(reference_frames)
    results = {}

    # Create results directory
    if save_results and output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.getcwd(), f'comparison_results_{timestamp}')

    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created results directory: {output_dir}")

    # Store the original image for comparison
    results['Original'] = {
        'image': object_mask,
        'method': 'Original',
        'ssim': 1.0,
        'snr': float('inf'),
        'psnr': float('inf'),
        'rmse': 0.0
    }

    # 1. Implement traditional ghost imaging (GI)
    print("Computing traditional ghost imaging (GI)...")
    bucket_fluctuations = bucket_signals - np.mean(bucket_signals)
    gi_image = np.zeros_like(reference_frames[0])

    for i in range(num_frames):
        gi_image += bucket_fluctuations[i] * reference_frames[i]
    gi_image /= num_frames

    gi_metrics = calculate_quality_metrics(
        object_mask, gi_image, norm_type='centered')
    results['GI'] = {
        'image': gi_image,
        'method': 'GI',
        **gi_metrics
    }

    # 2. Weighted traditional ghost imaging (Weighted GI)
    print("Computing weighted traditional ghost imaging (Weighted GI)...")
    weights = np.abs(bucket_fluctuations)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights) * num_frames
        wgi_image = np.zeros_like(reference_frames[0])

        for i in range(num_frames):
            wgi_image += weights[i] * \
                bucket_fluctuations[i] * reference_frames[i]
        wgi_image /= num_frames

        wgi_metrics = calculate_quality_metrics(
            object_mask, wgi_image, norm_type='centered')
        results['Weighted_GI'] = {
            'image': wgi_image,
            'method': 'Weighted GI',
            **wgi_metrics
        }

    # 3. Differential ghost imaging (DGI)
    print("Computing differential ghost imaging (DGI)...")
    ref_integrated = np.sum(reference_frames, axis=(1, 2))
    ref_integrated_mean = np.mean(ref_integrated)
    bucket_mean = np.mean(bucket_signals)
    differential_bucket = bucket_signals - \
        bucket_mean * ref_integrated / ref_integrated_mean

    dgi_image = np.zeros_like(reference_frames[0])
    for i in range(num_frames):
        dgi_image += differential_bucket[i] * reference_frames[i]
    dgi_image /= num_frames

    dgi_metrics = calculate_quality_metrics(
        object_mask, dgi_image, norm_type='centered')
    results['DGI'] = {
        'image': dgi_image,
        'method': 'DGI',
        **dgi_metrics
    }

    # 4. Time-correspondence differential ghost imaging (TCDGI) with different parameter combinations
    print("Computing time-correspondence differential ghost imaging (TCDGI) with multiple parameters...")
    for k in thresholds:
        for sigma in filter_sigmas:
            for weighted in use_weighted_avgs:
                # Build the parameter description
                param_desc = f'T{k:.2f}_S{sigma:.1f}'
                if weighted:
                    param_desc += '_W'
                else:
                    param_desc += '_NW'

                method_name = f'TCDGI_{param_desc}'
                print(f"  Computing {method_name}...")

                # Call the TCDGI algorithm
                tcdgi_image = tcdgi(
                    reference_frames,
                    bucket_signals,
                    threshold=k,
                    filter_sigma=sigma,
                    use_weighted_avg=weighted
                )

                if tcdgi_image is not None:
                    tcdgi_metrics = calculate_quality_metrics(
                        object_mask, tcdgi_image, norm_type='centered')
                    results[method_name] = {
                        'image': tcdgi_image,
                        'method': 'TCDGI',
                        'threshold': k,
                        'filter_sigma': sigma,
                        'use_weighted_avg': weighted,
                        'param_desc': param_desc,
                        **tcdgi_metrics
                    }

                    # 5. If edge enhancement is enabled, compute the edge-enhanced TCDGI result
                    if include_edge_enhancement and tcdgi_image is not None:
                        for edge_factor in edge_factors:
                            for edge_method in edge_methods:
                                edge_desc = f'{param_desc}_E{edge_factor:.1f}_{edge_method[:3]}'
                                edge_method_name = f'TCDGI_Edge_{edge_desc}'
                                print(f"    Computing edge-enhanced {edge_method_name}...")

                                # Apply edge enhancement
                                enhanced_image = enhance_edges(
                                    tcdgi_image,
                                    enhancement_factor=edge_factor,
                                    method=edge_method
                                )

                                edge_metrics = calculate_quality_metrics(
                                    object_mask, enhanced_image, norm_type='centered')
                                results[edge_method_name] = {
                                    'image': enhanced_image,
                                    'method': 'TCDGI_Edge',
                                    'threshold': k,
                                    'filter_sigma': sigma,
                                    'use_weighted_avg': weighted,
                                    'edge_factor': edge_factor,
                                    'edge_method': edge_method,
                                    'param_desc': edge_desc,
                                    **edge_metrics
                                }

    # 6. Iteratively optimized TCDGI (if enabled)
    if include_iterative:
        print("Computing the iteratively optimized TCDGI algorithm...")

        # Use the best basic TCDGI parameters as the starting point for iteration
        best_tcdgi = None
        best_snr = -1

        for key, value in results.items():
            if 'method' in value and value['method'] == 'TCDGI' and 'snr' in value:
                if value['snr'] > best_snr:
                    best_snr = value['snr']
                    best_tcdgi = value

        if best_tcdgi is not None:
            best_threshold = best_tcdgi.get('threshold', thresholds[0])
            best_sigma = best_tcdgi.get('filter_sigma', filter_sigmas[0])
            best_weighted = best_tcdgi.get(
                'use_weighted_avg', use_weighted_avgs[0])

            for iterations in iterations_list:
                iter_desc = f'T{best_threshold:.2f}_S{best_sigma:.1f}_I{iterations}'
                iter_method_name = f'TCDGI_Iter_{iter_desc}'
                print(f"  Computing iterative optimization {iter_method_name}...")

                # Base TCDGI image
                current_image = tcdgi(
                    reference_frames,
                    bucket_signals,
                    threshold=best_threshold,
                    filter_sigma=best_sigma,
                    use_weighted_avg=best_weighted
                )

                if current_image is not None:
                    # Iterative optimization
                    for i in range(1, iterations):
                        print(f"    Performing iteration {i+1}/{iterations}...")

                        # Weight the reference frames based on the previous result
                        weighted_refs = reference_frames.copy()
                        for j in range(len(reference_frames)):
                            # Compute correlation as the weighting factor
                            correlation = np.sum(
                                reference_frames[j] * current_image) / np.sum(reference_frames[j]**2)
                            weighted_refs[j] = reference_frames[j] * \
                                (1 + correlation * i / iterations)

                        # Gradually adjust the parameters
                        current_threshold = best_threshold * \
                            (1 + 0.1*i)  # Gradually increase the threshold
                        current_sigma = best_sigma * (1 - 0.05*i)  # Gradually reduce the blur

                        # Run TCDGI with the adjusted parameters and weighted reference frames
                        new_image = tcdgi(
                            weighted_refs,
                            bucket_signals,
                            threshold=current_threshold,
                            filter_sigma=current_sigma,
                            use_weighted_avg=best_weighted
                        )

                        if new_image is not None:
                            current_image = new_image

                    # Record the final iteration result
                    iter_metrics = calculate_quality_metrics(
                        object_mask, current_image, norm_type='centered')
                    results[iter_method_name] = {
                        'image': current_image,
                        'method': 'TCDGI_Iterative',
                        'threshold': best_threshold,
                        'filter_sigma': best_sigma,
                        'use_weighted_avg': best_weighted,
                        'iterations': iterations,
                        'param_desc': iter_desc,
                        **iter_metrics
                    }

                    # 7. If edge enhancement is also enabled, add the iterative + edge-enhanced result
                    if include_edge_enhancement:
                        for edge_factor in edge_factors:
                            for edge_method in edge_methods:
                                iter_edge_desc = f'{iter_desc}_E{edge_factor:.1f}_{edge_method[:3]}'
                                iter_edge_name = f'TCDGI_Iter_Edge_{iter_edge_desc}'
                                print(f"    Computing iterative + edge-enhanced {iter_edge_name}...")

                                # Apply edge enhancement
                                enhanced_iter_image = enhance_edges(
                                    current_image,
                                    enhancement_factor=edge_factor,
                                    method=edge_method
                                )

                                iter_edge_metrics = calculate_quality_metrics(
                                    object_mask, enhanced_iter_image, norm_type='centered')
                                results[iter_edge_name] = {
                                    'image': enhanced_iter_image,
                                    'method': 'TCDGI_Iterative_Edge',
                                    'threshold': best_threshold,
                                    'filter_sigma': best_sigma,
                                    'use_weighted_avg': best_weighted,
                                    'iterations': iterations,
                                    'edge_factor': edge_factor,
                                    'edge_method': edge_method,
                                    'param_desc': iter_edge_desc,
                                    **iter_edge_metrics
                                }

    # Build a metrics data frame consolidating quality metrics for all results
    metrics_data = []
    for method_name, result in results.items():
        if method_name == 'Original':
            continue

        method_data = {
            'Method': method_name,
            'Category': result.get('method', 'Unknown'),
            'SSIM': result.get('ssim', 0),
            'SNR': result.get('snr', 0),
            'PSNR': result.get('psnr', 0),
            'RMSE': result.get('rmse', 0),
        }

        # Add extra parameter information
        if 'threshold' in result:
            method_data['Threshold'] = result['threshold']
        if 'filter_sigma' in result:
            method_data['Filter_Sigma'] = result['filter_sigma']
        if 'use_weighted_avg' in result:
            method_data['Weighted_Avg'] = result['use_weighted_avg']
        if 'edge_factor' in result:
            method_data['Edge_Factor'] = result['edge_factor']
        if 'edge_method' in result:
            method_data['Edge_Method'] = result['edge_method']
        if 'iterations' in result:
            method_data['Iterations'] = result['iterations']

        metrics_data.append(method_data)

    metrics_df = pd.DataFrame(metrics_data)

    # If results need to be saved
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save all images
        for method_name, result in results.items():
            if 'image' in result:
                image_path = os.path.join(
                    output_dir, f"{method_name}_{timestamp}.png")
                plt.figure(figsize=(6, 6))
                plt.imshow(normalize_image(
                    result['image'], norm_type='centered'), cmap='gray')
                plt.title(method_name)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(image_path, dpi=300)
                plt.close()

        # Save the metrics results to CSV
        csv_path = os.path.join(
            output_dir, f"metrics_comparison_{timestamp}.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics results saved to: {csv_path}")

        # Save the complete results as a pickle file (including image data)
        pickle_path = os.path.join(
            output_dir, f"complete_results_{timestamp}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Complete results saved to: {pickle_path}")

    return results, metrics_df


def visualize_comparison_results(results, metrics_df=None, num_methods=None, sort_by='SNR'):
    """
    Generates a detailed comparison visualization, including image quality comparison and parameter impact analysis.

    Parameters
    -----------
    results : dict
        Dictionary containing all reconstructed images and their quality metrics
    metrics_df : DataFrame, optional
        Comparison table of quality metrics for all methods
    num_methods : int, optional
        Number of top-ranked methods to display; shows all by default
    sort_by : str
        Metric to sort by ('SNR', 'SSIM', 'PSNR')

    Returns
    --------
    best_method : str
        Name of the best reconstruction method
    """
    if metrics_df is None:
        # If no metrics data frame was provided, create one from the results
        metrics_data = []
        for method_name, result in results.items():
            if method_name == 'Original':
                continue

            method_data = {
                'Method': method_name,
                'Category': result.get('method', 'Unknown'),
                'SSIM': result.get('ssim', 0),
                'SNR': result.get('snr', 0),
                'PSNR': result.get('psnr', 0),
                'RMSE': result.get('rmse', 0),
            }
            metrics_data.append(method_data)

        metrics_df = pd.DataFrame(metrics_data)

    # Sort by the specified metric
    sorted_df = metrics_df.sort_values(
        by=sort_by, ascending=False).reset_index(drop=True)

    if num_methods is not None:
        sorted_df = sorted_df.head(num_methods)

    # Extract the methods needed for image comparison
    methods_to_show = sorted_df['Method'].tolist()

    # 1. Configure CJK font support

    # 2. Create the image comparison visualization
    num_methods = len(methods_to_show)
    if num_methods > 0:
        # Display the original image and the top results
        fig_height = 12
        fig_width = 15
        if num_methods > 5:
            fig_height = min(20, 4 + num_methods * 1.2)

        plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(3, 3)

        # Display the original image
        ax0 = plt.subplot(gs[0, 0])
        if 'Original' in results and 'image' in results['Original']:
            ax0.imshow(normalize_image(
                results['Original']['image'], norm_type='centered'), cmap='gray')
            ax0.set_title('Original Image')
            ax0.axis('off')

        # Display a bar chart comparing quality metrics
        ax1 = plt.subplot(gs[0, 1:])
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))

        # Show SSIM and SNR side by side
        bar_width = 0.35
        x = np.arange(len(sorted_df))

        ax1.bar(x - bar_width/2, sorted_df['SSIM'],
                bar_width, label='SSIM', color=colors, alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.bar(
            x + bar_width/2, sorted_df['SNR'], bar_width, label='SNR', color=colors, alpha=0.4)

        ax1.set_ylabel('SSIM Value', fontsize=10)
        ax1_twin.set_ylabel('SNR Value', fontsize=10)
        ax1.set_title('SSIM and SNR Metric Comparison', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels([name[:15]+'...' if len(name) > 15 else name for name in sorted_df['Method']],
                            rotation=45, ha='right', fontsize=8)

        # Add legend
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2,
                   loc='upper right', fontsize=8)

        # Set an appropriate y-axis range
        ax1.set_ylim(0, min(1, sorted_df['SSIM'].max() * 1.2))
        ax1_twin.set_ylim(0, sorted_df['SNR'].max() * 1.2)

        # Add value labels
        for i, (ssim, snr) in enumerate(zip(sorted_df['SSIM'], sorted_df['SNR'])):
            ax1.text(i - bar_width/2, ssim + 0.02, f"{ssim:.3f}",
                     ha='center', va='bottom', fontsize=7, rotation=0)
            ax1_twin.text(i + bar_width/2, snr + 0.1, f"{snr:.3f}",
                          ha='center', va='bottom', fontsize=7, rotation=0)

        # Display a grid of reconstructed images
        rows = (num_methods + 2) // 3
        if rows < 2:
            rows = 2

        for i, method_name in enumerate(methods_to_show):
            row = 1 + i // 3
            col = i % 3

            if row < 3:  # Only the first two rows go into the GridSpec
                ax = plt.subplot(gs[row, col])
            else:
                # For methods exceeding the GridSpec, add subplots below the grid
                plt.subplot(rows, 3, 3 + i + 1)

            if method_name in results and 'image' in results[method_name]:
                result = results[method_name]
                plt.imshow(normalize_image(
                    result['image'], norm_type='centered'), cmap='gray')

                # Build a more detailed title
                method_desc = method_name
                if 'method' in result:
                    base_method = result['method']

                    if 'param_desc' in result:
                        param_desc = result['param_desc']
                        title = f"{base_method}\n{param_desc}"
                    else:
                        title = base_method

                    # Add quality metrics
                    metrics_text = f"\nSSIM: {result.get('ssim', 0):.3f}, SNR: {result.get('snr', 0):.3f}"
                    title += metrics_text

                    plt.title(title, fontsize=9)
                else:
                    plt.title(method_name, fontsize=9)

                plt.axis('off')

        plt.tight_layout()
        plt.suptitle('Ghost Imaging Algorithm Quality Comparison', fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
        plt.show()

        # 3. Create a heatmap of evaluation metrics
        plt.figure(figsize=(12, 6))

        # Extract the main metrics
        metrics_heatmap = sorted_df[['Method', 'SSIM', 'SNR', 'PSNR']].copy()

        # Normalize metrics for easier comparison
        for col in ['SSIM', 'SNR', 'PSNR']:
            min_val = metrics_heatmap[col].min()
            max_val = metrics_heatmap[col].max()
            metrics_heatmap[col + '_norm'] = (metrics_heatmap[col] - min_val) / (
                max_val - min_val) if max_val > min_val else 0

        # Create the heatmap
        heatmap_data = metrics_heatmap[[
            'SSIM_norm', 'SNR_norm', 'PSNR_norm']].values

        plt.subplot(1, 2, 1)
        sns.heatmap(heatmap_data, annot=False, cmap='viridis',
                    xticklabels=['SSIM', 'SNR', 'PSNR'],
                    yticklabels=[name[:20]+'...' if len(name) > 20 else name for name in metrics_heatmap['Method']])
        plt.title('Normalized Metrics Heatmap', fontsize=12)

        # Add a table of the raw values
        cell_text = []
        for i, row in metrics_heatmap.iterrows():
            cell_text.append(
                [f"{row['SSIM']:.3f}", f"{row['SNR']:.3f}", f"{row['PSNR']:.2f}"])

        plt.subplot(1, 2, 2)
        table = plt.table(
            cellText=cell_text,
            rowLabels=[
                name[:20]+'...' if len(name) > 20 else name for name in metrics_heatmap['Method']],
            colLabels=['SSIM', 'SNR', 'PSNR'],
            loc='center',
            cellLoc='center'
        )

        # Adjust the table font size
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        plt.axis('off')
        plt.title('Metrics Value Table', fontsize=12)

        plt.tight_layout()
        plt.suptitle('Ghost Imaging Algorithm Metric Comparison', fontsize=16, y=0.98)
        plt.subplots_adjust(top=0.85)
        plt.show()

        # 4. If there is enough data, create a parameter impact analysis plot
        if len(metrics_df) > 5:
            # Check whether there is enough TCDGI data for parameter analysis
            tcdgi_df = metrics_df[metrics_df['Category'].isin(
                ['TCDGI', 'TCDGI_Edge', 'TCDGI_Iterative'])]

            if len(tcdgi_df) >= 4 and 'Threshold' in tcdgi_df.columns:
                plt.figure(figsize=(15, 8))

                # Effect of threshold on SSIM
                plt.subplot(2, 2, 1)
                sns.lineplot(x='Threshold', y='SSIM',
                             hue='Category', data=tcdgi_df, marker='o')
                plt.title('Effect of Threshold on SSIM', fontsize=12)
                plt.grid(True, alpha=0.3)

                # Effect of threshold on SNR
                plt.subplot(2, 2, 2)
                sns.lineplot(x='Threshold', y='SNR',
                             hue='Category', data=tcdgi_df, marker='o')
                plt.title('Effect of Threshold on SNR', fontsize=12)
                plt.grid(True, alpha=0.3)

                # Effect of filter parameter on SSIM (if available)
                if 'Filter_Sigma' in tcdgi_df.columns:
                    plt.subplot(2, 2, 3)
                    sns.lineplot(x='Filter_Sigma', y='SSIM',
                                 hue='Category', data=tcdgi_df, marker='o')
                    plt.title('Effect of Filter Parameter on SSIM', fontsize=12)
                    plt.grid(True, alpha=0.3)

                # Weighted vs unweighted comparison
                if 'Weighted_Avg' in tcdgi_df.columns:
                    plt.subplot(2, 2, 4)
                    sns.barplot(x='Weighted_Avg', y='SNR',
                                hue='Category', data=tcdgi_df)
                    plt.title('Effect of Weighting Method on SNR', fontsize=12)
                    plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.suptitle('TCDGI Parameter Impact Analysis', fontsize=16, y=0.98)
                plt.subplots_adjust(top=0.9)
                plt.show()

            # 5. If edge enhancement and iterative optimization data are available, create dedicated comparison plots
            if 'Edge_Factor' in metrics_df.columns:
                edge_df = metrics_df[metrics_df['Category'].isin(
                    ['TCDGI_Edge', 'TCDGI_Iterative_Edge'])]
                if len(edge_df) > 0:
                    plt.figure(figsize=(12, 6))

                    plt.subplot(1, 2, 1)
                    sns.lineplot(x='Edge_Factor', y='SSIM',
                                 hue='Category', data=edge_df, marker='o')
                    plt.title('Effect of Edge Enhancement Factor on SSIM', fontsize=12)
                    plt.grid(True, alpha=0.3)

                    plt.subplot(1, 2, 2)
                    sns.lineplot(x='Edge_Factor', y='SNR',
                                 hue='Category', data=edge_df, marker='o')
                    plt.title('Effect of Edge Enhancement Factor on SNR', fontsize=12)
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.suptitle('Edge Enhancement Effect Analysis', fontsize=16, y=0.98)
                    plt.subplots_adjust(top=0.85)
                    plt.show()

            if 'Iterations' in metrics_df.columns:
                iter_df = metrics_df[metrics_df['Category'].isin(
                    ['TCDGI_Iterative', 'TCDGI_Iterative_Edge'])]
                if len(iter_df) > 0:
                    plt.figure(figsize=(12, 6))

                    plt.subplot(1, 2, 1)
                    sns.lineplot(x='Iterations', y='SSIM',
                                 hue='Category', data=iter_df, marker='o')
                    plt.title('Effect of Iteration Count on SSIM', fontsize=12)
                    plt.grid(True, alpha=0.3)

                    plt.subplot(1, 2, 2)
                    sns.lineplot(x='Iterations', y='SNR',
                                 hue='Category', data=iter_df, marker='o')
                    plt.title('Effect of Iteration Count on SNR', fontsize=12)
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.suptitle('Iterative Optimization Effect Analysis', fontsize=16, y=0.98)
                    plt.subplots_adjust(top=0.85)
                    plt.show()

    # Print detailed information about the best method
    if len(sorted_df) > 0:
        best_method = sorted_df.iloc[0]['Method']
        print("\nBest method details:")
        best_result = results[best_method]

        print(f"Method name: {best_method}")
        print(f"Method category: {best_result.get('method', 'Unknown')}")
        print(f"Quality metrics:")
        print(f"  SSIM: {best_result.get('ssim', 0):.4f}")
        print(f"  SNR: {best_result.get('snr', 0):.4f}")
        print(f"  PSNR: {best_result.get('psnr', 0):.4f}")
        print(f"  RMSE: {best_result.get('rmse', 0):.4f}")

        print("Parameter information:")
        if 'threshold' in best_result:
            print(f"  Threshold: {best_result['threshold']:.4f}")
        if 'filter_sigma' in best_result:
            print(f"  Filter parameter: {best_result['filter_sigma']:.4f}")
        if 'use_weighted_avg' in best_result:
            print(f"  Weighted average: {'Yes' if best_result['use_weighted_avg'] else 'No'}")
        if 'edge_factor' in best_result:
            print(f"  Edge enhancement factor: {best_result['edge_factor']:.4f}")
        if 'edge_method' in best_result:
            print(f"  Edge detection method: {best_result['edge_method']}")
        if 'iterations' in best_result:
            print(f"  Iterations: {best_result['iterations']}")

        return best_method
    else:
        print("Not enough data for comparison")
        return None


def run_comprehensive_comparison(image_path=None, batch_mode=False, save_results=True):
    """
    Runs a comprehensive algorithm comparison analysis.

    Parameters
    -----------
    image_path : str, optional
        Image file path; if None, the user is prompted to choose one
    batch_mode : bool
        Whether to run in batch mode with preset parameters without prompting the user
    save_results : bool
        Whether to save the results

    Returns
    --------
    results : dict
        Dictionary of results for all algorithms
    metrics_df : DataFrame
        Evaluation metrics data frame
    """
    print("===== Comprehensive Ghost Imaging Algorithm Comparison Analysis =====")

    # Select the image file
    if image_path is None:
        print("\nStep 1: Select test image")
        try:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
            )

            if not image_path:
                print("No image file selected, using default path: examples/sample_images/sample.jpg")
                image_path = 'examples/sample_images/sample.jpg'
            else:
                print(f"Selected image: {image_path}")
        except:
            print("GUI selector unavailable, please enter the image path:")
            image_path = input().strip() or 'examples/sample_images/sample.jpg'

    # Load the image mask
    print(f"Loading and preprocessing image: {image_path}")
    object_mask = load_and_preprocess_image(image_path)
    if object_mask is None:
        print("Image loading failed, exiting program.")
        return None, None

    # Set parameters
    if batch_mode:
        # Preset parameters for batch run
        num_frames = 2000
        speckle_size = 2.5
        thresholds = [0.4, 0.5, 0.6]
        filter_sigmas = [1.0, 2.0, 2.5]
        use_weighted_avgs = [True]
        include_edge_enhancement = True
        edge_factors = [0.8, 1.0, 1.2]
        edge_methods = ['sobel', 'laplacian']
        include_iterative = True
        iterations_list = [2, 3, 4]
    else:
        # Interactively prompt the user to set parameters
        print("\nStep 2: Set algorithm parameters")

        frames_input = input("Enter number of simulated frames (default: 2000): ").strip()
        num_frames = int(
            frames_input) if frames_input and frames_input.isdigit() else 2000

        speckle_input = input("Enter speckle size (default: 2.5): ").strip()
        speckle_size = float(speckle_input) if speckle_input and speckle_input.replace(
            '.', '', 1).isdigit() else 2.5

        # TCDGI algorithm parameters
        print("\nSet TCDGI algorithm parameters:")

        threshold_input = input("Enter threshold list (comma-separated, default: 0.4,0.5,0.6): ").strip()
        if threshold_input:
            try:
                thresholds = [float(t.strip())
                              for t in threshold_input.split(',')]
            except:
                thresholds = [0.4, 0.5, 0.6]
        else:
            thresholds = [0.4, 0.5, 0.6]

        sigma_input = input("Enter filter parameter list (comma-separated, default: 1.0,2.0,2.5): ").strip()
        if sigma_input:
            try:
                filter_sigmas = [float(s.strip())
                                 for s in sigma_input.split(',')]
            except:
                filter_sigmas = [1.0, 2.0, 2.5]
        else:
            filter_sigmas = [1.0, 2.0, 2.5]

        weighted_input = input("Use weighted average? (y/n, default: y): ").strip().lower()
        use_weighted_avgs = [True] if weighted_input != 'n' else [False]

        # Edge enhancement
        edge_input = input("Include edge enhancement? (y/n, default: y): ").strip().lower()
        include_edge_enhancement = edge_input != 'n'

        if include_edge_enhancement:
            edge_factor_input = input(
                "Enter edge enhancement factor list (comma-separated, default: 0.8,1.0,1.2): ").strip()
            if edge_factor_input:
                try:
                    edge_factors = [float(e.strip())
                                    for e in edge_factor_input.split(',')]
                except:
                    edge_factors = [0.8, 1.0, 1.2]
            else:
                edge_factors = [0.8, 1.0, 1.2]

            print("Select edge detection methods (comma-separated):")
            print("1. Sobel operator")
            print("2. Laplacian operator")
            print("3. Prewitt operator")

            edge_method_input = input("Select methods (e.g. 1,2, default: 1,2): ").strip()

            edge_methods = []
            if edge_method_input:
                methods = edge_method_input.split(',')
                for m in methods:
                    if m.strip() == '1':
                        edge_methods.append('sobel')
                    elif m.strip() == '2':
                        edge_methods.append('laplacian')
                    elif m.strip() == '3':
                        edge_methods.append('prewitt')

            if not edge_methods:
                edge_methods = ['sobel', 'laplacian']
        else:
            edge_factors = [1.0]
            edge_methods = ['sobel']

        # Iterative optimization
        iter_input = input("Include iterative optimization? (y/n, default: y): ").strip().lower()
        include_iterative = iter_input != 'n'

        if include_iterative:
            iterations_input = input("Enter iteration count list (comma-separated, default: 2,3,4): ").strip()
            if iterations_input:
                try:
                    iterations_list = [int(i.strip())
                                       for i in iterations_input.split(',')]
                except:
                    iterations_list = [2, 3, 4]
            else:
                iterations_list = [2, 3, 4]
        else:
            iterations_list = [3]

    # Create the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f'comparison_results_{timestamp}')
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created results directory: {output_dir}")

    # Simulate the ghost imaging experiment
    print(f"\nSimulating ghost imaging experiment, frames={num_frames}, speckle size={speckle_size}...")
    reference_frames, bucket_signals = simulate_ghost_imaging(
        object_mask, num_frames=num_frames, speckle_size=speckle_size)

    # Run the comprehensive comparison
    print("\nStarting algorithm comparison...")
    results, metrics_df = advanced_compare_imaging_methods(
        reference_frames=reference_frames,
        bucket_signals=bucket_signals,
        object_mask=object_mask,
        thresholds=thresholds,
        filter_sigmas=filter_sigmas,
        use_weighted_avgs=use_weighted_avgs,
        include_edge_enhancement=include_edge_enhancement,
        edge_factors=edge_factors,
        edge_methods=edge_methods,
        include_iterative=include_iterative,
        iterations_list=iterations_list,
        save_results=save_results,
        output_dir=output_dir
    )

    # Visualize the comparison results
    print("\nGenerating comparison visualizations...")
    best_method = visualize_comparison_results(results, metrics_df)

    # Save the summary report
    if save_results:
        with open(os.path.join(output_dir, f"summary_report_{timestamp}.txt"), 'w', encoding='utf-8') as f:
            f.write("Comprehensive Ghost Imaging Algorithm Comparison Analysis Report\n")
            f.write("========================\n\n")
            f.write(f"Analysis time: {timestamp}\n")
            f.write(f"Test image: {image_path}\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Speckle size: {speckle_size}\n\n")

            f.write("Test parameters:\n")
            f.write(f"  Threshold list: {thresholds}\n")
            f.write(f"  Filter parameter list: {filter_sigmas}\n")
            f.write(f"  Weighted average: {'Yes' if use_weighted_avgs[0] else 'No'}\n")
            f.write(f"  Edge enhancement: {'Yes' if include_edge_enhancement else 'No'}\n")
            if include_edge_enhancement:
                f.write(f"  Edge enhancement factors: {edge_factors}\n")
                f.write(f"  Edge detection methods: {edge_methods}\n")
            f.write(f"  Iterative optimization: {'Yes' if include_iterative else 'No'}\n")
            if include_iterative:
                f.write(f"  Iterations: {iterations_list}\n\n")

            f.write("Test results summary:\n")
            f.write(f"  Total methods tested: {len(metrics_df)}\n")
            if best_method:
                f.write(f"  Best method: {best_method}\n")
                best_result = results[best_method]
                f.write(f"  Best SSIM: {best_result.get('ssim', 0):.4f}\n")
                f.write(f"  Best SNR: {best_result.get('snr', 0):.4f}\n")
                f.write(f"  Best PSNR: {best_result.get('psnr', 0):.4f}\n\n")

            f.write("Performance ranking of all methods (sorted by SNR):\n")
            sorted_df = metrics_df.sort_values(
                by='SNR', ascending=False).reset_index(drop=True)
            for i, row in sorted_df.iterrows():
                f.write(
                    f"  {i+1}. {row['Method']} - SSIM:{row['SSIM']:.4f}, SNR:{row['SNR']:.4f}, PSNR:{row['PSNR']:.4f}\n")

        print(
            f"\nAnalysis summary saved to: {os.path.join(output_dir, f'summary_report_{timestamp}.txt')}")

    print("\nGhost imaging algorithm comparison analysis complete!")

    return results, metrics_df


def analyze_parameter_sensitivity(reference_frames, bucket_signals, object_mask,
                                  base_threshold=0.5, base_sigma=1.0, base_weighted=True,
                                  threshold_range=(0.1, 1.0), threshold_steps=10,
                                  sigma_range=(0.1, 5.0), sigma_steps=10,
                                  save_results=False, output_dir=None):
    """
    Analyzes the sensitivity of the TCDGI algorithm to parameter changes.

    Parameters
    -----------
    reference_frames : ndarray
        Array of reference detector frames
    bucket_signals : ndarray
        Array of bucket detector signals
    object_mask : ndarray
        Original object mask
    base_threshold, base_sigma, base_weighted : float, float, bool
        Baseline parameter settings
    threshold_range, threshold_steps : tuple, int
        Threshold variation range and number of steps
    sigma_range, sigma_steps : tuple, int
        Filter parameter variation range and number of steps
    save_results : bool
        Whether to save the results
    output_dir : str
        Output directory path

    Returns
    --------
    sensitivity_results : dict
        Dictionary containing the parameter sensitivity analysis results
    """
    print("Starting parameter sensitivity analysis...")

    if save_results and output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.getcwd(), f'sensitivity_analysis_{timestamp}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created results directory: {output_dir}")

    # Generate parameter lists
    thresholds = np.linspace(
        threshold_range[0], threshold_range[1], threshold_steps)
    sigmas = np.linspace(sigma_range[0], sigma_range[1], sigma_steps)

    # Result storage
    threshold_results = {'thresholds': thresholds,
                         'ssim': [], 'snr': [], 'psnr': [], 'rmse': []}
    sigma_results = {'sigmas': sigmas, 'ssim': [],
                     'snr': [], 'psnr': [], 'rmse': []}
    weighted_results = {}

    # Vary the threshold while keeping other parameters fixed
    print("Analyzing threshold sensitivity...")
    for t in thresholds:
        print(f"  Testing threshold: {t:.3f}")
        image = tcdgi(
            reference_frames,
            bucket_signals,
            threshold=t,
            filter_sigma=base_sigma,
            use_weighted_avg=base_weighted
        )

        if image is not None:
            metrics = calculate_quality_metrics(
                object_mask, image, norm_type='centered')
            for key, value in metrics.items():
                threshold_results[key].append(value)
        else:
            # If some parameters fail to produce a valid image, use zero values
            for key in ['ssim', 'snr', 'psnr', 'rmse']:
                threshold_results[key].append(0)

    # Vary the filter parameter while keeping other parameters fixed
    print("Analyzing filter parameter sensitivity...")
    for s in sigmas:
        print(f"  Testing filter parameter: {s:.3f}")
        image = tcdgi(
            reference_frames,
            bucket_signals,
            threshold=base_threshold,
            filter_sigma=s,
            use_weighted_avg=base_weighted
        )

        if image is not None:
            metrics = calculate_quality_metrics(
                object_mask, image, norm_type='centered')
            for key, value in metrics.items():
                sigma_results[key].append(value)
        else:
            for key in ['ssim', 'snr', 'psnr', 'rmse']:
                sigma_results[key].append(0)

    # Comparison between weighted and unweighted
    print("Analyzing the difference between weighted and unweighted...")
    for weighted in [True, False]:
        weighted_key = "weighted" if weighted else "unweighted"
        weighted_results[weighted_key] = {}

        image = tcdgi(
            reference_frames,
            bucket_signals,
            threshold=base_threshold,
            filter_sigma=base_sigma,
            use_weighted_avg=weighted
        )

        if image is not None:
            metrics = calculate_quality_metrics(
                object_mask, image, norm_type='centered')
            weighted_results[weighted_key].update(metrics)
            weighted_results[weighted_key]['image'] = image
        else:
            # If a valid image cannot be generated, record zero values
            weighted_results[weighted_key] = {
                'ssim': 0, 'snr': 0, 'psnr': 0, 'rmse': 0
            }

    # Combine the results
    sensitivity_results = {
        'threshold': threshold_results,
        'sigma': sigma_results,
        'weighted': weighted_results
    }

    # Visualize the results

    # Threshold sensitivity curves
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(
        thresholds, threshold_results['ssim'], 'o-', color='blue', linewidth=2)
    plt.title('Effect of Threshold on SSIM', fontsize=12)
    plt.xlabel('Threshold')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(thresholds, threshold_results['snr'],
             'o-', color='green', linewidth=2)
    plt.title('Effect of Threshold on SNR', fontsize=12)
    plt.xlabel('Threshold')
    plt.ylabel('SNR')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(
        thresholds, threshold_results['psnr'], 'o-', color='red', linewidth=2)
    plt.title('Effect of Threshold on PSNR', fontsize=12)
    plt.xlabel('Threshold')
    plt.ylabel('PSNR')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(
        thresholds, threshold_results['rmse'], 'o-', color='purple', linewidth=2)
    plt.title('Effect of Threshold on RMSE', fontsize=12)
    plt.xlabel('Threshold')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Threshold Sensitivity Analysis', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.9)

    if save_results:
        plt.savefig(os.path.join(
            output_dir, 'threshold_sensitivity.png'), dpi=300)
    plt.show()

    # Filter parameter sensitivity curves
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.plot(sigmas, sigma_results['ssim'], 'o-', color='blue', linewidth=2)
    plt.title('Effect of Filter Parameter on SSIM', fontsize=12)
    plt.xlabel('Filter Parameter')
    plt.ylabel('SSIM')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(sigmas, sigma_results['snr'], 'o-', color='green', linewidth=2)
    plt.title('Effect of Filter Parameter on SNR', fontsize=12)
    plt.xlabel('Filter Parameter')
    plt.ylabel('SNR')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(sigmas, sigma_results['psnr'], 'o-', color='red', linewidth=2)
    plt.title('Effect of Filter Parameter on PSNR', fontsize=12)
    plt.xlabel('Filter Parameter')
    plt.ylabel('PSNR')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(sigmas, sigma_results['rmse'], 'o-', color='purple', linewidth=2)
    plt.title('Effect of Filter Parameter on RMSE', fontsize=12)
    plt.xlabel('Filter Parameter')
    plt.ylabel('RMSE')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Filter Parameter Sensitivity Analysis', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.9)

    if save_results:
        plt.savefig(os.path.join(output_dir, 'sigma_sensitivity.png'), dpi=300)
    plt.show()

    # Weighted comparison
    plt.figure(figsize=(15, 6))

    w_metrics = ['ssim', 'snr', 'psnr', 'rmse']
    w_values = [
        [weighted_results['weighted'][m], weighted_results['unweighted'][m]]
        for m in w_metrics
    ]

    plt.subplot(1, 2, 1)
    x = np.arange(len(w_metrics))
    width = 0.35
    plt.bar(x - width/2, [w[0] for w in w_values], width, label='Weighted Average')
    plt.bar(x + width/2, [w[1] for w in w_values], width, label='Simple Average')
    plt.xticks(x, w_metrics)
    plt.ylabel('Metric Value')
    plt.title('Comparison of Weighted vs Unweighted Averaging')
    plt.legend()

    # Show the weighted and unweighted images
    if 'image' in weighted_results['weighted'] and 'image' in weighted_results['unweighted']:
        plt.subplot(1, 2, 2)

        # Show the image difference
        diff_image = np.abs(
            weighted_results['weighted']['image'] - weighted_results['unweighted']['image'])
        plt.imshow(normalize_image(diff_image,
                   norm_type='centered'), cmap='hot')
        plt.colorbar(label='Difference Magnitude')
        plt.title('Difference Between Weighted and Unweighted Results')
        plt.axis('off')

    plt.tight_layout()

    if save_results:
        plt.savefig(os.path.join(
            output_dir, 'weighted_comparison.png'), dpi=300)
    plt.show()

    # Save the data
    if save_results:
        # Save results as CSV
        threshold_df = pd.DataFrame({
            'threshold': thresholds,
            'ssim': threshold_results['ssim'],
            'snr': threshold_results['snr'],
            'psnr': threshold_results['psnr'],
            'rmse': threshold_results['rmse']
        })
        threshold_df.to_csv(os.path.join(
            output_dir, 'threshold_sensitivity.csv'), index=False)

        sigma_df = pd.DataFrame({
            'sigma': sigmas,
            'ssim': sigma_results['ssim'],
            'snr': sigma_results['snr'],
            'psnr': sigma_results['psnr'],
            'rmse': sigma_results['rmse']
        })
        sigma_df.to_csv(os.path.join(
            output_dir, 'sigma_sensitivity.csv'), index=False)

        # Save the complete results
        with open(os.path.join(output_dir, 'sensitivity_results.pkl'), 'wb') as f:
            pickle.dump(sensitivity_results, f)

        print(f"Sensitivity analysis results saved to: {output_dir}")

    # Print the best parameters
    best_threshold_idx = np.argmax(threshold_results['snr'])
    best_sigma_idx = np.argmax(sigma_results['snr'])

    print("\nSensitivity analysis results summary:")
    print(
        f"Best threshold by SNR: {thresholds[best_threshold_idx]:.4f} (SNR: {threshold_results['snr'][best_threshold_idx]:.4f})")
    print(
        f"Best filter parameter by SNR: {sigmas[best_sigma_idx]:.4f} (SNR: {sigma_results['snr'][best_sigma_idx]:.4f})")
    print(
        f"Weighted average outperforms unweighted average: {'Yes' if weighted_results['weighted']['snr'] > weighted_results['unweighted']['snr'] else 'No'}")
    print(f"  Weighted SNR: {weighted_results['weighted']['snr']:.4f}")
    print(f"  Unweighted SNR: {weighted_results['unweighted']['snr']:.4f}")

    return sensitivity_results


def compare_encryption_systems(reference_frames, bucket_signals, object_mask,
                               thresholds=[0.5], filter_sigmas=[1.0],
                               use_weighted_avgs=[True],
                               perform_security_analysis=True,
                               perform_sensitivity_test=True,
                               save_results=False, output_dir=None):
    """
    Compares the performance of different ghost imaging methods as encryption systems.

    Parameters
    -----------
    reference_frames : ndarray
        Array of reference detector frames
    bucket_signals : ndarray
        Array of bucket detector signals
    object_mask : ndarray
        Original object mask
    thresholds : list
        List of TCDGI thresholds
    filter_sigmas : list
        List of Gaussian filter standard deviations
    use_weighted_avgs : list
        List of booleans indicating whether to use weighted averaging
    perform_security_analysis : bool
        Whether to perform security analysis
    perform_sensitivity_test : bool
        Whether to perform sensitivity testing
    save_results : bool
        Whether to save the results
    output_dir : str
        Output directory path

    Returns
    --------
    crypto_results : dict
        Dictionary containing the encryption system performance evaluation
    """
    # Create the results directory
    if save_results and output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.getcwd(), f'crypto_comparison_{timestamp}')

    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created results directory: {output_dir}")

    # Initialize the results dictionary
    crypto_results = {}

    # Create and evaluate the encryption system for each method
    methods = ['GI', 'DGI', 'TCDGI']

    for method in methods:
        print(f"Evaluating {method} encryption system...")

        # For TCDGI, we need to test multiple parameter combinations
        if method == 'TCDGI':
            for threshold in thresholds:
                for sigma in filter_sigmas:
                    for use_weighted_avg in use_weighted_avgs:
                        param_desc = f'T{threshold:.2f}_S{sigma:.1f}{"_W" if use_weighted_avg else "_NW"}'
                        method_key = f'{method}_{param_desc}'

                        # Create the encryption system
                        crypto_system = create_encryption_system(
                            method=method,
                            reference_frames=reference_frames,
                            bucket_signals=bucket_signals,
                            threshold=threshold,
                            filter_sigma=sigma,
                            use_weighted_avg=use_weighted_avg
                        )

                        # Run the encryption and decryption test
                        result = evaluate_encryption_system(
                            crypto_system=crypto_system,
                            original_image=object_mask,
                            perform_security_analysis=perform_security_analysis,
                            perform_sensitivity_test=perform_sensitivity_test,
                            save_results=save_results,
                            output_dir=output_dir,
                            method_name=method_key
                        )

                        crypto_results[method_key] = result
        else:
            # For GI and DGI, create and test directly
            crypto_system = create_encryption_system(
                method=method,
                reference_frames=reference_frames,
                bucket_signals=bucket_signals
            )

            result = evaluate_encryption_system(
                crypto_system=crypto_system,
                original_image=object_mask,
                perform_security_analysis=perform_security_analysis,
                perform_sensitivity_test=perform_sensitivity_test,
                save_results=save_results,
                output_dir=output_dir,
                method_name=method
            )

            crypto_results[method] = result

    # Visualize the comparison results
    visualize_encryption_comparison(
        crypto_results, output_dir=output_dir if save_results else None)

    return crypto_results


def create_encryption_system(method, reference_frames, bucket_signals,
                             threshold=0.5, filter_sigma=1.0, use_weighted_avg=True):
    """
    Creates an encryption system based on the specified method.

    Parameters
    -----------
    method : str
        Encryption method ('GI', 'DGI', 'TCDGI')
    reference_frames, bucket_signals : ndarray
        Reference frames and bucket signals
    threshold, filter_sigma, use_weighted_avg :
        TCDGI-specific parameters

    Returns
    --------
    crypto_system : object
        Encryption system object
    """
    # For each method, we create a simplified encryption system
    class SimpleCryptoSystem:
        def __init__(self, method, reference_frames, bucket_signals, **kwargs):
            self.method = method
            self.reference_frames = reference_frames
            self.bucket_signals = bucket_signals
            self.params = kwargs

        def encrypt(self, image):
            """Simulates the encryption process and returns the encrypted data (bucket signals)"""
            # In a real scenario, the encrypted data is the bucket signal and the key is the reference frames
            # Here we simply return the existing bucket signals as the encrypted data
            return {'bucket_signals': self.bucket_signals,
                    'image_shape': image.shape}

        def decrypt(self, encrypted_data):
            """Decrypts the image"""
            if self.method == 'GI':
                return self._decrypt_gi(encrypted_data)
            elif self.method == 'DGI':
                return self._decrypt_dgi(encrypted_data)
            elif self.method == 'TCDGI':
                return self._decrypt_tcdgi(encrypted_data)

        def _decrypt_gi(self, encrypted_data):
            """Decrypts using the GI algorithm"""
            bucket_signals = encrypted_data['bucket_signals']
            bucket_fluctuations = bucket_signals - np.mean(bucket_signals)
            gi_image = np.zeros_like(self.reference_frames[0])

            for i in range(len(self.reference_frames)):
                gi_image += bucket_fluctuations[i] * self.reference_frames[i]
            gi_image /= len(self.reference_frames)

            return normalize_image(gi_image, norm_type='centered')

        def _decrypt_dgi(self, encrypted_data):
            """Decrypts using the DGI algorithm"""
            bucket_signals = encrypted_data['bucket_signals']
            ref_integrated = np.sum(self.reference_frames, axis=(1, 2))
            ref_integrated_mean = np.mean(ref_integrated)
            bucket_mean = np.mean(bucket_signals)
            differential_bucket = bucket_signals - \
                bucket_mean * ref_integrated / ref_integrated_mean

            dgi_image = np.zeros_like(self.reference_frames[0])
            for i in range(len(self.reference_frames)):
                dgi_image += differential_bucket[i] * self.reference_frames[i]
            dgi_image /= len(self.reference_frames)

            return normalize_image((dgi_image), norm_type='centered')

        def _decrypt_tcdgi(self, encrypted_data):
            """Decrypts using the TCDGI algorithm"""
            bucket_signals = encrypted_data['bucket_signals']
            # Call the external TCDGI function
            tcdgi_image = tcdgi(
                self.reference_frames,
                bucket_signals,
                threshold=self.params.get('threshold', 0.5),
                filter_sigma=self.params.get('filter_sigma', 1.0),
                use_weighted_avg=self.params.get('use_weighted_avg', True)
            )

            if tcdgi_image is None:
                # If TCDGI fails, fall back to DGI
                return self._decrypt_dgi(encrypted_data)

            return normalize_image((tcdgi_image), norm_type='centered')

    # Create the corresponding encryption system based on the method
    params = {}
    if method == 'TCDGI':
        params = {
            'threshold': threshold,
            'filter_sigma': filter_sigma,
            'use_weighted_avg': use_weighted_avg
        }

    return SimpleCryptoSystem(method, reference_frames, bucket_signals, **params)


def evaluate_encryption_system(crypto_system, original_image,
                               perform_security_analysis=True,
                               perform_sensitivity_test=True,
                               save_results=False, output_dir=None,
                               method_name=""):
    """
    Evaluates the performance of the encryption system.

    Parameters
    -----------
    crypto_system : object
        Encryption system object
    original_image : ndarray
        Original image
    perform_security_analysis : bool
        Whether to perform security analysis
    perform_sensitivity_test : bool
        Whether to perform sensitivity testing
    save_results : bool
        Whether to save the results
    output_dir : str
        Output directory
    method_name : str
        Method name (used for output)

    Returns
    --------
    result : dict
        Dictionary containing the evaluation results
    """
    result = {
        'method': method_name,
        'encryption_metrics': {},
        'decryption_metrics': {},
        'security_metrics': {},
        'sensitivity_metrics': {}
    }

    # Encryption process
    print(f"Encrypting with {method_name}...")
    encrypted_data = crypto_system.encrypt(original_image)

    # Decryption process
    print(f"Decrypting with {method_name}...")
    decrypted_image = crypto_system.decrypt(encrypted_data)

    # Compute basic metrics
    correlation = np.corrcoef(original_image.flatten(),
                              decrypted_image.flatten())[0, 1]
    quality_metrics = calculate_quality_metrics(
        original_image, decrypted_image, norm_type='centered')

    result['decryption_metrics'] = {
        'correlation': correlation,
        **quality_metrics
    }

    print(
        f"{method_name} decryption quality metrics: correlation={correlation:.4f}, SSIM={quality_metrics['ssim']:.4f}, SNR={quality_metrics['snr']:.4f}")

    # If security analysis needs to be performed
    if perform_security_analysis:
        print(f"Performing {method_name} security analysis...")
        security_metrics = analyze_encryption_security(
            crypto_system, original_image)
        result['security_metrics'] = security_metrics

    # If sensitivity testing needs to be performed
    if perform_sensitivity_test:
        print(f"Performing {method_name} sensitivity test...")
        sensitivity_metrics = test_key_sensitivity(
            crypto_system, original_image)
        result['sensitivity_metrics'] = sensitivity_metrics

    # Save the results
    if save_results and output_dir:
        save_path = os.path.join(output_dir, f"{method_name}_results.png")
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(132)
        # Visualize the encrypted data (bucket signal)
        plt.plot(encrypted_data['bucket_signals'])
        plt.title(f'{method_name} Encrypted Data')
        plt.xlabel('Frame Index')
        plt.ylabel('Signal Intensity')

        plt.subplot(133)
        plt.imshow(decrypted_image, cmap='gray')
        plt.title(f'{method_name} Decrypted Image\nCorrelation={correlation:.4f}')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Results saved to: {save_path}")

    return result


def analyze_encryption_security(crypto_system, original_image):
    """
    Analyzes the security of the encryption system.

    Parameters
    -----------
    crypto_system : object
        Encryption system object
    original_image : ndarray
        Original image

    Returns
    --------
    security_metrics : dict
        Security metrics
    """
    security_metrics = {}

    # Encrypt the original image
    encrypted_data = crypto_system.encrypt(original_image)

    # 1. We do not compute the correlation between the original image and the bucket signal
    # because they have different lengths. Instead, compute statistical properties of the bucket signal
    bucket_signals = encrypted_data['bucket_signals']

    # Compute the mean and standard deviation of the bucket signal
    mean_value = np.mean(bucket_signals)
    std_value = np.std(bucket_signals)
    security_metrics['bucket_signal_mean'] = mean_value
    security_metrics['bucket_signal_std'] = std_value

    # Histogram analysis - check the distribution uniformity of the bucket signal
    hist, bin_edges = np.histogram(bucket_signals, bins=20)
    normalized_hist = hist / np.sum(hist)
    # Compute the ideal entropy of a uniform distribution
    ideal_entropy = np.log2(len(hist))
    # Compute the actual entropy
    actual_entropy = -np.sum(normalized_hist *
                             np.log2(normalized_hist + 1e-10))
    # Entropy ratio (closer to 1 means more random)
    entropy_ratio = actual_entropy / ideal_entropy

    security_metrics['bucket_signal_entropy'] = actual_entropy
    security_metrics['entropy_ratio'] = entropy_ratio
    security_metrics['distribution_uniformity'] = "High" if entropy_ratio > 0.9 else "Medium" if entropy_ratio > 0.7 else "Low"

    # 2. Estimate the theoretical key space size
    # Assuming the reference frames are 32-bit floats, compute the total number of bits
    # Note: this is a very loose upper-bound estimate (assuming every bit of every float
    # is independent and uniform), not the effective key space reported in the paper.
    # A more conservative, entropy-corrected estimate is given by analyze_key_space()
    # in cryptosystem.py.
    ref_frames_shape = crypto_system.reference_frames.shape
    key_space_bits = ref_frames_shape[0] * \
        ref_frames_shape[1] * ref_frames_shape[2] * 32
    security_metrics['key_space_bits'] = key_space_bits
    security_metrics['key_space'] = f"2^{key_space_bits}"

    # 3. Theoretical estimate of resistance to different attacks
    security_metrics['brute_force_resistance'] = "High" if key_space_bits > 128 else "Medium" if key_space_bits > 64 else "Low"
    security_metrics['statistical_attack_resistance'] = "High" if entropy_ratio > 0.9 else "Medium" if entropy_ratio > 0.7 else "Low"

    return security_metrics


def test_key_sensitivity(crypto_system, original_image):
    """
    Tests the sensitivity of the encryption system to small changes in the key.

    Parameters
    -----------
    crypto_system : object
        Encryption system object
    original_image : ndarray
        Original image

    Returns
    --------
    sensitivity_metrics : dict
        Sensitivity metrics
    """
    sensitivity_metrics = {}

    # Encrypt the original image
    encrypted_data = crypto_system.encrypt(original_image)

    # Decrypt using the correct key
    correct_decryption = crypto_system.decrypt(encrypted_data)

    # Create a copy of the key with a small modification
    modified_system = copy.deepcopy(crypto_system)
    # Add small noise to the reference frames
    noise_level = 0.01
    modified_system.reference_frames += np.random.normal(
        0, noise_level, modified_system.reference_frames.shape)

    # Decrypt using the modified key
    try:
        modified_decryption = modified_system.decrypt(encrypted_data)

        # Compute the correlation between the correct decryption and the modified-key decryption
        correct_correlation = np.corrcoef(
            original_image.flatten(), correct_decryption.flatten())[0, 1]
        modified_correlation = np.corrcoef(
            original_image.flatten(), modified_decryption.flatten())[0, 1]
        correlation_difference = np.abs(
            correct_correlation - modified_correlation)

        sensitivity_metrics['correct_correlation'] = correct_correlation
        sensitivity_metrics['modified_correlation'] = modified_correlation
        sensitivity_metrics['correlation_difference'] = correlation_difference

        # Compute the difference between the correct decryption and the modified-key decryption
        pixel_difference = np.mean(
            np.abs(correct_decryption - modified_decryption))
        sensitivity_metrics['pixel_difference'] = pixel_difference

        # Compute the avalanche effect
        avalanche_effect = pixel_difference / \
            np.mean(np.abs(correct_decryption))
        sensitivity_metrics['avalanche_effect'] = avalanche_effect

        # Determine the sensitivity level
        sensitivity_metrics['key_sensitivity_level'] = "High" if avalanche_effect > 0.5 else "Medium" if avalanche_effect > 0.2 else "Low"

    except Exception as e:
        # If decryption fails, the sensitivity is extremely high
        sensitivity_metrics['key_sensitivity_level'] = "Very High"
        sensitivity_metrics['error'] = str(e)

    return sensitivity_metrics


def visualize_encryption_comparison(crypto_results, output_dir=None):
    """
    Visualizes the encryption system comparison results.

    Parameters
    -----------
    crypto_results : dict
        Encryption system evaluation results
    output_dir : str
        Output directory; if None, images are not saved
    """
    # Extract the key metrics from the results
    methods = list(crypto_results.keys())

    # Decryption quality metrics
    correlations = [result['decryption_metrics'].get(
        'correlation', 0) for result in crypto_results.values()]
    ssim_values = [result['decryption_metrics'].get(
        'ssim', 0) for result in crypto_results.values()]
    snr_values = [result['decryption_metrics'].get(
        'snr', 0) for result in crypto_results.values()]

    # Security metrics
    entropy_values = [result.get('security_metrics', {}).get('bucket_signal_entropy', 0)
                      for result in crypto_results.values()]
    uniformity_scores = []
    for result in crypto_results.values():
        uniformity = result.get('security_metrics', {}).get(
            'distribution_uniformity', 'N/A')
        if uniformity == "High":
            uniformity_scores.append(3)
        elif uniformity == "Medium":
            uniformity_scores.append(2)
        elif uniformity == "Low":
            uniformity_scores.append(1)
        else:
            uniformity_scores.append(0)

    # Extract information from the security metrics
    security_levels = []
    sensitivity_levels = []

    for result in crypto_results.values():
        security = result.get('security_metrics', {})
        sensitivity = result.get('sensitivity_metrics', {})

        # Security score (simplified version)
        security_score = 0
        if 'brute_force_resistance' in security:
            if security['brute_force_resistance'] == "High":
                security_score += 2
            elif security['brute_force_resistance'] == "Medium":
                security_score += 1

        if 'statistical_attack_resistance' in security:
            if security['statistical_attack_resistance'] == "High":
                security_score += 2
            elif security['statistical_attack_resistance'] == "Medium":
                security_score += 1

        security_levels.append(security_score)

        # Sensitivity score (simplified version)
        sensitivity_score = 0
        if 'key_sensitivity_level' in sensitivity:
            if sensitivity['key_sensitivity_level'] == "Very High":
                sensitivity_score = 3
            elif sensitivity['key_sensitivity_level'] == "High":
                sensitivity_score = 2
            elif sensitivity['key_sensitivity_level'] == "Medium":
                sensitivity_score = 1

        sensitivity_levels.append(sensitivity_score)

    # Create the visualization
    plt.figure(figsize=(15, 10))

    # 1. Decryption quality metric comparison
    plt.subplot(2, 2, 1)
    x = np.arange(len(methods))
    width = 0.3

    plt.bar(x - width, correlations, width, label='Correlation Coefficient')
    plt.bar(x, ssim_values, width, label='SSIM')
    plt.bar(x + width, snr_values, width, label='SNR')

    plt.xlabel('Encryption Method')
    plt.ylabel('Metric Value')
    plt.title('Decryption Quality Metric Comparison')
    plt.xticks(x, methods, rotation=45)
    plt.legend()

    # 2. Security metric comparison
    plt.subplot(2, 2, 2)
    width = 0.4

    plt.bar(x - width/2, security_levels, width, label='Security Score')
    plt.bar(x + width/2, entropy_values, width, label='Signal Entropy')

    plt.xlabel('Encryption Method')
    plt.ylabel('Score / Entropy Value')
    plt.title('Security Metric Comparison')
    plt.xticks(x, methods, rotation=45)
    plt.legend()

    # 3. Comprehensive comparison radar chart
    plt.subplot(2, 2, 3, polar=True)

    # Normalize each metric
    max_correlation = max(max(correlations), 0.0001)
    max_ssim = max(max(ssim_values), 0.0001)
    max_snr = max(max(snr_values), 0.0001)
    max_security = max(max(security_levels), 0.0001)
    max_entropy = max(max(entropy_values), 0.0001)

    categories = ['Decryption Correlation', 'SSIM', 'SNR', 'Security', 'Entropy']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the radar chart

    ax = plt.gca()
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories)

    for i, method in enumerate(methods):
        values = [
            correlations[i] / max_correlation,
            ssim_values[i] / max_ssim,
            snr_values[i] / max_snr,
            security_levels[i] / max_security,
            entropy_values[i] / max_entropy
        ]
        values += values[:1]  # Close the radar chart

        plt.plot(angles, values, linewidth=1, linestyle='solid', label=method)
        plt.fill(angles, values, alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    # 4. Detailed comparison table
    plt.subplot(2, 2, 4)
    plt.axis('off')

    data = []
    for i, method in enumerate(methods):
        result = crypto_results[method]
        security = result.get('security_metrics', {})
        sensitivity = result.get('sensitivity_metrics', {})

        data.append([
            method,
            f"{correlations[i]:.4f}",
            f"{ssim_values[i]:.4f}",
            f"{snr_values[i]:.4f}",
            security.get('brute_force_resistance', 'N/A'),
            security.get('statistical_attack_resistance', 'N/A'),
            f"{entropy_values[i]:.2f}"
        ])

    column_labels = ['Method', 'Correlation', 'SSIM', 'SNR', 'Brute-Force Resistance', 'Statistical Resistance', 'Entropy']
    table = plt.table(
        cellText=data,
        colLabels=column_labels,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.title('Detailed Encryption Method Comparison', y=0.8)

    plt.tight_layout()
    plt.suptitle('Ghost Imaging Encryption System Comparison Analysis', fontsize=16, y=0.98)
    plt.subplots_adjust(top=0.9)

    # Save the image
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            output_dir, f"encryption_comparison_{timestamp}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Comparison results saved to: {save_path}")

    plt.show()


def run_encryption_comparison(image_path=None, batch_mode=False, save_results=True):
    """
    Runs the encryption system comparison analysis.

    Parameters
    -----------
    image_path : str, optional
        Image file path; if None, the user is prompted to choose one
    batch_mode : bool
        Whether to run in batch mode with preset parameters without prompting the user
    save_results : bool
        Whether to save the results

    Returns
    --------
    crypto_results : dict
        Encryption system comparison results
    """
    print("===== Ghost Imaging Encryption System Comparison Analysis =====")

    # Select the image file
    if image_path is None:
        print("\nStep 1: Select test image")
        try:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select Image File",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
            )

            if not image_path:
                print("No image file selected, using default path: examples/sample_images/sample.jpg")
                image_path = 'examples/sample_images/sample.jpg'
            else:
                print(f"Selected image: {image_path}")
        except:
            print("GUI selector unavailable, please enter the image path:")
            image_path = input().strip() or 'examples/sample_images/sample.jpg'

    # Load the image mask
    print(f"Loading and preprocessing image: {image_path}")
    object_mask = load_and_preprocess_image(image_path)
    if object_mask is None:
        print("Image loading failed, exiting program.")
        return None

    # Set parameters
    if batch_mode:
        # Preset parameters for batch run
        num_frames = 2000
        speckle_size = 2.5
        thresholds = [0.4, 0.5, 0.6]
        filter_sigmas = [1.0, 2.0]
        use_weighted_avgs = [True]
        perform_security_analysis = True
        perform_sensitivity_test = True
    else:
        # Interactively prompt the user to set parameters
        print("\nStep 2: Set algorithm parameters")

        frames_input = input("Enter number of simulated frames (default: 2000): ").strip()
        num_frames = int(
            frames_input) if frames_input and frames_input.isdigit() else 2000

        speckle_input = input("Enter speckle size (default: 2.5): ").strip()
        speckle_size = float(speckle_input) if speckle_input and speckle_input.replace(
            '.', '', 1).isdigit() else 2.5

        # TCDGI algorithm parameters
        print("\nSet TCDGI algorithm parameters:")

        threshold_input = input("Enter threshold list (comma-separated, default: 0.4,0.5,0.6): ").strip()
        if threshold_input:
            try:
                thresholds = [float(t.strip())
                              for t in threshold_input.split(',')]
            except:
                thresholds = [0.4, 0.5, 0.6]
        else:
            thresholds = [0.4, 0.5, 0.6]

        sigma_input = input("Enter filter parameter list (comma-separated, default: 1.0,2.0): ").strip()
        if sigma_input:
            try:
                filter_sigmas = [float(s.strip())
                                 for s in sigma_input.split(',')]
            except:
                filter_sigmas = [1.0, 2.0]
        else:
            filter_sigmas = [1.0, 2.0]

        weighted_input = input("Use weighted average? (y/n, default: y): ").strip().lower()
        use_weighted_avgs = [True] if weighted_input != 'n' else [False]

        # Security analysis option
        security_input = input("Perform security analysis? (y/n, default: y): ").strip().lower()
        perform_security_analysis = security_input != 'n'

        sensitivity_input = input("Perform sensitivity test? (y/n, default: y): ").strip().lower()
        perform_sensitivity_test = sensitivity_input != 'n'

    # Create the output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f'crypto_comparison_{timestamp}')
    if save_results and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created results directory: {output_dir}")

    # Simulate the ghost imaging experiment
    print(f"\nSimulating ghost imaging experiment, frames={num_frames}, speckle size={speckle_size}...")
    reference_frames, bucket_signals = simulate_ghost_imaging(
        object_mask, num_frames=num_frames, speckle_size=speckle_size)

    # Run the encryption system comparison
    print("\nStarting encryption system comparison...")
    crypto_results = compare_encryption_systems(
        reference_frames=reference_frames,
        bucket_signals=bucket_signals,
        object_mask=object_mask,
        thresholds=thresholds,
        filter_sigmas=filter_sigmas,
        use_weighted_avgs=use_weighted_avgs,
        perform_security_analysis=perform_security_analysis,
        perform_sensitivity_test=perform_sensitivity_test,
        save_results=save_results,
        output_dir=output_dir
    )

    # Save the summary report
    if save_results:
        with open(os.path.join(output_dir, f"crypto_summary_report_{timestamp}.txt"), 'w', encoding='utf-8') as f:
            f.write("Ghost Imaging Encryption System Comparison Analysis Report\n")
            f.write("========================\n\n")
            f.write(f"Analysis time: {timestamp}\n")
            f.write(f"Test image: {image_path}\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Speckle size: {speckle_size}\n\n")

            f.write("Test parameters:\n")
            f.write(f"  Threshold list: {thresholds}\n")
            f.write(f"  Filter parameter list: {filter_sigmas}\n")
            f.write(f"  Weighted average: {'Yes' if True in use_weighted_avgs else 'No'}\n")
            f.write(f"  Security analysis: {'Yes' if perform_security_analysis else 'No'}\n")
            f.write(f"  Sensitivity test: {'Yes' if perform_sensitivity_test else 'No'}\n\n")

            f.write("Test results summary:\n")
            f.write(f"  Total methods tested: {len(crypto_results)}\n\n")

            f.write("Per-method performance comparison:\n")
            for method, result in crypto_results.items():
                f.write(f"  {method}:\n")
                f.write(f"    Decryption quality:\n")
                for metric, value in result['decryption_metrics'].items():
                    if isinstance(value, (float, int)):
                        f.write(f"      {metric}: {value:.4f}\n")
                    else:
                        f.write(f"      {metric}: {value}\n")

                if 'security_metrics' in result and result['security_metrics']:
                    f.write(f"    Security metrics:\n")
                    for metric, value in result['security_metrics'].items():
                        f.write(f"      {metric}: {value}\n")

                if 'sensitivity_metrics' in result and result['sensitivity_metrics']:
                    f.write(f"    Sensitivity metrics:\n")
                    for metric, value in result['sensitivity_metrics'].items():
                        f.write(f"      {metric}: {value}\n")

                f.write("\n")

        print(
            f"\nAnalysis summary saved to: {os.path.join(output_dir, f'crypto_summary_report_{timestamp}.txt')}")

    print("\nGhost imaging encryption system comparison analysis complete!")

    return crypto_results


# Main function
if __name__ == "__main__":
    print("===== Detailed Ghost Imaging Algorithm Comparison Tool =====")

    while True:
        print("\nSelect an operation:")
        print("1. Run comprehensive algorithm comparison")
        print("2. Perform parameter sensitivity analysis")
        print("3. Run comparison in batch mode")
        print("4. Run encryption system comparison analysis")  # New option
        print("5. Exit")

        choice = input("Please choose (1-5): ").strip()

        if choice == '1':
            # Run the comprehensive algorithm comparison
            results, metrics_df = run_comprehensive_comparison()

        elif choice == '2':
            # Perform parameter sensitivity analysis
            # (original code unchanged)
            print("\nStep 1: Select test image")
            image_path = None
            try:
                from tkinter import Tk, filedialog
                root = Tk()
                root.withdraw()
                image_path = filedialog.askopenfilename(
                    title="Select Image File",
                    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
                )

                if not image_path:
                    print("No image file selected, using default path: examples/sample_images/sample.jpg")
                    image_path = 'examples/sample_images/sample.jpg'
                else:
                    print(f"Selected image: {image_path}")
            except:
                print("GUI selector unavailable, please enter the image path:")
                image_path = input().strip() or 'examples/sample_images/sample.jpg'

            # Load the image mask
            object_mask = load_and_preprocess_image(image_path)
            if object_mask is None:
                print("Image loading failed, skipping sensitivity analysis.")
                continue

            # Set basic parameters
            print("\nStep 2: Set basic parameters")
            frames_input = input("Enter number of simulated frames (default: 2000): ").strip()
            num_frames = int(
                frames_input) if frames_input and frames_input.isdigit() else 2000

            speckle_input = input("Enter speckle size (default: 2.5): ").strip()
            speckle_size = float(speckle_input) if speckle_input and speckle_input.replace(
                '.', '', 1).isdigit() else 2.5

            # Simulate the ghost imaging experiment
            print(f"\nSimulating ghost imaging experiment, frames={num_frames}, speckle size={speckle_size}...")
            reference_frames, bucket_signals = simulate_ghost_imaging(
                object_mask, num_frames=num_frames, speckle_size=speckle_size)

            # Set sensitivity analysis parameters
            threshold_input = input("Enter baseline threshold (default: 0.5): ").strip()
            base_threshold = float(threshold_input) if threshold_input and threshold_input.replace(
                '.', '', 1).isdigit() else 0.5

            sigma_input = input("Enter baseline filter parameter (default: 1.0): ").strip()
            base_sigma = float(sigma_input) if sigma_input and sigma_input.replace(
                '.', '', 1).isdigit() else 1.0

            weighted_input = input(
                "Use weighted average as the baseline? (y/n, default: y): ").strip().lower()
            base_weighted = weighted_input != 'n'

            # Sensitivity analysis range
            print("\nSet parameter variation range:")

            steps_input = input("Enter number of analysis steps (default: 10): ").strip()
            steps = int(
                steps_input) if steps_input and steps_input.isdigit() else 10

            save_input = input("Save analysis results? (y/n, default: y): ").strip().lower()
            save_results = save_input != 'n'

            # Run the sensitivity analysis
            sensitivity_results = analyze_parameter_sensitivity(
                reference_frames, bucket_signals, object_mask,
                base_threshold=base_threshold,
                base_sigma=base_sigma,
                base_weighted=base_weighted,
                threshold_range=(0.1, 1.0),
                threshold_steps=steps,
                sigma_range=(0.1, 5.0),
                sigma_steps=steps,
                save_results=save_results
            )

        elif choice == '3':
            # Run comparison in batch mode
            print("\nBatch mode will run the full comparison using preset parameters")
            path_input = input("Enter image path (default: examples/sample_images/sample.jpg): ").strip()
            image_path = path_input if path_input else 'examples/sample_images/sample.jpg'

            batch_results, batch_metrics = run_comprehensive_comparison(
                image_path=image_path,
                batch_mode=True,
                save_results=True
            )

        elif choice == '4':
            # Run the encryption system comparison analysis
            crypto_results = run_encryption_comparison()

        elif choice == '5':
            # Exit
            print("Exiting program.")
            break

        else:
            print("Invalid choice, please try again.")
