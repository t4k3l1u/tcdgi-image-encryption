import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
import cv2



def normalize_image(image, norm_type='centered'):
    """
    Normalize an image.

    Parameters
    -----------
    image : ndarray
        Input image
    norm_type : str, optional
        Normalization type, defaults to 'centered'

    Returns
    --------
    normalized_image : ndarray
        Normalized image
    """
    # Copy the image to avoid modifying the original data
    img = image.copy()

    if norm_type == 'standard':
        # Standard normalization to the [0, 1] range
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            return (img - img_min) / (img_max - img_min)
        return img

    elif norm_type == 'centered':
        # Centered normalization to the [-1, 1] range
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            return 2 * (img - img_min) / (img_max - img_min) - 1
        return img

    # Default to centered normalization
    return normalize_image(img, 'centered')


def evaluate_decryption_quality(original_image_path, decrypted_image, output_dir=None, prefix="quality", show_plots=True):
    """
    Enhanced quality evaluation of the decrypted image.

    Parameters
    -----------
    original_image_path : str
        Path to the original image
    decrypted_image : ndarray
        Decrypted image
    output_dir : str
        Output directory
    prefix : str
        Output file prefix
    show_plots : bool
        Whether to display the plots

    Returns
    --------
    metrics : dict
        Dictionary containing all quality evaluation metrics
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load the original image
    try:
        original_image = np.array(Image.open(
            original_image_path).convert('L')) / 255.0
        print(f"Original image size: {original_image.shape}")
    except Exception as e:
        print(f"Failed to load original image: {e}")
        return None

    # If the decrypted image and original image have different sizes, resize to match
    if original_image.shape != decrypted_image.shape:
        from skimage.transform import resize
        print(f"Resizing decrypted image from {decrypted_image.shape} to {original_image.shape}")
        decrypted_image = resize(decrypted_image, original_image.shape)

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(original_image_path)
    os.makedirs(output_dir, exist_ok=True)

    # Apply centered normalization
    original_image_norm = normalize_image(original_image)
    decrypted_image_norm = normalize_image(decrypted_image)

    # Compute basic evaluation metrics
    mse = np.mean((original_image_norm - decrypted_image_norm) ** 2)
    correlation = np.corrcoef(original_image_norm.flatten(),
                              decrypted_image_norm.flatten())[0, 1]
    ssim_value = ssim(original_image_norm, decrypted_image_norm,
                      data_range=2.0)  # Range is [-1, 1], so data_range=2.0
    psnr_value = psnr(original_image_norm,
                      decrypted_image_norm, data_range=2.0)

    # Additional evaluation metrics
    nrmse_value = nrmse(original_image_norm, decrypted_image_norm)
    uqi_value = universal_quality_index(
        original_image_norm, decrypted_image_norm)
    entropy_orig = image_entropy(original_image_norm)
    entropy_decrypt = image_entropy(decrypted_image_norm)
    entropy_diff = abs(entropy_orig - entropy_decrypt)

    # Compute edge preservation quality
    edge_quality = edge_preservation_quality(
        original_image_norm, decrypted_image_norm)

    # Compute local contrast
    contrast_measure = local_contrast(decrypted_image_norm)

    # Compute local region quality analysis
    region_quality = analyze_local_regions(
        original_image_norm, decrypted_image_norm)

    # Compute absolute error map
    error_map = np.abs(original_image_norm - decrypted_image_norm)

    # Compute Fourier-domain analysis
    fourier_diff = fourier_domain_analysis(
        original_image_norm, decrypted_image_norm)

    # Create visualization
    plt.figure(figsize=(20, 16))

    # Original image
    plt.subplot(3, 4, 1)
    plt.imshow(original_image_norm, cmap='gray', vmin=-1, vmax=1)
    plt.title('Original Image')
    plt.axis('off')

    # Decrypted image
    plt.subplot(3, 4, 2)
    plt.imshow(decrypted_image_norm, cmap='gray', vmin=-1, vmax=1)
    plt.title('Decrypted Image')
    plt.axis('off')

    # Error map
    plt.subplot(3, 4, 3)
    plt.imshow(error_map, cmap='hot')
    plt.colorbar(label='Error')
    plt.title('Absolute Error Map')
    plt.axis('off')

    # Edge map of the original image
    plt.subplot(3, 4, 4)
    orig_edges = detect_edges(original_image_norm)
    plt.imshow(orig_edges, cmap='gray')
    plt.title('Original Image Edges')
    plt.axis('off')

    # Edge map of the decrypted image
    plt.subplot(3, 4, 5)
    decrypt_edges = detect_edges(decrypted_image_norm)
    plt.imshow(decrypt_edges, cmap='gray')
    plt.title('Decrypted Image Edges')
    plt.axis('off')

    # Edge difference map
    plt.subplot(3, 4, 6)
    edge_diff = np.abs(orig_edges - decrypt_edges)
    plt.imshow(edge_diff, cmap='hot')
    plt.colorbar(label='Edge Difference')
    plt.title('Edge Preservation Evaluation')
    plt.axis('off')

    # Fourier-domain analysis map
    plt.subplot(3, 4, 7)
    plt.imshow(fourier_diff, cmap='viridis')
    plt.colorbar(label='Frequency Difference')
    plt.title('Fourier-Domain Analysis')
    plt.axis('off')

    # Error distribution histogram
    plt.subplot(3, 4, 8)
    plt.hist(error_map.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Error Distribution Histogram')
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    # Local region quality heatmap
    plt.subplot(3, 4, 9)
    plt.imshow(region_quality['quality_map'], cmap='jet')
    plt.colorbar(label='Local Quality')
    plt.title('Local Region Quality Evaluation')
    plt.axis('off')

    # Local contrast map
    plt.subplot(3, 4, 10)
    plt.imshow(contrast_measure, cmap='plasma')
    plt.colorbar(label='Contrast')
    plt.title('Local Contrast Analysis')
    plt.axis('off')

    # Quality metrics summary
    plt.subplot(3, 4, 11)
    plt.axis('off')

    quality_info = [
        "Decryption Quality Evaluation:",
        "="*30,
        f"Correlation Coefficient: {correlation:.6f}",
        f"Mean Squared Error (MSE): {mse:.6f}",
        f"Normalized Root Mean Squared Error (NRMSE): {nrmse_value:.6f}",
        f"Structural Similarity (SSIM): {ssim_value:.6f}",
        f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.2f} dB",
        f"Universal Quality Index (UQI): {uqi_value:.6f}",
        f"Edge Preservation Quality: {edge_quality:.6f}",
        f"Original Image Entropy: {entropy_orig:.4f}",
        f"Decrypted Image Entropy: {entropy_decrypt:.4f}",
        f"Entropy Difference: {entropy_diff:.4f}",
        f"Mean Local Region Quality: {region_quality['mean_quality']:.4f}",
        f"Max Error: {np.max(error_map):.6f}",
        f"Mean Error: {np.mean(error_map):.6f}",
    ]

    plt.text(0.05, 0.95, '\n'.join(quality_info), fontsize=10,
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.title('Image Quality Evaluation Metrics')

    # Add extra quality information
    plt.subplot(3, 4, 12)
    plt.axis('off')
    extra_info = [
        "Regional Quality Analysis:",
        "="*20
    ]

    # Add quality information for each region
    for region, quality in region_quality['regions'].items():
        extra_info.append(f"Region {region}: {quality:.4f}")

    plt.text(0.05, 0.95, '\n'.join(extra_info), fontsize=10,
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.title('Regional Quality Details')

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'{prefix}_quality_analysis_{timestamp}.png'), dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close()

    # Save evaluation results
    with open(os.path.join(output_dir, f'{prefix}_quality_metrics_{timestamp}.txt'), 'w') as f:
        f.write("Decrypted Image Quality Evaluation\n")
        f.write("="*40 + "\n\n")
        f.write('\n'.join(quality_info))
        f.write("\n\nLocal Region Quality Analysis:\n")
        for region, quality in region_quality['regions'].items():
            f.write(f"Region {region}: {quality:.6f}\n")

    # Summary of all metrics
    metrics = {
        'correlation': correlation,
        'mse': mse,
        'nrmse': nrmse_value,
        'ssim': ssim_value,
        'psnr': psnr_value,
        'uqi': uqi_value,
        'edge_quality': edge_quality,
        'entropy_orig': entropy_orig,
        'entropy_decrypt': entropy_decrypt,
        'entropy_diff': entropy_diff,
        'region_quality': region_quality['mean_quality'],
        'max_error': np.max(error_map),
        'mean_error': np.mean(error_map)
    }

    print(f"Quality evaluation complete")
    print(f"Evaluation results saved to: {output_dir}")

    return metrics


def universal_quality_index(x, y, window_size=8):
    """Compute the Universal Quality Index (UQI)"""
    N = window_size ** 2
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_var = np.var(x)
    y_var = np.var(y)
    xy_cov = np.mean((x - x_mean) * (y - y_mean))

    # Avoid division by zero
    if x_var * y_var == 0:
        return 0

    # Compute UQI
    numerator = 4 * xy_cov * x_mean * y_mean
    denominator = (x_var + y_var) * (x_mean**2 + y_mean**2)

    if denominator == 0:
        return 0

    return numerator / denominator


def image_entropy(img):
    """Compute the image entropy"""
    # First rescale the [-1,1] range back to [0,1] for entropy computation
    if np.min(img) < 0:
        scaled_img = (img + 1) / 2
    else:
        scaled_img = img

    # Compute the histogram of the normalized image
    hist = np.histogram(scaled_img, bins=256, range=(0, 1))[0]
    hist = hist / hist.sum()

    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return entropy


def detect_edges(img, method='canny'):
    """Detect image edges"""
    # Map the [-1,1] range to [0,255] for edge detection
    if np.min(img) < 0:
        img_uint8 = ((img + 1) / 2 * 255).astype(np.uint8)
    else:
        img_uint8 = (img * 255).astype(np.uint8)

    if method == 'canny':
        # Use Canny edge detection
        edges = cv2.Canny(img_uint8, 50, 150) / 255.0
    else:
        # Use the Sobel operator
        from scipy import ndimage
        dx = ndimage.sobel(img, axis=0)
        dy = ndimage.sobel(img, axis=1)
        edges = np.hypot(dx, dy)
        edges = edges / edges.max()

    return edges


def edge_preservation_quality(original, decrypted):
    """Edge preservation quality evaluation"""
    # Detect edges
    orig_edges = detect_edges(original)
    decrypt_edges = detect_edges(decrypted)

    # Compute edge preservation quality
    if np.sum(orig_edges) == 0:
        return 0

    edge_similarity = np.sum(orig_edges * decrypt_edges) / \
        (np.sum(orig_edges) + 1e-10)
    return edge_similarity


def local_contrast(img, window_size=15):
    """Compute local contrast"""
    from scipy.ndimage import uniform_filter

    # Compute local mean
    local_mean = uniform_filter(img, size=window_size)

    # Compute local variance
    local_var = uniform_filter(img**2, size=window_size) - local_mean**2

    # Compute local standard deviation
    local_std = np.sqrt(np.maximum(local_var, 0))

    # Local contrast = local std / local mean
    # Avoid division by zero
    local_contrast = np.zeros_like(local_mean)
    valid_mask = local_mean > 1e-6
    local_contrast[valid_mask] = local_std[valid_mask] / local_mean[valid_mask]

    return local_contrast


def analyze_local_regions(original, decrypted, num_regions=9):
    """Analyze the local region quality of the image"""
    h, w = original.shape
    region_h = h // 3
    region_w = w // 3

    quality_map = np.zeros_like(original)
    regions = {}

    for i in range(3):
        for j in range(3):
            region_idx = i * 3 + j + 1
            y_start = i * region_h
            y_end = (i + 1) * region_h if i < 2 else h
            x_start = j * region_w
            x_end = (j + 1) * region_w if j < 2 else w

            orig_region = original[y_start:y_end, x_start:x_end]
            decrypt_region = decrypted[y_start:y_end, x_start:x_end]

            # Compute local SSIM
            region_ssim = ssim(orig_region, decrypt_region,
                               data_range=2.0)  # Range is [-1,1], so data_range=2.0
            regions[f"{region_idx}"] = region_ssim

            # Fill in the quality map
            quality_map[y_start:y_end, x_start:x_end] = region_ssim

    return {
        'regions': regions,
        'mean_quality': np.mean(list(regions.values())),
        'quality_map': quality_map
    }


def fourier_domain_analysis(original, decrypted):
    """Fourier-domain analysis"""
    # Compute the Fourier transform
    f_orig = np.fft.fft2(original)
    f_orig_shift = np.fft.fftshift(f_orig)
    f_decrypt = np.fft.fft2(decrypted)
    f_decrypt_shift = np.fft.fftshift(f_decrypt)

    # Compute the magnitude spectrum
    mag_orig = np.abs(f_orig_shift)
    mag_decrypt = np.abs(f_decrypt_shift)

    # Compute the log magnitude spectrum
    mag_orig_log = np.log(mag_orig + 1)
    mag_decrypt_log = np.log(mag_decrypt + 1)

    # Compute the magnitude spectrum difference
    mag_diff = np.abs(mag_orig_log - mag_decrypt_log)
    mag_diff_norm = mag_diff / np.max(mag_diff)

    return mag_diff_norm


def compare_multiple_decryptions(original_image_path, decrypted_images, labels, output_dir=None, prefix="comparison"):
    """
    Compare multiple decryption results.

    Parameters
    -----------
    original_image_path : str
        Path to the original image
    decrypted_images : list
        List of decrypted images
    labels : list
        Label for each decrypted image
    output_dir : str
        Output directory
    prefix : str
        Output file prefix
    """
    # Ensure the number of images and labels match
    if len(decrypted_images) != len(labels):
        print("Error: the number of decrypted images does not match the number of labels!")
        return

    # Evaluate each decrypted image
    metrics_list = []
    for i, (image, label) in enumerate(zip(decrypted_images, labels)):
        print(f"\nEvaluating decryption quality for {label}...")
        metrics = evaluate_decryption_quality(
            original_image_path,
            image,
            output_dir=output_dir,
            prefix=f"{prefix}_{i+1}_{label}",
            show_plots=False
        )
        metrics['label'] = label
        metrics_list.append(metrics)

    # Generate comparison charts
    _generate_comparison_charts(metrics_list, output_dir, prefix)

    return metrics_list


def _generate_comparison_charts(metrics_list, output_dir, prefix):
    """Generate comparison charts"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels = [m['label'] for m in metrics_list]

    plt.figure(figsize=(15, 10))

    # Comparison of the main quality metrics
    plt.subplot(2, 2, 1)

    # Prepare data
    metrics_to_plot = ['ssim', 'correlation', 'edge_quality']
    metrics_labels = ['SSIM', 'Correlation', 'Edge Preservation']

    x = np.arange(len(labels))
    width = 0.25

    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] for m in metrics_list]
        plt.bar(x + i*width - width, values, width, label=metrics_labels[i])

    plt.xlabel('Decryption Method')
    plt.ylabel('Metric Value')
    plt.title('Comparison of Main Image Quality Metrics')
    plt.xticks(x, labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # PSNR comparison
    plt.subplot(2, 2, 2)
    psnr_values = [m['psnr'] for m in metrics_list]
    bars = plt.bar(labels, psnr_values, color='lightgreen')

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom')

    plt.title('PSNR Comparison (dB)')
    plt.ylabel('dB')
    plt.grid(axis='y', alpha=0.3)

    # Error metrics comparison
    plt.subplot(2, 2, 3)

    # Prepare data
    metrics_to_plot = ['mse', 'nrmse', 'entropy_diff']
    metrics_labels = ['MSE', 'NRMSE', 'Entropy Difference']

    # Normalize the error values for easier comparison
    normalized_errors = []
    for metric in metrics_to_plot:
        values = [m[metric] for m in metrics_list]
        max_val = max(values)
        if max_val > 0:
            normalized_errors.append([v/max_val for v in values])
        else:
            normalized_errors.append(values)

    x = np.arange(len(labels))
    width = 0.25

    for i, (metric, values) in enumerate(zip(metrics_labels, normalized_errors)):
        plt.bar(x + i*width - width, values, width, label=metric)

    plt.xlabel('Decryption Method')
    plt.ylabel('Normalized Error')
    plt.title('Error Metric Comparison (Normalized)')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # Regional quality comparison
    plt.subplot(2, 2, 4)
    region_values = [m['region_quality'] for m in metrics_list]
    bars = plt.bar(labels, region_values, color='skyblue')

    # Add value labels to the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.3f}', ha='center', va='bottom')

    plt.title('Mean Local Region Quality')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'{prefix}_comparison_{timestamp}.png'), dpi=300)
    plt.show()

    # Save comparison results to CSV
    import csv
    csv_path = os.path.join(output_dir, f'{prefix}_comparison_{timestamp}.csv')

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Method', 'SSIM', 'PSNR',
                      'Correlation', 'MSE', 'NRMSE', 'Edge Quality', 'Entropy Diff']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow({
                'Method': metrics['label'],
                'SSIM': f"{metrics['ssim']:.4f}",
                'PSNR': f"{metrics['psnr']:.2f}",
                'Correlation': f"{metrics['correlation']:.4f}",
                'MSE': f"{metrics['mse']:.6f}",
                'NRMSE': f"{metrics['nrmse']:.6f}",
                'Edge Quality': f"{metrics['edge_quality']:.4f}",
                'Entropy Diff': f"{metrics['entropy_diff']:.4f}"
            })

    print(f"Comparison results saved to: {csv_path}")
