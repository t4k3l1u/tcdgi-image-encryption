import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image


def tcdgi(reference_frames, bucket_signals, threshold=0, filter_sigma=1.0, use_weighted_avg=True):
    """
    Time-Correspondence Differential Ghost Imaging (TCDGI) algorithm
    with added denoising and weighted-averaging options.

    Parameters
    -----------
    reference_frames : ndarray
        Reference detector frame array, shape (num_frames, height, width)
    bucket_signals : ndarray
        Bucket detector signal array, shape (num_frames,)
    threshold : float, optional
        Intensity threshold used for signal selection
    filter_sigma : float, optional
        Standard deviation of the Gaussian filter, controls denoising strength
    use_weighted_avg : bool, optional
        Whether to use weighted averaging to enhance the reconstruction

    Returns
    --------
    image : ndarray
        Reconstructed image
    """
    # Compute the mean of the reference frames and bucket signals
    ref_mean = np.mean(reference_frames, axis=0)
    bucket_mean = np.mean(bucket_signals)

    # Compute the integrated reference signal
    integrated_ref = np.sum(reference_frames, axis=(1, 2))
    integrated_ref_mean = np.mean(integrated_ref)

    # Compute the differential bucket signal
    differential_bucket = bucket_signals - \
        bucket_mean * integrated_ref / integrated_ref_mean

    # Find frames where the differential signal is above or below the threshold
    positive_indices = differential_bucket > threshold
    negative_indices = differential_bucket < -threshold

    # If no frames satisfy the condition, skip processing
    if not np.any(positive_indices) or not np.any(negative_indices):
        return None

    # Denoise the reference frames (optional)
    if filter_sigma > 0:
        reference_frames = np.array([ndimage.gaussian_filter(
            frame, sigma=filter_sigma) for frame in reference_frames])

    # If weighted averaging is selected
    if use_weighted_avg:
        # Weights: based on the values of the differential bucket signal
        positive_weights = differential_bucket[positive_indices]
        negative_weights = -differential_bucket[negative_indices]

        positive_image = np.average(
            reference_frames[positive_indices], axis=0, weights=positive_weights)
        negative_image = np.average(
            reference_frames[negative_indices], axis=0, weights=negative_weights)
    else:
        positive_image = np.mean(reference_frames[positive_indices], axis=0)
        negative_image = np.mean(reference_frames[negative_indices], axis=0)

    # Combine the positive and negative images
    tcdgi_image = positive_image - negative_image

    # Add frame-count statistics
    total_frames = len(bucket_signals)
    used_frames = np.sum(positive_indices) + np.sum(negative_indices)
    print(f"Threshold {threshold:.4f} used {used_frames}/{total_frames} frames ({used_frames/total_frames*100:.1f}%)")

    return tcdgi_image


def normalize_image(image, norm_type='standard'):
    """
    Normalize an image, supporting multiple normalization modes.

    Parameters
    -----------
    image : ndarray
        Input image
    norm_type : str, optional
        Normalization type:
        - 'standard': standard normalization to the [0, 1] range (default)
        - 'centered': centered normalization to the [-1, 1] range
        - 'zero_mean': zero-mean normalization with unit standard deviation
        - 'preserve': preserve the original value range, only linear contrast stretching
        - 'adaptive': adaptive normalization, preserves image characteristics

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

    elif norm_type == 'zero_mean':
        # Zero-mean normalization with unit standard deviation
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            return (img - mean) / std
        return img - mean

    elif norm_type == 'preserve':
        # Preserve the original value range, only linear contrast stretching
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max > img_min:
            # Preserve the sign, apply linear stretching
            mean = (img_max + img_min) / 2
            amplitude = max(abs(img_max - mean), abs(img_min - mean))
            if amplitude > 0:
                return (img - mean) / amplitude
        return img

    elif norm_type == 'adaptive':
        # Adaptive normalization, choose the best mode based on image characteristics
        # Check whether the image has a clear positive/negative distribution
        mean = np.mean(img)
        min_val = np.min(img)
        max_val = np.max(img)

        # Compute the proportion of positive and negative values
        pos_ratio = np.sum(img > 0) / img.size
        neg_ratio = np.sum(img < 0) / img.size

        # If there is already a positive/negative distribution, use preserve-type normalization
        if neg_ratio > 0.1 and pos_ratio > 0.1:
            return normalize_image(img, 'preserve')
        # Otherwise use centered normalization
        else:
            return normalize_image(img, 'centered')

    # Default to standard normalization
    return normalize_image(img, 'standard')


def calculate_snr(original, reconstructed, norm_type='adaptive'):
    """
    Compute the signal-to-noise ratio, supporting multiple normalization modes.

    Parameters
    -----------
    original : ndarray
        Original object mask
    reconstructed : ndarray
        Reconstructed image
    norm_type : str, optional
        Normalization type

    Returns
    --------
    snr : float
        Signal-to-noise ratio
    """
    # Normalize the images
    original_norm = normalize_image(original, norm_type)
    reconstructed_norm = normalize_image(reconstructed, norm_type)

    # Compute the mean of the original image
    T0_mean = np.mean(original_norm)

    # Compute the signal power (numerator of Equation 10)
    signal = np.sum((original_norm - T0_mean)**2)

    # Compute the noise power (denominator of Equation 10)
    noise = np.sum((reconstructed_norm - original_norm)**2)

    # Avoid division by zero
    if noise < 1e-10:  # add a small value to avoid division by zero
        noise = 1e-10

    return np.sqrt(signal / noise)


def calculate_adaptive_thresholds(bucket_signals, integrated_ref):
    """
    Compute adaptive thresholds based on the standard deviation of the differential bucket signal.

    Parameters
    -----------
    bucket_signals : ndarray
        Bucket detector signal array
    integrated_ref : ndarray
        Integrated reference signal

    Returns
    --------
    thresholds : list
        Computed list of thresholds
    """
    # Compute the differential bucket signal
    bucket_mean = np.mean(bucket_signals)
    integrated_ref_mean = np.mean(integrated_ref)
    differential_bucket = bucket_signals - \
        bucket_mean * integrated_ref / integrated_ref_mean

    # Compute the standard deviation
    sigma = np.std(differential_bucket)

    # Return a list of thresholds based on the standard deviation
    # Typically 0.5σ, 1.0σ, 1.5σ are used as thresholds
    thresholds = [0.4 * sigma, 0.6 * sigma, 0.8 * sigma]

    print(f"Differential bucket signal standard deviation: {sigma:.4f}")
    print(f"Adaptive thresholds: {[f'{t:.4f}' for t in thresholds]}")

    # Add a histogram analysis
    plt.figure(figsize=(8, 4))
    plt.hist(differential_bucket, bins=50, density=True)
    plt.title('Differential Bucket Signal Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    for t in thresholds:
        plt.axvline(x=t, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=-t, color='r', linestyle='--', alpha=0.5)
    plt.show()

    return thresholds


def compare_imaging_methods(reference_frames, bucket_signals, object_mask, thresholds=[0],
                            filter_sigmas=[1.0], use_weighted_avgs=[True]):
    """
    Compare different ghost-imaging methods: GI, DGI, and TCDGI (with different thresholds and parameters).

    Parameters
    -----------
    reference_frames : ndarray
        Reference detector frame array
    bucket_signals : ndarray
        Bucket detector signal array
    object_mask : ndarray
        Original object mask
    thresholds : list, optional
        List of TCDGI thresholds
    filter_sigmas : list, optional
        List of Gaussian filter standard deviations
    use_weighted_avgs : list, optional
        List of booleans for whether to use weighted averaging

    Returns
    --------
    results : dict
        Dictionary of reconstructed images and their SNR values
    """
    num_frames = len(reference_frames)
    results = {}

    # Conventional ghost imaging (GI)
    bucket_fluctuations = bucket_signals - np.mean(bucket_signals)
    gi_image = np.zeros_like(reference_frames[0])
    for i in range(num_frames):
        gi_image += bucket_fluctuations[i] * reference_frames[i]
    gi_image /= num_frames
    results['GI'] = {'image': gi_image,
                     'snr': calculate_snr(object_mask, gi_image),
                     'method': 'GI'}

    # Differential ghost imaging (DGI)
    ref_integrated = np.sum(reference_frames, axis=(1, 2))
    ref_integrated_mean = np.mean(ref_integrated)
    bucket_mean = np.mean(bucket_signals)
    differential_bucket = bucket_signals - \
        bucket_mean * ref_integrated / ref_integrated_mean

    dgi_image = np.zeros_like(reference_frames[0])
    for i in range(num_frames):
        dgi_image += differential_bucket[i] * reference_frames[i]
    dgi_image /= num_frames
    results['DGI'] = {'image': dgi_image,
                      'snr': calculate_snr(object_mask, dgi_image),
                      'method': 'DGI'}

    # Time-Correspondence Differential Ghost Imaging (TCDGI) with different parameter combinations
    for k in thresholds:
        for sigma in filter_sigmas:
            for weighted in use_weighted_avgs:
                # Build the parameter description
                param_desc = f'k{k}_sigma{sigma}'
                if weighted:
                    param_desc += '_weighted'
                else:
                    param_desc += '_unweighted'

                method_name = f'TCDGI_{param_desc}'

                # Call the TCDGI algorithm
                tcdgi_image = tcdgi(
                    reference_frames,
                    bucket_signals,
                    threshold=k,
                    filter_sigma=sigma,
                    use_weighted_avg=weighted
                )

                if tcdgi_image is not None:
                    results[method_name] = {
                        'image': tcdgi_image,
                        'snr': calculate_snr(object_mask, tcdgi_image),
                        'method': 'TCDGI',
                        'threshold': k,
                        'filter_sigma': sigma,
                        'use_weighted_avg': weighted,
                        'param_desc': param_desc
                    }

    return results


def simulate_ghost_imaging(object_mask, num_frames=1000, speckle_size=5, seed=None):
    """
    Simulate a pseudo-thermal-light ghost-imaging experiment.

    Parameters
    -----------
    object_mask : ndarray
        Object transmission function
    num_frames : int, optional
        Number of frames to simulate
    speckle_size : int, optional
        Average speckle size (in pixels)
    seed : int, optional
        Random seed used to reproduce the speckle sequence; if None, the global random state is used

    Returns
    --------
    reference_frames : ndarray
        Simulated reference detector frames
    bucket_signals : ndarray
        Simulated bucket detector signals
    """
    height, width = object_mask.shape
    reference_frames = np.zeros((num_frames, height, width))
    bucket_signals = np.zeros(num_frames)
    rng = np.random.RandomState(seed) if seed is not None else np.random

    for i in range(num_frames):
        if i % 100 == 0:  # show progress every 100 frames
            print(f"Processing frame {i}/{num_frames}...")
        # Generate a random speckle pattern
        random_phase = rng.random((height, width)) * 2 * np.pi
        speckle = np.abs(np.fft.fft2(np.exp(1j * random_phase)))**2
        speckle = ndimage.gaussian_filter(speckle, sigma=speckle_size)
        speckle = speckle / np.mean(speckle)  # normalize

        # Store the reference frame
        reference_frames[i] = speckle

        # Compute the bucket signal (total intensity after passing through the object)
        bucket_signals[i] = np.sum(speckle * object_mask)

    return reference_frames, bucket_signals


def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """
    Load and preprocess an image to use as the object mask.

    Parameters
    -----------
    image_path : str
        Path to the image file
    target_size : tuple, optional
        Target size

    Returns
    --------
    object_mask : ndarray
        Preprocessed object mask
    """
    try:
        print(f"Processing image: {image_path}")  # debug info
        img = Image.open(image_path)
        print(f"Original image size: {img.size}")
        img = img.resize(target_size)
        img = img.convert('L')
        object_mask = np.array(img).astype(float) / 255.0
        print(f"Processed image shape: {object_mask.shape}")
        return object_mask
    except Exception as e:
        print(f"Image processing error: {e}")
        return None


def calculate_ssim(original, reconstructed, win_size=11, k1=0.01, k2=0.03, L=2.0, norm_type='adaptive'):
    """
    Compute the Structural Similarity Index (SSIM), supporting multiple normalization types.

    Parameters
    -----------
    original : ndarray
        Original reference image
    reconstructed : ndarray
        Reconstructed image
    win_size : int, optional
        Sliding window size, should be odd
    k1, k2 : float, optional
        Constants in the SSIM formula
    L : float, optional
        Dynamic range of pixel values (after normalization, typically 2 for the [-1,1] range)
    norm_type : str, optional
        Normalization type

    Returns
    --------
    ssim : float
        Structural Similarity Index, range [0,1], higher means better image quality
    """
    # Ensure the window size is odd
    if win_size % 2 == 0:
        win_size += 1

    # Normalize the images - use the adaptive or specified normalization type
    original_norm = normalize_image(original, norm_type)
    reconstructed_norm = normalize_image(reconstructed, norm_type)

    # Define the Gaussian window
    def gaussian_window(win_size, sigma=1.5):
        x, y = np.mgrid[-win_size//2 + 1:win_size //
                        2 + 1, -win_size//2 + 1:win_size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    window = gaussian_window(win_size)

    # Dynamically adjust L based on the normalization type
    if norm_type == 'centered' or norm_type == 'preserve':
        L = 2.0  # [-1,1] range
    elif norm_type == 'zero_mean':
        # For zero-mean normalization, adjust L to an estimate of the original image's dynamic range
        L = 4.0 * np.std(original_norm)
    elif norm_type == 'standard':
        L = 1.0  # [0,1] range

    # Constants to avoid division by zero
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Compute the means
    mu1 = ndimage.convolve(original_norm, window)
    mu2 = ndimage.convolve(reconstructed_norm, window)

    # Compute variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = ndimage.convolve(original_norm**2, window) - mu1_sq
    sigma2_sq = ndimage.convolve(reconstructed_norm**2, window) - mu2_sq
    sigma12 = ndimage.convolve(
        original_norm * reconstructed_norm, window) - mu1_mu2

    # Compute SSIM
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den

    # Return the average SSIM
    return np.mean(ssim_map)


def calculate_quality_metrics(original, reconstructed, norm_type='adaptive'):
    """
    Compute several image quality metrics, supporting multiple normalization modes.

    Parameters
    -----------
    original : ndarray
        Original reference image
    reconstructed : ndarray
        Reconstructed image
    norm_type : str, optional
        Normalization type:
        - 'standard': standard normalization to [0, 1]
        - 'centered': centered normalization to [-1, 1]
        - 'zero_mean': zero-mean normalization
        - 'preserve': preserve the original value range
        - 'adaptive': adaptively choose the best normalization mode (default)

    Returns
    --------
    metrics : dict
        Dictionary containing several quality metrics
    """
    # Normalize the images
    original_norm = normalize_image(original, norm_type)
    reconstructed_norm = normalize_image(reconstructed, norm_type)

    # Compute SSIM
    ssim_value = calculate_ssim(original, reconstructed, norm_type=norm_type)

    # Compute SNR
    snr_value = calculate_snr(original, reconstructed, norm_type=norm_type)

    # Compute the root-mean-square error (RMSE) - using the normalized images
    mse = np.mean((original_norm - reconstructed_norm) ** 2)
    rmse = np.sqrt(mse)

    # Compute the peak signal-to-noise ratio (PSNR)
    # Dynamically adjust MAX_I based on the normalization type
    if norm_type == 'centered' or norm_type == 'preserve':
        MAX_I = 2.0  # dynamic range for the [-1,1] range
    elif norm_type == 'zero_mean':
        # For zero-mean normalization, use the actual range
        MAX_I = np.max(original_norm) - np.min(original_norm)
        if MAX_I < 1e-10:
            MAX_I = 1.0
    else:
        MAX_I = 1.0  # dynamic range for the [0,1] range

    if mse < 1e-10:
        psnr = 100
    else:
        psnr = 20 * np.log10(MAX_I / np.sqrt(mse))

    # Compute the correlation coefficient (CORR)
    orig_flat = original_norm.flatten() - np.mean(original_norm)
    recon_flat = reconstructed_norm.flatten() - np.mean(reconstructed_norm)
    correlation = np.sum(orig_flat * recon_flat) / \
        (np.sqrt(np.sum(orig_flat**2) * np.sum(recon_flat**2)) + 1e-10)

    # Return all metrics together with the normalization type used
    return {
        'ssim': ssim_value,
        'snr': snr_value,
        'rmse': rmse,
        'psnr': psnr,
        'correlation': correlation,
        'norm_type': norm_type
    }


def visualize_results(results, nrows=None, ncols=None, title="Imaging Method Comparison", norm_type='adaptive'):
    """
    Visualize the results of different imaging methods and compare SNR and SSIM
    quality metrics, supporting different normalization methods.

    Parameters
    -----------
    results : dict
        Dictionary of reconstruction results
    nrows, ncols : int, optional
        Number of rows/columns for the visualization
    title : str, optional
        Chart title
    norm_type : str, optional
        Normalization type
    """
    from collections import defaultdict

    # Ensure every result has quality metrics
    for method_name, result in results.items():
        if ('ssim' not in result or 'snr' not in result) and method_name != 'original':
            # Compute the metrics if they are not specified
            if 'image' in result and method_name != 'original':
                original_image = results['original']['image']
                metrics = calculate_quality_metrics(
                    original_image, result['image'], norm_type=norm_type)
                # Update the results dictionary
                for metric_name, value in metrics.items():
                    result[metric_name] = value

    # Use SNR as the primary sort metric
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('snr', float('-inf')),
        reverse=True
    )

    # Extract method names and quality metric values
    method_names = []
    ssim_values = []
    snr_values = []
    method_types = defaultdict(list)  # use defaultdict to avoid key errors

    # Extract parameter information from the results
    thresholds = set()
    filter_sigmas = set()

    for method_name, result in sorted_results:
        if method_name == 'original':
            continue  # skip the original image

        method_names.append(method_name)
        ssim_values.append(result.get('ssim', 0))
        snr_values.append(result.get('snr', 0))

        # Group by method type
        if 'method' in result:
            method_type = result['method']
            method_types[method_type].append(
                (method_name, result.get('ssim', 0), result.get('snr', 0)))

        # Extract parameter information
        if 'threshold' in result:
            thresholds.add(result['threshold'])
        if 'filter_sigma' in result:
            filter_sigmas.add(result['filter_sigma'])

    # Convert to sorted lists
    thresholds = sorted(list(thresholds))
    filter_sigmas = sorted(list(filter_sigmas))

    # Determine the number of rows and columns
    num_methods = len(method_names)
    if num_methods == 0:
        print("No method results to visualize")
        return None, 0, 0

    if ncols is None:
        ncols = min(3, num_methods)
    if nrows is None:
        nrows = (num_methods + ncols - 1) // ncols

    # Create a large figure containing all the visualizations
    plt.figure(figsize=(15, 10))

    # 1. Bar chart comparing the SNR metric across all methods (placed first)
    plt.subplot(2, 2, 1)
    y_pos = np.arange(len(method_names))

    # Use different colors to identify different method types
    colors = []
    for name in method_names:
        if 'GI' == name:
            colors.append('blue')
        elif 'DGI' == name:
            colors.append('green')
        else:
            colors.append('red')

    bars = plt.bar(y_pos, snr_values, align='center',
                   alpha=0.7, color=colors)
    plt.xticks(y_pos, method_names, rotation=45, ha='right')
    plt.ylabel('SNR Value')
    plt.title(f'SNR Comparison Across Methods (Normalization: {norm_type})')

    # Set a more precise y-axis range
    if snr_values:
        min_snr = min(snr_values)
        max_snr = max(snr_values)
        # Determine an appropriate range
        range_snr = max_snr - min_snr
        # Narrow the range to make differences more visible
        plt.ylim(max(0, min_snr - range_snr * 0.05),
                 max_snr + range_snr * 0.05)

        # Set finer y-axis ticks
        plt.yticks(np.linspace(min_snr, max_snr, 10).round(4))

    # Add precise value labels to each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max_snr - min_snr) * 0.01,
                 f'{snr_values[i]:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)

    # 2. Bar chart comparing the SSIM metric across all methods (placed second)
    plt.subplot(2, 2, 2)
    bars = plt.bar(y_pos, ssim_values, align='center',
                   alpha=0.7, color=colors)
    plt.xticks(y_pos, method_names, rotation=45, ha='right')
    plt.ylabel('SSIM Value')
    plt.title(f'SSIM Comparison Across Methods (Normalization: {norm_type})')

    # Set a more precise y-axis range
    if ssim_values:
        min_ssim = min(ssim_values)
        max_ssim = max(ssim_values)
        # Determine an appropriate range to make differences more visible
        range_ssim = max_ssim - min_ssim
        # If the value range is too small, use a narrower range to highlight the differences
        if range_ssim < 0.1:
            plt.ylim(min_ssim - range_ssim * 0.1, max_ssim + range_ssim * 0.1)
        else:
            # Narrow the range to make differences more visible
            plt.ylim(max(0, min_ssim - range_ssim * 0.05),
                     min(1, max_ssim + range_ssim * 0.05))

    # Add precise value labels to each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (max_ssim - min_ssim) * 0.01,
                 f'{ssim_values[i]:.4f}', ha='center', va='bottom', rotation=0, fontsize=9)

    # 3. Heatmap or line plot of parameter effects (TCDGI methods only)
    tcdgi_results = [(k, v) for k, v in sorted_results if 'TCDGI' in k]

    if len(thresholds) > 1:
        # Threshold effect plot
        plt.subplot(2, 2, 3)
        x_values = thresholds
        y_values_ssim = []
        y_values_snr = []

        for t in thresholds:
            # Find the best result using this threshold
            best_ssim = float('-inf')
            best_snr = float('-inf')
            for method_name, result in sorted_results:
                if 'method' in result and result['method'] == 'TCDGI' and 'threshold' in result and result['threshold'] == t:
                    ssim_val = result.get('ssim', 0)
                    snr_val = result.get('snr', 0)
                    best_ssim = max(best_ssim, ssim_val)
                    best_snr = max(best_snr, snr_val)
            y_values_ssim.append(best_ssim)
            y_values_snr.append(best_snr)

        plt.plot(x_values, y_values_snr, 'o-',
                 linewidth=2, label='SNR', color='blue')
        plt.plot(x_values, y_values_ssim, 's--',
                 linewidth=2, label='SSIM', color='red')
        plt.xlabel('Threshold')
        plt.ylabel('Quality Metric Value')
        plt.title('Effect of Threshold on Quality Metrics')
        plt.legend()
        plt.grid(True)

    # 4. Show a comparison between the original image and the best result
    # Get the best valid method (excluding the original image)
    best_method = None
    best_ssim = 0
    best_snr = 0
    best_result = None

    for method_name, result in sorted_results:
        if method_name != 'original' and 'image' in result:
            best_method = method_name
            best_ssim = result.get('ssim', 0)
            best_snr = result.get('snr', 0)
            best_result = result
            break

    # Show the original image (if available)
    has_original = False
    for method_name, result in sorted_results:
        if method_name == 'original' and 'image' in result:
            plt.subplot(2, 2, 3)
            plt.imshow(normalize_image(
                result['image'], norm_type), cmap='gray')
            plt.title(f'Original Image (Normalization: {norm_type})')
            plt.axis('off')
            has_original = True
            break

    # Show the best result
    if best_result and 'image' in best_result:
        plt.subplot(2, 2, 4)
        plt.imshow(normalize_image(
            best_result['image'], norm_type), cmap='gray')

        best_desc = best_method
        if 'method' in best_result and best_result['method'] == 'TCDGI':
            best_desc = f"TCDGI\nthreshold={best_result.get('threshold', 'N/A'):.4f}\nsigma={best_result.get('filter_sigma', 'N/A')}"
            if 'use_weighted_avg' in best_result:
                best_desc += "\nweighted average" if best_result['use_weighted_avg'] else "\nplain average"
            if 'iteration' in best_result:
                best_desc += f"\niteration {best_result['iteration']}"
            if 'edge_factor' in best_result and best_result['edge_factor'] > 0:
                best_desc += f"\nedge enhancement={best_result['edge_factor']:.2f}"
                if 'edge_method' in best_result:
                    best_desc += f"\n({best_result['edge_method']})"

        plt.title(
            f"Best method: {best_desc}\nSNR: {best_snr:.4f}, SSIM: {best_ssim:.4f}\nNormalization: {norm_type}")
        plt.axis('off')

    # Add a table
    plt.tight_layout()
    plt.suptitle(f"{title} (SNR & SSIM, Normalization: {norm_type})", fontsize=16)
    plt.subplots_adjust(top=0.9, bottom=0.15)  # leave space for the table

    # Show a detailed comparison of multiple top results
    plt.figure(figsize=(15, 6))

    # Show the top few results
    top_n = min(
        5, len([r for m, r in sorted_results if m != 'original' and 'image' in r]))
    shown_count = 0

    for i, (method_name, result) in enumerate(sorted_results):
        if method_name == 'original' or 'image' not in result:
            continue  # skip the original image and invalid results

        if shown_count >= top_n:
            break

        plt.subplot(1, top_n, shown_count+1)
        plt.imshow(normalize_image(result['image'], norm_type), cmap='gray')

        method_desc = method_name
        if 'method' in result and result['method'] == 'TCDGI':
            method_desc = f"TCDGI\nthreshold={result.get('threshold', 'N/A'):.2f}"
            if 'filter_sigma' in result:
                method_desc += f"\nsigma={result['filter_sigma']}"
            if 'use_weighted_avg' in result:
                method_desc += "\nweighted" if result['use_weighted_avg'] else "\nunweighted"
            if 'iteration' in result:
                method_desc += f"\niteration {result['iteration']}"
            if 'edge_factor' in result and result['edge_factor'] > 0:
                method_desc += f"\nedge={result['edge_factor']:.1f}"

        plt.title(
            f"{method_desc}\nSNR: {result.get('snr', 0):.3f}\nSSIM: {result.get('ssim', 0):.3f}")
        plt.axis('off')
        shown_count += 1

    plt.tight_layout()
    plt.suptitle(
        f"Image Comparison of the Top {top_n} Methods (SNR & SSIM, Normalization: {norm_type})", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()

    # Print the quality comparison results
    print(f"\nSNR & SSIM comparison results (sorted by SNR, descending, Normalization: {norm_type}):")
    for method_name, result in sorted_results:
        if method_name == 'original':
            continue  # skip the original image

        if 'ssim' not in result or 'snr' not in result:
            continue  # skip results without quality metrics

        method_info = f"{method_name}: SNR={result['snr']:.4f}, SSIM={result['ssim']:.4f}"

        # Add extra parameter info (TCDGI only)
        if 'method' in result and result['method'] == 'TCDGI':
            method_info += f" [threshold={result.get('threshold', 'N/A'):.3f}, sigma={result.get('filter_sigma', 'N/A')}"
            if 'use_weighted_avg' in result:
                method_info += ", weighted" if result['use_weighted_avg'] else ", unweighted"
            if 'iteration' in result:
                method_info += f", iteration {result['iteration']}"
            if 'edge_factor' in result and result['edge_factor'] > 0:
                method_info += f", edge enhancement={result['edge_factor']:.2f}"
                if 'edge_method' in result:
                    method_info += f"({result['edge_method']})"
            method_info += "]"

        print(method_info)

    # Return the best method
    return best_method, best_ssim, best_snr


def select_image_file():
    """Let the user select an image file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # hide the main window
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        return file_path if file_path else None
    except ImportError:
        # If tkinter is unavailable, fall back to command-line input
        print("Please enter the full path to the image file:")
        return input().strip()


def enhance_edges(image, enhancement_factor=1.0, method='sobel'):
    """
    Apply edge enhancement to an image.

    Parameters
    -----------
    image : ndarray
        Input image
    enhancement_factor : float, optional
        Edge enhancement strength factor
    method : str, optional
        Edge detection method ('sobel', 'laplacian', 'prewitt')

    Returns
    --------
    enhanced_image : ndarray
        Edge-enhanced image
    """
    from scipy import ndimage

    # Copy the original image
    enhanced_image = image.copy()

    # Apply edge detection
    if method.lower() == 'sobel':
        # Sobel operator
        edges_x = ndimage.sobel(image, axis=0)
        edges_y = ndimage.sobel(image, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)
    elif method.lower() == 'laplacian':
        # Laplacian operator
        edges = np.abs(ndimage.laplace(image))
    elif method.lower() == 'prewitt':
        # Prewitt operator
        edges_x = ndimage.prewitt(image, axis=0)
        edges_y = ndimage.prewitt(image, axis=1)
        edges = np.sqrt(edges_x**2 + edges_y**2)
    else:
        raise ValueError(f"Unsupported edge detection method: {method}")

    # Normalize the edge map
    edges = edges / np.max(edges) if np.max(edges) > 0 else edges

    # Enhance the original image
    enhanced_image = image + enhancement_factor * edges

    # Normalize the result
    enhanced_image = normalize_image(enhanced_image)

    return enhanced_image


def iterative_tcdgi(reference_frames, bucket_signals, object_mask=None, iterations=3,
                    threshold=0, filter_sigma=1.0, use_edge_enhancement=False,
                    edge_factor=1.0, edge_method='sobel'):
    """
    Iterative TCDGI algorithm, refining the reconstruction at each iteration.

    Parameters
    -----------
    reference_frames : ndarray
        Reference detector frame array
    bucket_signals : ndarray
        Bucket detector signal array
    object_mask : ndarray, optional
        Original object mask (used for SNR computation)
    iterations : int, optional
        Number of iterations
    threshold : float, optional
        Intensity threshold used for signal selection
    filter_sigma : float, optional
        Standard deviation of the Gaussian filter
    use_edge_enhancement : bool, optional
        Whether to use edge enhancement
    edge_factor : float, optional
        Edge enhancement strength
    edge_method : str, optional
        Edge detection method

    Returns
    --------
    results : dict
        Dictionary containing images and SNR values from the iterative process
    """
    results = {}
    current_image = None

    print(f"Starting iterative TCDGI, number of iterations: {iterations}")

    for i in range(iterations):
        print(f"Running iteration {i+1}/{iterations}...")

        # The first iteration uses basic TCDGI
        if i == 0:
            current_image = tcdgi(reference_frames, bucket_signals,
                                  threshold=threshold,
                                  filter_sigma=filter_sigma,
                                  use_weighted_avg=True)
        else:
            # Subsequent iterations refine using the previous result
            # Weight the reference frames based on the previous result
            weighted_refs = reference_frames.copy()
            for j in range(len(reference_frames)):
                # Compute the correlation as a weighting factor
                correlation = np.sum(
                    reference_frames[j] * current_image) / np.sum(reference_frames[j]**2)
                weighted_refs[j] = reference_frames[j] * \
                    (1 + correlation * i / iterations)

            # Gradually adjust the parameters
            current_threshold = threshold * (1 + 0.1*i)  # gradually increase the threshold
            current_sigma = filter_sigma * (1 - 0.05*i)  # gradually decrease the blur

            # Run TCDGI with the adjusted parameters and weighted reference frames
            current_image = tcdgi(weighted_refs, bucket_signals,
                                  threshold=current_threshold,
                                  filter_sigma=current_sigma,
                                  use_weighted_avg=True)

        # Apply edge enhancement (if enabled)
        if use_edge_enhancement and current_image is not None:
            # The edge enhancement factor increases with iterations
            current_edge_factor = edge_factor * (i+1)/iterations
            enhanced_image = enhance_edges(current_image,
                                           enhancement_factor=current_edge_factor,
                                           method=edge_method)
            current_image = enhanced_image

        # Store the result of the current iteration
        if current_image is not None:
            if object_mask is not None:
                metrics = calculate_quality_metrics(object_mask, current_image)
                results[f'Iteration {i+1}'] = {
                    'image': current_image,
                    **metrics,
                    'method': 'TCDGI_Iterative',
                    'iteration': i+1,
                    'threshold': threshold * (1 + 0.1*i) if i > 0 else threshold,
                    'filter_sigma': filter_sigma * (1 - 0.05*i) if i > 0 else filter_sigma,
                    'use_edge_enhancement': use_edge_enhancement,
                    'edge_factor': edge_factor * (i+1)/iterations if use_edge_enhancement else 0,
                    'edge_method': edge_method if use_edge_enhancement else None
                }
                print(
                    f"Iteration {i+1} complete, SSIM: {metrics['ssim']:.4f}, SNR: {metrics['snr']:.4f}")
            else:
                results[f'Iteration {i+1}'] = {
                    'image': current_image,
                    'method': 'TCDGI_Iterative',
                    'iteration': i+1,
                    'threshold': threshold * (1 + 0.1*i) if i > 0 else threshold,
                    'filter_sigma': filter_sigma * (1 - 0.05*i) if i > 0 else filter_sigma,
                    'use_edge_enhancement': use_edge_enhancement,
                    'edge_factor': edge_factor * (i+1)/iterations if use_edge_enhancement else 0,
                    'edge_method': edge_method if use_edge_enhancement else None
                }
                print(f"Iteration {i+1} complete")
        else:
            print(f"Iteration {i+1} failed to produce a valid image")
            break
    return results


def batch_threshold_test(reference_frames, bucket_signals, object_mask=None,
                         filter_sigma=1.0, num_thresholds=15,
                         custom_range=False, min_threshold=None, max_threshold=None,
                         norm_type='adaptive'):
    """
    Batch-test multiple thresholds and find the best one, supporting a custom
    threshold range and normalization method.

    Parameters
    -----------
    reference_frames : ndarray
        Reference detector frame array
    bucket_signals : ndarray
        Bucket detector signal array
    object_mask : ndarray, optional
        Original object mask (used to compute SNR)
    filter_sigma : float, optional
        Standard deviation of the Gaussian filter
    num_thresholds : int, optional
        Number of thresholds to test
    custom_range : bool, optional
        Whether to use a custom threshold range
    min_threshold, max_threshold : float, optional
        Minimum and maximum values of the custom threshold range
    norm_type : str, optional
        Normalization type

    Returns
    --------
    results : dict
        Dictionary containing the test results for each threshold
    best_threshold : float
        Best threshold
    best_snr : float
        Best SNR value
    """
    # Compute the differential bucket signal
    bucket_mean = np.mean(bucket_signals)
    integrated_ref = np.sum(reference_frames, axis=(1, 2))
    integrated_ref_mean = np.mean(integrated_ref)
    differential_bucket = bucket_signals - \
        bucket_mean * integrated_ref / integrated_ref_mean

    # Determine the threshold range
    if not custom_range:
        # Use the original approach: compute the threshold range based on the signal distribution
        signal_std = np.std(differential_bucket)
        min_threshold = 0
        max_threshold = 2.0 * signal_std
        print(f"Threshold range based on signal distribution: {min_threshold:.4f} - {max_threshold:.4f}")
    else:
        # Use the user-specified custom range
        print(f"Using custom threshold range: {min_threshold:.4f} - {max_threshold:.4f}")

    # Generate evenly spaced thresholds
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)
    # Evaluate each threshold
    results = {}
    best_snr = 0
    best_threshold = thresholds[0]
    snr_values = []
    images = []

    print(f"Starting batch test of {num_thresholds} thresholds...")
    print(f"Threshold range: {min_threshold:.4f} - {max_threshold:.4f}")
    print(f"Using normalization method: {norm_type}")

    for i, threshold in enumerate(thresholds):
        print(f"Testing threshold {i+1}/{num_thresholds}: {threshold:.4f}")

        # Run the TCDGI algorithm (always using weighted averaging)
        image = tcdgi(reference_frames, bucket_signals,
                      threshold=threshold,
                      filter_sigma=filter_sigma,
                      use_weighted_avg=True)

        if image is not None:
            # Compute or estimate the image quality
            if object_mask is not None:
                metrics = calculate_quality_metrics(
                    object_mask, image, norm_type=norm_type)
                snr = metrics['snr']
            else:
                # Without a reference image, estimate quality using contrast
                contrast = np.std(image) / np.mean(np.abs(image) + 1e-10)

                # Compute the frame utilization rate
                positive_indices = differential_bucket > threshold
                negative_indices = differential_bucket < -threshold
                usage = (np.sum(positive_indices) +
                         np.sum(negative_indices)) / len(bucket_signals)

                # The ideal frame utilization range is roughly 10%-40%
                usage_score = 1.0 if 0.1 <= usage <= 0.4 else max(
                    0.1, 1.0 - abs(usage - 0.25) / 0.15)

                snr = contrast * usage_score
                metrics = {'snr': snr, 'norm_type': norm_type}

            # Save the result
            results[f'threshold_{threshold:.4f}'] = {
                'image': image,
                **metrics,
                'method': 'TCDGI',
                'threshold': threshold,
                'filter_sigma': filter_sigma,
                'use_weighted_avg': True
            }

            snr_values.append(snr)
            images.append(image)

            # Update the best threshold
            if snr > best_snr:
                best_snr = snr
                best_threshold = threshold

            print(f"  SNR: {snr:.4f}")
            if 'ssim' in metrics:
                print(f"  SSIM: {metrics['ssim']:.4f}")
        else:
            snr_values.append(0)
            images.append(None)
            print(f"  Unable to generate a valid image")

    # Visualize the threshold test results
    plt.figure(figsize=(15, 10))

    # Plot the threshold-SNR curve
    plt.subplot(2, 2, 1)
    plt.plot(thresholds, snr_values, 'o-', color='blue', linewidth=2)
    plt.axvline(x=best_threshold, color='r', linestyle='--')
    plt.title(f'Threshold-SNR Relationship (Best: {best_threshold:.4f})')
    plt.xlabel('Threshold')
    plt.ylabel('SNR Value')
    plt.grid(True)

    # Show the image at the best threshold
    best_idx = np.argmax(snr_values)
    if images[best_idx] is not None:
        plt.subplot(2, 2, 2)
        plt.imshow(normalize_image(images[best_idx]), cmap='gray')
        plt.title(f'Reconstructed Image at Best Threshold ({best_threshold:.4f})\nNormalization method: {norm_type}')
        plt.axis('off')

    # Show images for a few representative thresholds
    plt.subplot(2, 2, 3)
    idx_min = 0  # minimum threshold
    idx_max = len(thresholds) - 1  # maximum threshold
    idx_mid = len(thresholds) // 2  # middle threshold

    if images[idx_min] is not None:
        plt.subplot(2, 3, 4)
        plt.imshow(normalize_image(images[idx_min]), cmap='gray')
        plt.title(
            f'Threshold: {thresholds[idx_min]:.4f}\nSNR: {snr_values[idx_min]:.2f}')
        plt.axis('off')

    if images[idx_mid] is not None:
        plt.subplot(2, 3, 5)
        plt.imshow(normalize_image(images[idx_mid]), cmap='gray')
        plt.title(
            f'Threshold: {thresholds[idx_mid]:.4f}\nSNR: {snr_values[idx_mid]:.2f}')
        plt.axis('off')

    if images[idx_max] is not None:
        plt.subplot(2, 3, 6)
        plt.imshow(normalize_image(images[idx_max]), cmap='gray')
        plt.title(
            f'Threshold: {thresholds[idx_max]:.4f}\nSNR: {snr_values[idx_max]:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Threshold Test Result Analysis (Normalization method: {norm_type})', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

    print(f"\nBest threshold: {best_threshold:.4f}, SNR: {best_snr:.4f}")

    return results, best_threshold, best_snr


def um_to_pixels(um_size, pixel_size_um=1.0):
    """
    Convert a size in micrometers to pixels.

    Parameters
    -----------
    um_size : float
        Size in micrometers
    pixel_size_um : float, optional
        Number of micrometers per pixel

    Returns
    --------
    pixel_size : float
        Size in pixels
    """
    return um_size / pixel_size_um


def compare_normalization_methods(original, reconstructed, title="Normalization Method Comparison"):
    """
    Compare the effect of different normalization methods on the computed SNR and SSIM.

    Parameters
    -----------
    original : ndarray
        Original reference image
    reconstructed : ndarray
        Reconstructed image
    title : str, optional
        Chart title

    Returns
    --------
    best_method : str
        Best normalization method
    metrics : dict
        Metric results for each normalization method
    """
    norm_methods = ['standard', 'centered',
                    'zero_mean', 'preserve', 'adaptive']
    results = {}

    for method in norm_methods:
        # Compute quality metrics for each normalization method
        metrics = calculate_quality_metrics(
            original, reconstructed, norm_type=method)
        results[method] = metrics

    # Print the comparison results
    print("\nQuality metric comparison across normalization methods:")
    print(f"{'Method':<10} {'SNR':<10} {'SSIM':<10} {'PSNR':<10} {'RMSE':<10}")
    print("-" * 50)

    for method, metrics in results.items():
        print(
            f"{method:<10} {metrics['snr']:<10.4f} {metrics['ssim']:<10.4f} {metrics['psnr']:<10.4f} {metrics['rmse']:<10.4f}")

    # Visualize the comparison
    plt.figure(figsize=(12, 10))

    # SNR comparison
    plt.subplot(2, 2, 1)
    methods = list(results.keys())
    snr_values = [results[m]['snr'] for m in methods]
    bars = plt.bar(methods, snr_values, color='blue', alpha=0.7)
    plt.title('SNR Comparison Across Normalization Methods')
    plt.ylabel('SNR Value')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # SSIM comparison
    plt.subplot(2, 2, 2)
    ssim_values = [results[m]['ssim'] for m in methods]
    bars = plt.bar(methods, ssim_values, color='green', alpha=0.7)
    plt.title('SSIM Comparison Across Normalization Methods')
    plt.ylabel('SSIM Value')
    plt.xticks(rotation=45)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # Show the original image
    plt.subplot(2, 2, 3)
    plt.imshow(normalize_image(original, 'standard'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Show the reconstructed image
    plt.subplot(2, 2, 4)
    plt.imshow(normalize_image(reconstructed, 'standard'), cmap='gray')
    plt.title('Reconstructed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Determine the best method (based on a weighted average of SNR and SSIM)
    best_method = max(results.keys(),
                      key=lambda m: results[m]['snr'] * 0.5 + results[m]['ssim'] * 0.5)

    print(f"\nRecommended best normalization method: {best_method}")
    print(
        f"SNR: {results[best_method]['snr']:.4f}, SSIM: {results[best_method]['ssim']:.4f}")

    return best_method, results


if __name__ == "__main__":
    print("===== Enhanced TCDGI Algorithm Test Tool =====")

    # Step 1: Select an image
    print("\nStep 1: Select a test image")
    image_path = select_image_file()

    if not image_path:
        print("No image file selected, using the default path: examples/sample_images/sample.jpg")
        image_path = 'examples/sample_images/sample.jpg'
    else:
        print(f"Selected image: {image_path}")

    # Load the image mask
    object_mask = load_and_preprocess_image(image_path)
    if object_mask is None:
        print("Image loading failed, exiting program.")
        exit()

    # Step 2: Set the speckle size and number of frames
    print("\nStep 2: Set the speckle size and number of frames")

    # Number of frames
    frames_input = input("Enter the number of simulated frames (default: 1000): ").strip()
    num_frames = int(
        frames_input) if frames_input and frames_input.isdigit() else 1000

    # Speckle size
    speckle_input = input("Enter the speckle size (default: 2.5): ").strip()
    speckle_size = float(speckle_input) if speckle_input and speckle_input.replace(
        '.', '', 1).isdigit() else 2.5

    # Step 3: Select enhancement features
    print("\nStep 3: Select enhancement features")
    print("1. Batch threshold test (recommended)")
    print("2. Iterative reconstruction optimization")
    print("3. Edge enhancement")
    print("4. Apply all (batch test + iterative optimization + edge enhancement)")
    print("5. Normalization method comparison analysis (new feature)")

    enhancement_choice = input("Select a feature (1-5): ").strip()

    # Ask the user to choose a normalization method
    print("\nSelect a normalization method:")
    print("1. Standard normalization [0,1] (standard)")
    print("2. Centered normalization [-1,1] (centered) - recommended")
    print("3. Zero-mean normalization (zero_mean)")
    print("4. Preserve original value range (preserve)")
    print("5. Adaptively choose the best normalization (adaptive)")

    norm_choice = input("Select a normalization method (1-5, default: 2): ").strip()

    # Set the normalization type
    norm_types = ['standard', 'centered', 'zero_mean', 'preserve', 'adaptive']
    norm_idx = int(norm_choice) - \
        1 if norm_choice and norm_choice.isdigit() and 1 <= int(norm_choice) <= 5 else 1
    norm_type = norm_types[norm_idx]

    print(f"Selected normalization method: {norm_type}")

    # Simulate the ghost-imaging experiment
    print(f"\nSimulating ghost-imaging experiment, num_frames={num_frames}, speckle_size={speckle_size}...")
    reference_frames, bucket_signals = simulate_ghost_imaging(
        object_mask, num_frames=num_frames, speckle_size=speckle_size)

    # Add the original image to the results for comparison
    results = {'original': {'image': object_mask,
                            'ssim': float('inf'),
                            'snr': float('inf'),
                            'method': 'original'}}

    # Add conventional ghost imaging and differential ghost imaging to the results
    # Conventional ghost imaging (GI)
    bucket_fluctuations = bucket_signals - np.mean(bucket_signals)
    gi_image = np.zeros_like(reference_frames[0])
    for i in range(num_frames):
        gi_image += bucket_fluctuations[i] * reference_frames[i]
    gi_image /= num_frames

    gi_metrics = calculate_quality_metrics(
        object_mask, gi_image, norm_type=norm_type)
    results['GI'] = {
        'image': gi_image,
        **gi_metrics,
        'method': 'GI'
    }

    # Differential ghost imaging (DGI)
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
        object_mask, dgi_image, norm_type=norm_type)
    results['DGI'] = {
        'image': dgi_image,
        **dgi_metrics,
        'method': 'DGI'
    }

    # Set default parameters
    best_threshold = 0.5
    filter_sigma = 1.0

    # Apply the selected enhancement feature
    if enhancement_choice == '5':  # newly added normalization method comparison analysis
        print("\nRunning normalization method comparison analysis...")

        # Generate a basic TCDGI image
        print("Generating a basic TCDGI image for normalization comparison...")
        tcdgi_image = tcdgi(reference_frames, bucket_signals,
                            threshold=0.5,
                            filter_sigma=1.0,
                            use_weighted_avg=True)

        if tcdgi_image is not None:
            # Compare different normalization methods
            best_method, norm_metrics = compare_normalization_methods(
                object_mask, tcdgi_image, title="Normalization Method Comparison Analysis")

            # Update the selected normalization method
            norm_type = best_method
            print(f"Based on the analysis, the recommended normalization method is {norm_type}")
        else:
            print("Unable to generate a TCDGI image for normalization analysis")

    if enhancement_choice == '1' or enhancement_choice == '4':
        # Batch threshold test
        print("\nRunning batch threshold test...")
        num_thresholds_input = input("Enter the number of thresholds to test (default: 15): ").strip()
        num_thresholds = int(
            num_thresholds_input) if num_thresholds_input and num_thresholds_input.isdigit() else 15

        sigma_input = input("Enter the Gaussian filter sigma value (default: 1.0): ").strip()
        filter_sigma = float(sigma_input) if sigma_input and sigma_input.replace(
            '.', '', 1).isdigit() else 1.0

        # Add a custom threshold range option
        custom_range = input(
            "Use a custom threshold range? (y/n, default: n): ").strip().lower() == 'y'
        min_threshold = None
        max_threshold = None

        if custom_range:
            min_threshold_input = input("Enter the minimum threshold (e.g. 20): ").strip()
            min_threshold = float(min_threshold_input) if min_threshold_input and min_threshold_input.replace(
                '.', '', 1).isdigit() else 20

            max_threshold_input = input("Enter the maximum threshold (e.g. 35): ").strip()
            max_threshold = float(max_threshold_input) if max_threshold_input and max_threshold_input.replace(
                '.', '', 1).isdigit() else 35

            print(
                f"Will test {num_thresholds} values within the threshold range {min_threshold} - {max_threshold}")

        threshold_results, best_threshold, _ = batch_threshold_test(
            reference_frames, bucket_signals, object_mask,
            filter_sigma=filter_sigma,
            num_thresholds=num_thresholds,
            custom_range=custom_range,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            norm_type=norm_type
        )

        # Merge the results
        results.update(threshold_results)

    if enhancement_choice == '2' or enhancement_choice == '4':
        # Iterative reconstruction optimization
        print("\nRunning iterative reconstruction optimization...")

        iterations_input = input("Enter the number of iterations (default: 3): ").strip()
        iterations = int(
            iterations_input) if iterations_input and iterations_input.isdigit() else 3

        if enhancement_choice != '4':  # if not applying all features, ask for the parameters separately
            sigma_input = input("Enter the Gaussian filter sigma value (default: 1.0): ").strip()
            filter_sigma = float(sigma_input) if sigma_input and sigma_input.replace(
                '.', '', 1).isdigit() else 1.0

            threshold_input = input("Enter the threshold (default: 0.5): ").strip()
            best_threshold = float(threshold_input) if threshold_input and threshold_input.replace(
                '.', '', 1).isdigit() else 0.5

        # Decide whether to apply edge enhancement
        use_edge_enhancement = enhancement_choice == '4'

        edge_factor = 0
        edge_method = 'sobel'

        if use_edge_enhancement:
            edge_factor_input = input("Enter the edge enhancement strength (0-2, default: 1.0): ").strip()
            edge_factor = float(edge_factor_input) if edge_factor_input and edge_factor_input.replace(
                '.', '', 1).isdigit() else 1.0

            print("Select an edge detection method:")
            print("1. Sobel operator (default)")
            print("2. Laplacian operator")
            print("3. Prewitt operator")

            edge_method_choice = input("Select (1-3): ").strip()
            if edge_method_choice == '2':
                edge_method = 'laplacian'
            elif edge_method_choice == '3':
                edge_method = 'prewitt'
            else:
                edge_method = 'sobel'

        # Run iterative optimization
        iterative_results = iterative_tcdgi(
            reference_frames, bucket_signals, object_mask,
            iterations=iterations,
            threshold=best_threshold,
            filter_sigma=filter_sigma,
            use_edge_enhancement=use_edge_enhancement,
            edge_factor=edge_factor,
            edge_method=edge_method
        )

        # Compute quality metrics using the selected normalization method
        for method_name, result in iterative_results.items():
            if 'image' in result and object_mask is not None:
                metrics = calculate_quality_metrics(
                    object_mask, result['image'], norm_type=norm_type)
                # Update the results dictionary
                for metric_name, value in metrics.items():
                    result[metric_name] = value

        # Merge the results
        results.update(iterative_results)

    elif enhancement_choice == '3':  # apply only edge enhancement
        print("\nRunning edge enhancement...")

        # Set the parameters
        sigma_input = input("Enter the Gaussian filter sigma value (default: 1.0): ").strip()
        filter_sigma = float(sigma_input) if sigma_input and sigma_input.replace(
            '.', '', 1).isdigit() else 1.0

        threshold_input = input("Enter the threshold (default: 0.5): ").strip()
        threshold = float(threshold_input) if threshold_input and threshold_input.replace(
            '.', '', 1).isdigit() else 0.5

        # Generate a basic TCDGI image
        base_image = tcdgi(reference_frames, bucket_signals,
                           threshold=threshold,
                           filter_sigma=filter_sigma,
                           use_weighted_avg=True)  # use weighted averaging by default

        if base_image is not None:
            # Set the edge enhancement parameters
            edge_factor_input = input("Enter the edge enhancement strength (0-2, default: 1.0): ").strip()
            edge_factor = float(edge_factor_input) if edge_factor_input and edge_factor_input.replace(
                '.', '', 1).isdigit() else 1.0

            print("Select an edge detection method:")
            print("1. Sobel operator (default)")
            print("2. Laplacian operator")
            print("3. Prewitt operator")

            edge_method_choice = input("Select (1-3): ").strip()
            if edge_method_choice == '2':
                edge_method = 'laplacian'
            elif edge_method_choice == '3':
                edge_method = 'prewitt'
            else:
                edge_method = 'sobel'

            # Apply edge enhancement
            enhanced_image = enhance_edges(base_image,
                                           enhancement_factor=edge_factor,
                                           method=edge_method)

            # Save the results
            base_metrics = calculate_quality_metrics(
                object_mask, base_image, norm_type=norm_type)
            results['TCDGI_Basic'] = {
                'image': base_image,
                **base_metrics,
                'method': 'TCDGI',
                'threshold': threshold,
                'filter_sigma': filter_sigma,
                'use_weighted_avg': True
            }

            enhanced_metrics = calculate_quality_metrics(
                object_mask, enhanced_image, norm_type=norm_type)
            results['TCDGI_Edge_Enhanced'] = {
                'image': enhanced_image,
                **enhanced_metrics,
                'method': 'TCDGI_Edge',
                'threshold': threshold,
                'filter_sigma': filter_sigma,
                'use_weighted_avg': True,
                'edge_factor': edge_factor,
                'edge_method': edge_method
            }
        else:
            print("Unable to generate a basic TCDGI image, edge enhancement failed")

    # Visualize the results using the selected normalization method
    best_method, best_ssim, best_snr = visualize_results(
        results, title="TCDGI Algorithm Comparison", norm_type=norm_type
    )

    print(f"\nEnhanced TCDGI algorithm test complete!")
    print(f"Best method: {best_method}")
    print(f"SSIM: {best_ssim:.4f}, SNR: {best_snr:.4f}")
    print(f"Normalization method: {norm_type}")
