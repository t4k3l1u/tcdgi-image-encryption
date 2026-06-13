import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from analysis import analyze_encrypted_data
from metrics import evaluate_decryption_quality
from cryptosystem import TcdgiImageCryptosystem
from TCDGI import simulate_ghost_imaging


def run_complete_tcdgi_analysis(image_path, output_dir=None, num_frames=1000, speckle_size=2.5,
                                 threshold=0.5, filter_sigma=1.0, seed=None):
    """
    Run the complete TCDGI encryption system analysis pipeline

    Parameters
    -----------
    image_path : str
        Input image path
    output_dir : str
        Output directory
    num_frames : int
        Number of frames
    speckle_size : float
        Speckle size
    threshold : float
        TCDGI algorithm threshold
    filter_sigma : float
        Standard deviation of the Gaussian filter
    seed : int, optional
        Random seed for the speckle sequence, used to reproduce the encryption key
    """
    from datetime import datetime

    # Ensure the output directory exists
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'data')

    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp and experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(output_dir, f'complete_analysis_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n=== Starting complete TCDGI encryption system analysis ({timestamp}) ===")

    # 1. Create and run the TCDGI encryption system
    from TCDGI import load_and_preprocess_image

    # Initialize the encryption system
    cryptosystem = TcdgiImageCryptosystem(
        speckle_size=speckle_size,
        num_frames=num_frames,
        threshold=threshold,
        filter_sigma=filter_sigma,
        use_weighted_avg=True,
        seed=seed
    )

    # 2. Encrypt the image
    print("\nStep 1: Encrypting the image...")
    encrypted_data, encrypted_data_path, key_path = cryptosystem.encrypt_and_save(
        image_path,
        encrypted_data_path=os.path.join(
            exp_dir, f'encrypted_data_{timestamp}.pkl'),
        key_path=os.path.join(exp_dir, f'key_{timestamp}.pkl')
    )

    # 3. Decrypt the image
    print("\nStep 2: Decrypting the image...")
    decrypted_image = cryptosystem.load_and_decrypt(
        encrypted_data_path, key_path)

    # Save the decrypted image
    decrypted_path = os.path.join(exp_dir, f'decrypted_{timestamp}.npy')
    np.save(decrypted_path, decrypted_image)

    # 4. Analyze the encrypted data
    print("\nStep 3: Analyzing the encrypted data...")
    analyze_encrypted_data(encrypted_data_path, exp_dir)

    # 5. Evaluate decryption quality
    print("\nStep 4: Evaluating decryption quality...")
    original_image = load_and_preprocess_image(image_path)
    quality_metrics = evaluate_decryption_quality(
        image_path, decrypted_image, exp_dir)

    # 6. Key sensitivity test
    print("\nStep 5: Testing key sensitivity...")

    # Create the modified key
    original_speckle_size = cryptosystem.speckle_size
    modified_key_path = os.path.join(exp_dir, f'modified_key_{timestamp}.pkl')

    # Save the original reference frames
    reference_frames_backup = cryptosystem.reference_frames.copy()

    # Slightly change the speckle size parameter
    cryptosystem.speckle_size += 0.01

    # Regenerate the reference frames with the new parameter
    # Note: use a derived seed (seed+1) rather than the main seed, so this "wrong key" is
    # reproducible while still differing from the main encryption key
    modified_seed = None if seed is None else seed + 1
    new_frames, _ = simulate_ghost_imaging(
        original_image,
        num_frames=cryptosystem.num_frames,
        speckle_size=cryptosystem.speckle_size,
        seed=modified_seed
    )

    # Save the modified key
    with open(modified_key_path, 'wb') as f:
        pickle.dump({
            'reference_frames': new_frames,
            'speckle_size': cryptosystem.speckle_size,
            'threshold': cryptosystem.threshold,
            'filter_sigma': cryptosystem.filter_sigma,
            'use_weighted_avg': cryptosystem.use_weighted_avg
        }, f)

    # Restore the original settings
    cryptosystem.speckle_size = original_speckle_size
    cryptosystem.reference_frames = reference_frames_backup

    # Try decrypting with the modified key
    try:
        # Reset the reference frames
        cryptosystem.reference_frames = None
        modified_decrypted = cryptosystem.decrypt(
            encrypted_data, modified_key_path)
        modified_decrypted_path = os.path.join(
            exp_dir, f'modified_decrypted_{timestamp}.npy')
        np.save(modified_decrypted_path, modified_decrypted)
    except Exception as e:
        print(f"Decryption with the modified key failed: {e}")
        modified_decrypted = np.random.rand(*original_image.shape)

    # Compare correct decryption results against modified-key decryption results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(decrypted_image, cmap='gray')
    plt.title('Correct decryption')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(modified_decrypted, cmap='gray')
    plt.title('Decryption with modified key')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(
        exp_dir, f'sensitivity_test_{timestamp}.png'), dpi=300)
    plt.show()

    # Compute the correlation coefficients
    correct_corr = np.corrcoef(
        original_image.flatten(), decrypted_image.flatten())[0, 1]
    incorrect_corr = np.corrcoef(
        original_image.flatten(), modified_decrypted.flatten())[0, 1]

    sensitivity_info = {
        'correct_correlation': correct_corr,
        'incorrect_correlation': incorrect_corr,
        'sensitivity_ratio': correct_corr / (incorrect_corr + 1e-10)
    }

    # 7. Multi-threshold comparison test
    print("\nStep 6: Testing the effect of different thresholds...")

    # Test different thresholds
    thresholds = [0, 0.25, 0.5, 0.75, 1.0]
    threshold_results = {}

    plt.figure(figsize=(15, 10))
    for i, threshold in enumerate(thresholds):
        # Set the new threshold
        cryptosystem.threshold = threshold
        cryptosystem.reference_frames = reference_frames_backup

        # Decrypt the image
        try:
            threshold_decrypted = cryptosystem.decrypt(
                encrypted_data, key_path)
            # Compute the correlation coefficient
            threshold_corr = np.corrcoef(
                original_image.flatten(), threshold_decrypted.flatten())[0, 1]
            threshold_results[threshold] = threshold_corr

            # Display the image
            plt.subplot(2, 3, i+1)
            plt.imshow(threshold_decrypted, cmap='gray')
            plt.title(f'threshold = {threshold} (correlation: {threshold_corr:.4f})')
            plt.axis('off')
        except Exception as e:
            print(f"Decryption failed for threshold {threshold}: {e}")
            threshold_results[threshold] = 0

    plt.tight_layout()
    plt.savefig(os.path.join(
        exp_dir, f'threshold_comparison_{timestamp}.png'), dpi=300)
    plt.show()

    # 8. Generate the final report
    print("\nStep 7: Generating the comprehensive analysis report...")

    report_path = os.path.join(exp_dir, f'analysis_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("TCDGI Encryption System Comprehensive Analysis Report\n")
        f.write("="*50 + "\n\n")

        f.write(f"Analysis time: {timestamp}\n")
        f.write(f"Original image: {image_path}\n")
        f.write(f"Experiment directory: {exp_dir}\n\n")

        f.write("Encryption parameters:\n")
        f.write("-"*30 + "\n")
        f.write(f"Speckle size: {speckle_size}\n")
        f.write(f"Number of frames: {num_frames}\n")
        f.write(f"Threshold: {cryptosystem.threshold}\n")
        f.write(f"Gaussian filter parameter: {cryptosystem.filter_sigma}\n")
        f.write(f"Weighted average: {cryptosystem.use_weighted_avg}\n\n")

        f.write("Decryption quality evaluation:\n")
        f.write("-"*30 + "\n")
        for key, value in quality_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Key sensitivity test:\n")
        f.write("-"*30 + "\n")
        for key, value in sensitivity_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Threshold comparison test:\n")
        f.write("-"*30 + "\n")
        for threshold, corr in threshold_results.items():
            f.write(f"Threshold {threshold}: correlation coefficient {corr}\n")
        f.write("\n")

        f.write("Conclusions:\n")
        f.write("-"*30 + "\n")

        # Provide recommendations based on the test results
        best_threshold = max(threshold_results.items(), key=lambda x: x[1])[0]
        f.write(f"1. Recommended threshold: {best_threshold}\n")

        if sensitivity_info['sensitivity_ratio'] > 10:
            f.write("2. Key sensitivity: Excellent - a tiny key change causes decryption to fail completely\n")
        elif sensitivity_info['sensitivity_ratio'] > 5:
            f.write("2. Key sensitivity: Good - a key change causes a significant drop in decryption quality\n")
        else:
            f.write("2. Key sensitivity: Average - consider increasing key complexity\n")

        if quality_metrics['ssim'] > 0.9:
            f.write("3. Decryption quality: Excellent\n")
        elif quality_metrics['ssim'] > 0.8:
            f.write("3. Decryption quality: Good\n")
        elif quality_metrics['ssim'] > 0.7:
            f.write("3. Decryption quality: Average\n")
        else:
            f.write("3. Decryption quality: Poor - consider adjusting parameters or increasing the number of frames\n")

    print(f"\nAnalysis complete! Comprehensive report saved to: {report_path}")
    print(f"All analysis files have been saved to: {exp_dir}")

    return {
        'experiment_dir': exp_dir,
        'quality_metrics': quality_metrics,
        'sensitivity_info': sensitivity_info,
        'threshold_results': threshold_results,
        'report_path': report_path
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='TCDGI image encryption pipeline')
    parser.add_argument('--mode', type=str, default='full', choices=['analyze', 'full'],
                        help='analyze: inspect an existing encrypted_data .pkl; full: run the end-to-end pipeline on an image')
    parser.add_argument('--input', type=str, required=True, help='input file path (image for "full", .pkl for "analyze")')
    parser.add_argument('--output', type=str, default='./data', help='output directory')
    parser.add_argument('--frames', type=int, default=1000, help='number of simulated frames ("full" mode)')
    parser.add_argument('--speckle', type=float,
                        default=2.5, help='speckle size ("full" mode)')
    parser.add_argument('--threshold', type=float,
                        default=0.5, help='TCDGI threshold ("full" mode)')
    parser.add_argument('--filter-sigma', type=float,
                        default=1.0, help='Gaussian filter sigma ("full" mode)')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for the speckle sequence ("full" mode); omit for a non-reproducible run')

    args = parser.parse_args()

    if args.mode == 'analyze':
        if args.input.endswith('.pkl') and 'encrypted_data' in args.input:
            analyze_encrypted_data(args.input, args.output)
        else:
            print("Expected an 'encrypted_data*.pkl' file for --mode analyze")
            exit(1)

    elif args.mode == 'full':
        run_complete_tcdgi_analysis(
            args.input, args.output, args.frames, args.speckle,
            threshold=args.threshold, filter_sigma=args.filter_sigma, seed=args.seed)
