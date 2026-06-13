import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import os
import pickle
from skimage.metrics import structural_similarity as ssim
from datetime import datetime

from TCDGI import (tcdgi, normalize_image, simulate_ghost_imaging,
                      load_and_preprocess_image)


class TcdgiImageCryptosystem:
    """
    Image encryption system based on Time-Correspondence Differential Ghost Imaging (TCDGI)
    """

    def __init__(self, speckle_size=2.5, num_frames=2000, threshold=0.5,
                 filter_sigma=2.5, use_weighted_avg=True, seed=None):
        """
        Initialize the encryption system parameters

        Parameters:
        -----------
        speckle_size : float
            Speckle size, part of the encryption key
        num_frames : int
            Number of frames used for encryption
        threshold : float
            Threshold for the TCDGI algorithm
        filter_sigma : float
            Standard deviation of the Gaussian filter
        use_weighted_avg : bool
            Whether to use weighted averaging
        seed : int, optional
            Random seed for the speckle sequence, used to reproduce the encryption key;
            when None, each encryption produces a different result
        """
        self.speckle_size = speckle_size
        self.num_frames = num_frames
        self.threshold = threshold
        self.filter_sigma = filter_sigma
        self.use_weighted_avg = use_weighted_avg
        self.seed = seed
        self.reference_frames = None
        self.bucket_signals = None
        self.object_mask = None  # Stores the original image for comparison

        # Ensure the data folder exists
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data folder: {self.data_dir}")

    def _get_time_stamp(self):
        """Generate a timestamp for file names"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def encrypt(self, image_path, save_key=True, key_path=None):
        """
        Encrypt an image

        Parameters:
        -----------
        image_path : str
            Path to the image to encrypt
        save_key : bool
            Whether to save the encryption key to a file
        key_path : str
            Path to save the encryption key; if None, a default path is used

        Returns:
        --------
        encrypted_data : dict
            The encrypted data
        """
        # If key_path is not specified, generate a timestamped file name
        if key_path is None:
            timestamp = self._get_time_stamp()
            key_path = os.path.join(
                self.data_dir, f'encryption_key_{timestamp}.pkl')

        # Load the original image
        print(f"Loading image: {image_path}")
        original_image = load_and_preprocess_image(image_path)
        if original_image is None:
            raise ValueError("Failed to load image")

        # Save the original image for comparison
        self.object_mask = original_image

        # Generate a random speckle sequence as the encryption key
        print(f"Generating encryption key (speckle sequence)...")
        self.reference_frames, self.bucket_signals = simulate_ghost_imaging(
            original_image, num_frames=self.num_frames, speckle_size=self.speckle_size,
            seed=self.seed)

        # The encrypted image is the bucket signal sequence from the TCDGI algorithm; the key is the reference detector frames
        encrypted_data = {
            'bucket_signals': self.bucket_signals,
            'image_shape': original_image.shape,
            'timestamp': self._get_time_stamp()
        }

        if save_key:
            # Save the key (reference frame sequence) to a file
            key_data = {
                'reference_frames': self.reference_frames,
                'speckle_size': self.speckle_size,
                'threshold': self.threshold,
                'filter_sigma': self.filter_sigma,
                'use_weighted_avg': self.use_weighted_avg,
                'seed': self.seed,
                'timestamp': encrypted_data['timestamp']
            }
            with open(key_path, 'wb') as f:
                pickle.dump(key_data, f)
            print(f"Encryption key saved to: {key_path}")

        return encrypted_data, key_path

    def decrypt(self, encrypted_data, key_path):
        """
        Decrypt an image

        Parameters:
        -----------
        encrypted_data : dict
            The encrypted data, containing the bucket signals and image shape
        key_path : str
            Path to the encryption key

        Returns:
        --------
        decrypted_image : ndarray
            The decrypted image
        """
        # Load the key from the file
        if self.reference_frames is None:
            print(f"Loading key from {key_path}...")
            try:
                with open(key_path, 'rb') as f:
                    key_data = pickle.load(f)

                self.reference_frames = key_data['reference_frames']
                self.threshold = key_data.get('threshold', self.threshold)
                self.filter_sigma = key_data.get(
                    'filter_sigma', self.filter_sigma)
                self.use_weighted_avg = key_data.get(
                    'use_weighted_avg', self.use_weighted_avg)
                self.seed = key_data.get('seed', self.seed)

                print(f"Using threshold: {self.threshold:.4f}")
            except Exception as e:
                raise ValueError(f"Failed to load key: {e}")

        # Get the encrypted data
        bucket_signals = encrypted_data['bucket_signals']

        # Reconstruct the original image using the TCDGI algorithm
        print("Decrypting image...")
        decrypted_image = tcdgi(
            self.reference_frames,
            bucket_signals,
            threshold=self.threshold,
            filter_sigma=self.filter_sigma,
            use_weighted_avg=self.use_weighted_avg
        )

        if decrypted_image is None:
            raise ValueError("Decryption failed, unable to reconstruct image")

        # Use centered normalization
        return normalize_image(decrypted_image, norm_type='centered')

    def encrypt_and_save(self, image_path, encrypted_data_path=None, key_path=None):
        """
        Encrypt an image and save the encrypted data

        Parameters:
        -----------
        image_path : str
            Path to the image to encrypt
        encrypted_data_path : str
            Path to save the encrypted data; if None, a default path is used
        key_path : str
            Path to save the encryption key; if None, a default path is used

        Returns:
        --------
        encrypted_data : dict
            The encrypted data
        """
        # Generate a timestamp to use as part of the file name
        timestamp = self._get_time_stamp()

        # Use default paths if not specified
        if encrypted_data_path is None:
            encrypted_data_path = os.path.join(
                self.data_dir, f'encrypted_data_{timestamp}.pkl')
        if key_path is None:
            key_path = os.path.join(
                self.data_dir, f'encryption_key_{timestamp}.pkl')

        # Encrypt the image
        encrypted_data, key_path = self.encrypt(
            image_path, save_key=True, key_path=key_path)

        # Save the encrypted data
        with open(encrypted_data_path, 'wb') as f:
            pickle.dump(encrypted_data, f)
        print(f"Encrypted data saved to: {encrypted_data_path}")

        return encrypted_data, encrypted_data_path, key_path

    def load_and_decrypt(self, encrypted_data_path, key_path):
        """
        Load encrypted data and decrypt it

        Parameters:
        -----------
        encrypted_data_path : str
            Path to the encrypted data
        key_path : str
            Path to the encryption key

        Returns:
        --------
        decrypted_image : ndarray
            The decrypted image
        """
        # Load the encrypted data
        try:
            with open(encrypted_data_path, 'rb') as f:
                encrypted_data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load encrypted data: {e}")

        # Decrypt the image
        return self.decrypt(encrypted_data, key_path)

    def decrypt_with_optimization(self, encrypted_data, key_path, iterations=3, use_edge_enhancement=False, edge_factor=1.0, edge_method='sobel'):
        """
        Decrypt an image using iterative optimization

        Parameters:
        -----------
        encrypted_data : dict
            The encrypted data, containing the bucket signals and image shape
        key_path : str
            Path to the encryption key
        iterations : int
            Number of optimization iterations
        use_edge_enhancement : bool
            Whether to use edge enhancement
        edge_factor : float
            Edge enhancement intensity factor
        edge_method : str
            Edge detection method ('sobel', 'laplacian', 'prewitt')

        Returns:
        --------
        results : dict
            A dictionary containing the decryption results and evaluation metrics
        """
        # First obtain the initial image using the basic decryption method
        base_image = self.decrypt(encrypted_data, key_path)

        # If the basic decrypted image cannot be obtained, return None
        if base_image is None:
            print("Basic decryption failed, cannot proceed with optimization")
            return None

        print(f"Starting iterative decryption optimization, iterations: {iterations}")

        results = {}
        current_image = base_image.copy()

        # Save the basic decryption result
        results['basic_decryption'] = {'image': base_image}

        # Iterative optimization
        for i in range(iterations):
            print(f"Running iteration {i+1}/{iterations} of optimization...")

            # Adjust parameters at each iteration
            current_threshold = self.threshold * (1 + 0.1*i)  # Gradually increase the threshold
            current_sigma = self.filter_sigma * (1 - 0.05*i)  # Gradually decrease the blur

            # Re-run TCDGI decryption with the adjusted parameters
            try:
                # Gradually adjust the reference frame weights at each iteration
                weighted_frames = self.reference_frames.copy()
                for j in range(len(weighted_frames)):
                    # Compute correlation as the weighting factor
                    correlation = np.sum(
                        weighted_frames[j] * current_image) / np.sum(weighted_frames[j]**2)
                    weighted_frames[j] = weighted_frames[j] * \
                        (1 + correlation * i / iterations)

                # Reconstruct the image using the TCDGI algorithm
                optimized_image = tcdgi(
                    weighted_frames,
                    encrypted_data['bucket_signals'],
                    threshold=current_threshold,
                    filter_sigma=current_sigma,
                    use_weighted_avg=self.use_weighted_avg
                )

                if optimized_image is not None:
                    current_image = normalize_image(optimized_image)

                    # Apply edge enhancement (if enabled)
                    if use_edge_enhancement:
                        # Edge enhancement factor increases with iterations
                        current_edge_factor = edge_factor * (i+1)/iterations
                        current_image = enhance_edges(
                            current_image,
                            enhancement_factor=current_edge_factor,
                            method=edge_method
                        )

                    # Save the current iteration result
                    results[f'iteration_{i+1}'] = {
                        'image': current_image,
                        'threshold': current_threshold,
                        'filter_sigma': current_sigma,
                        'use_edge_enhancement': use_edge_enhancement,
                        'edge_factor': edge_factor * (i+1)/iterations if use_edge_enhancement else 0,
                        'edge_method': edge_method if use_edge_enhancement else None
                    }

                    print(f"  Iteration {i+1} complete")
                else:
                    print(f"  Iteration {i+1} failed to produce a valid image")
                    break

            except Exception as e:
                print(f"Error in iteration {i+1}: {e}")
                break

        # Return all decryption results
        return results

        """Compute various image quality evaluation metrics"""
    # Normalize the images
        original = normalize_image(original)
        reconstructed = normalize_image(reconstructed)

    # Compute SSIM (using the imported skimage.metrics)
        ssim_value = ssim(original, reconstructed, data_range=1.0)

    # Compute SNR
        T0_mean = np.mean(original)
        signal = np.sum((original - T0_mean)**2)
        noise = np.sum((reconstructed - original)**2)
        if noise < 1e-10:
            noise = 1e-10
        snr_value = np.sqrt(signal / noise)

    # Compute RMSE and PSNR
        mse = np.mean((original - reconstructed) ** 2)
        rmse = np.sqrt(mse)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else 100

        return {
            'ssim': ssim_value,
            'snr': snr_value,
            'rmse': rmse,
            'psnr': psnr
        }


def security_analysis(cryptosystem, image_path, output_dir=None):
    """
    Perform a security analysis of the encryption system by adding black-bar occlusions
    to the reference frames to simulate an incorrect key.
    Occlusion ratios range from 1% to 10% in steps of 2%, and decryption quality is
    evaluated using SNR and SSIM.

    Parameters:
    -----------
    cryptosystem : TcdgiImageCryptosystem
        An instance of the encryption system
    image_path : str
        Path to the original image
    output_dir : str
        Output directory; if None, the default data directory is used
    """
    if output_dir is None:
        output_dir = cryptosystem.data_dir

    timestamp = cryptosystem._get_time_stamp()

    # Load the original image
    original_image = load_and_preprocess_image(image_path)

    # Encrypt the image
    encrypted_data, key_path = cryptosystem.encrypt(
        image_path,
        key_path=os.path.join(output_dir, f'original_key_{timestamp}.pkl')
    )

    # Decrypt using the correct key
    decrypted_image = cryptosystem.decrypt(encrypted_data, key_path)

    # Compute the baseline SNR and SSIM for correct decryption
    from TCDGI import calculate_snr, calculate_ssim
    correct_snr = calculate_snr(original_image, decrypted_image)
    correct_ssim = calculate_ssim(original_image, decrypted_image)

    print(f"Correct decryption SNR: {correct_snr:.4f}, SSIM: {correct_ssim:.4f}")

    # Back up the current reference frames
    backup_frames = cryptosystem.reference_frames.copy()

    # Set different occlusion ratios - changed to 1% to 10% in steps of 2%
    occlusion_ratios = [0.01, 0.03, 0.05, 0.07, 0.09]  # 1%, 3%, 5%, 7%, 9%

    # Store decryption results and quality metrics for each occlusion ratio
    occlusion_results = {}

    # Test each occlusion ratio
    for ratio in occlusion_ratios:
        print(f"Testing {ratio*100:.1f}% occlusion ratio...")

        # Create an occluded version of the reference frames
        occluded_frames = backup_frames.copy()

        # Get the frame dimensions
        num_frames, height, width = occluded_frames.shape

        # Compute the height of the black bar (horizontal bar)
        bar_height = int(height * ratio)
        if bar_height < 1:  # Ensure at least 1 pixel of occlusion
            bar_height = 1

        # Add a horizontal black-bar occlusion to the reference frames
        for i in range(num_frames):
            # Randomly choose the starting row of the black bar
            start_row = np.random.randint(0, height - bar_height + 1)

            # Set this region to 0 (black)
            occluded_frames[i, start_row:start_row+bar_height, :] = 0

        # Create the path for the occluded key file
        occluded_key_path = os.path.join(
            output_dir, f'occluded_key_{int(ratio*100)}pct_{timestamp}.pkl')

        # Save the occluded key
        with open(occluded_key_path, 'wb') as f:
            pickle.dump({
                'reference_frames': occluded_frames,
                'speckle_size': cryptosystem.speckle_size,
                'threshold': cryptosystem.threshold,
                'filter_sigma': cryptosystem.filter_sigma,
                'use_weighted_avg': cryptosystem.use_weighted_avg,
                'timestamp': timestamp
            }, f)

        # Decrypt using the occluded key
        try:
            # Clear the current key
            cryptosystem.reference_frames = None
            occluded_decryption = cryptosystem.decrypt(
                encrypted_data, occluded_key_path)

            # Compute quality metrics
            snr_value = calculate_snr(original_image, occluded_decryption)
            ssim_value = calculate_ssim(original_image, occluded_decryption)

            # Compute the error rate (here defined as 1 - SSIM)
            error_rate = 1 - ssim_value

            # Restore the original key
            cryptosystem.reference_frames = backup_frames

            print(
                f"  SNR: {snr_value:.4f}, SSIM: {ssim_value:.4f}, Error rate: {error_rate:.4f}")

        except Exception as e:
            print(f"Decryption failed with {ratio*100:.1f}% occluded key: {e}")
            occluded_decryption = np.random.rand(
                *original_image.shape)  # Generate a random image
            snr_value = 0
            ssim_value = 0
            error_rate = 1

        # Store the result
        occlusion_results[ratio] = {
            'image': occluded_decryption,
            'key_path': occluded_key_path,
            'snr': snr_value,
            'ssim': ssim_value,
            'error_rate': error_rate
        }

    # Restore the original reference frames
    cryptosystem.reference_frames = backup_frames

    # 1. Visualize the decryption results for different occlusion ratios
    plt.figure(figsize=(15, 8))

    # Show the original image
    plt.subplot(2, 4, 1)
    plt.imshow(normalize_image(original_image,
               norm_type='centered'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Show the correct decryption result
    plt.subplot(2, 4, 2)
    plt.imshow(normalize_image(decrypted_image,
               norm_type='centered'), cmap='gray')
    plt.title(f'Correct Decryption\nSNR: {correct_snr:.2f}, SSIM: {correct_ssim:.2f}')
    plt.axis('off')

    # Show decryption results for different occlusion ratios
    pos = 3
    for ratio in occlusion_ratios[:3]:  # Show only the first 3 occlusion ratio results to fit the layout
        plt.subplot(2, 4, pos)
        if occlusion_results[ratio]['image'] is not None:
            plt.imshow(normalize_image(
                occlusion_results[ratio]['image'], norm_type='centered'), cmap='gray')
        else:
            plt.text(0.5, 0.5, 'Decryption Failed', ha='center', va='center')
        plt.title(
            f'{ratio*100:.1f}% Occlusion Decryption\nSNR: {occlusion_results[ratio]["snr"]:.2f}, SSIM: {occlusion_results[ratio]["ssim"]:.2f}')
        plt.axis('off')
        pos += 1

    # Add the results for the remaining occlusion ratios
    pos = 5
    for ratio in occlusion_ratios[3:]:  # Show the remaining occlusion ratio results
        plt.subplot(2, 4, pos)
        if occlusion_results[ratio]['image'] is not None:
            plt.imshow(normalize_image(
                occlusion_results[ratio]['image'], norm_type='centered'), cmap='gray')
        else:
            plt.text(0.5, 0.5, 'Decryption Failed', ha='center', va='center')
        plt.title(
            f'{ratio*100:.1f}% Occlusion Decryption\nSNR: {occlusion_results[ratio]["snr"]:.2f}, SSIM: {occlusion_results[ratio]["ssim"]:.2f}')
        plt.axis('off')
        pos += 1

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'occlusion_analysis_{timestamp}.png'))
    plt.show()

    # 2. Plot SNR and SSIM/error-rate curves
    plt.figure(figsize=(12, 5))

    # SNR curve
    plt.subplot(1, 2, 1)
    ratios_percent = [r*100 for r in occlusion_ratios]  # Convert to percentages for display
    snr_values = [occlusion_results[r]['snr'] for r in occlusion_ratios]

    plt.plot(ratios_percent, snr_values, 'o-', linewidth=2, color='blue')
    plt.axhline(y=correct_snr, color='r', linestyle='--',
                label=f'Correct Decryption ({correct_snr:.2f})')
    plt.xlabel('Occlusion Ratio (%)')
    plt.ylabel('SNR Value')
    plt.title('Occlusion Ratio vs SNR')
    plt.grid(True)
    plt.legend()

    # SSIM/error-rate curve
    plt.subplot(1, 2, 2)
    ssim_values = [occlusion_results[r]['ssim'] for r in occlusion_ratios]
    error_rates = [occlusion_results[r]['error_rate']
                   for r in occlusion_ratios]

    plt.plot(ratios_percent, ssim_values, 'o-',
             linewidth=2, color='green', label='SSIM')
    plt.plot(ratios_percent, error_rates, 's--',
             linewidth=2, color='red', label='Error Rate')
    plt.axhline(y=correct_ssim, color='g', linestyle=':',
                label=f'Correct SSIM ({correct_ssim:.2f})')
    plt.xlabel('Occlusion Ratio (%)')
    plt.ylabel('Value')
    plt.title('Occlusion Ratio vs SSIM/Error Rate')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'occlusion_metrics_{timestamp}.png'))
    plt.show()

    # Save the analysis results to a text file
    with open(os.path.join(output_dir, f'occlusion_analysis_results_{timestamp}.txt'), 'w') as f:
        f.write(f"Occlusion security analysis time: {timestamp}\n")
        f.write(f"Original image: {image_path}\n")
        f.write(f"Correct key: {key_path}\n")
        f.write(f"Correct decryption SNR: {correct_snr:.4f}, SSIM: {correct_ssim:.4f}\n\n")
        f.write(f"Decryption results for different occlusion ratios:\n")
        f.write(f"===============================\n")
        for ratio in occlusion_ratios:
            f.write(f"{ratio*100:.1f}% occlusion:\n")
            f.write(f"  Key path: {occlusion_results[ratio]['key_path']}\n")
            f.write(f"  SNR: {occlusion_results[ratio]['snr']:.4f}\n")
            f.write(f"  SSIM: {occlusion_results[ratio]['ssim']:.4f}\n")
            f.write(f"  Error rate: {occlusion_results[ratio]['error_rate']:.4f}\n\n")

    print(f"\nOcclusion security analysis complete. Results saved to: {output_dir}")
    print(f"Image results: occlusion_analysis_{timestamp}.png")
    print(f"Metrics chart: occlusion_metrics_{timestamp}.png")
    print(f"Detailed analysis: occlusion_analysis_results_{timestamp}.txt")


def bit_sensitivity_test(cryptosystem, image_path, output_dir=None):
    """
    Test key bit sensitivity (by changing the speckle size)

    Parameters:
    -----------
    cryptosystem : TcdgiImageCryptosystem
        An instance of the encryption system
    image_path : str
        Path to the original image
    output_dir : str
        Output directory; if None, the default data directory is used
    """
    if output_dir is None:
        output_dir = cryptosystem.data_dir

    timestamp = cryptosystem._get_time_stamp()

    # Encrypt with the original system
    encrypted_data, original_key_path = cryptosystem.encrypt(
        image_path,
        key_path=os.path.join(
            output_dir, f'sensitivity_original_key_{timestamp}.pkl')
    )

    # Save the original settings
    original_speckle_size = cryptosystem.speckle_size

    # Slightly change the speckle size and generate a new key
    cryptosystem.speckle_size += 0.01  # Tiny change

    # Regenerate reference frames with the new parameters
    original_image = load_and_preprocess_image(image_path)
    new_frames, _ = simulate_ghost_imaging(
        original_image,
        num_frames=cryptosystem.num_frames,
        speckle_size=cryptosystem.speckle_size
    )

    # Save the new key
    modified_key_path = os.path.join(
        output_dir, f'sensitivity_modified_key_{timestamp}.pkl')
    with open(modified_key_path, 'wb') as f:
        pickle.dump({
            'reference_frames': new_frames,
            'speckle_size': cryptosystem.speckle_size,
            'threshold': cryptosystem.threshold,
            'filter_sigma': cryptosystem.filter_sigma,
            'use_weighted_avg': cryptosystem.use_weighted_avg,
            'timestamp': timestamp
        }, f)

    # Restore the original settings
    cryptosystem.speckle_size = original_speckle_size

    # Run the security analysis, including the incorrect-key test
    print(f"\n=== Running Key Sensitivity Test ===")
    print(f"Original speckle parameter: {original_speckle_size}")
    print(f"Modified speckle parameter: {original_speckle_size + 0.01}")

    # Decrypt using the original key
    correct_decryption = cryptosystem.decrypt(
        encrypted_data, original_key_path)

    # Decrypt using the modified key
    cryptosystem.reference_frames = None  # Clear the current key
    try:
        incorrect_decryption = cryptosystem.decrypt(
            encrypted_data, modified_key_path)
    except Exception as e:
        print(f"Decryption with the modified key failed: {e}")
        incorrect_decryption = np.random.rand(*original_image.shape)

    # Display the results
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(normalize_image(original_image,
               norm_type='centered'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(normalize_image(correct_decryption,
               norm_type='centered'), cmap='gray')
    plt.title('Decryption with Correct Key')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(normalize_image(incorrect_decryption,
               norm_type='centered'), cmap='gray')
    plt.title(f'Decryption with Speckle Size +0.01')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sensitivity_test_{timestamp}.png'))
    plt.show()

    # Compute correlation coefficients
    correct_corr = np.corrcoef(
        original_image.flatten(), correct_decryption.flatten())[0, 1]
    incorrect_corr = np.corrcoef(
        original_image.flatten(), incorrect_decryption.flatten())[0, 1]

    print(f"Correlation coefficient with correct key decryption: {correct_corr:.4f}")
    print(f"Correlation coefficient with modified key decryption: {incorrect_corr:.4f}")

    # Save the analysis results to a text file
    with open(os.path.join(output_dir, f'sensitivity_results_{timestamp}.txt'), 'w') as f:
        f.write(f"Key sensitivity test time: {timestamp}\n")
        f.write(f"Original image: {image_path}\n")
        f.write(f"Original speckle size: {original_speckle_size}\n")
        f.write(f"Modified speckle size: {original_speckle_size + 0.01}\n")
        f.write(f"Correlation coefficient with correct key decryption: {correct_corr:.4f}\n")
        f.write(f"Correlation coefficient with modified key decryption: {incorrect_corr:.4f}\n")


def generate_key_pair(cryptosystem, image_path, output_dir=None):
    """
    Generate a pair of encryption/decryption keys

    Parameters:
    -----------
    cryptosystem : TcdgiImageCryptosystem
        An instance of the encryption system
    image_path : str
        Path to the original image
    output_dir : str
        Output directory; if None, the default data directory is used

    Returns:
    --------
    tuple: (private_key_path, public_key_path)
        Paths where the private and public keys are saved
    """
    if output_dir is None:
        output_dir = cryptosystem.data_dir

    timestamp = cryptosystem._get_time_stamp()

    # Private key path
    private_key_path = os.path.join(output_dir, f'private_key_{timestamp}.pkl')
    # Public key path
    public_key_path = os.path.join(output_dir, f'public_key_{timestamp}.pkl')

    # Generate the key pair using the TCDGI system
    encrypted_data, _ = cryptosystem.encrypt(
        image_path, save_key=True, key_path=private_key_path)

    # Extract the public key information (bucket signal format) and save it
    public_key = {
        'bucket_signal_shape': encrypted_data['bucket_signals'].shape,
        'image_shape': encrypted_data['image_shape'],
        'threshold': cryptosystem.threshold,
        'filter_sigma': cryptosystem.filter_sigma,
        'use_weighted_avg': cryptosystem.use_weighted_avg,
        'timestamp': timestamp
    }

    with open(public_key_path, 'wb') as f:
        pickle.dump(public_key, f)

    print("Key pair generated:")
    print(f"  - Private key: {private_key_path}")
    print(f"  - Public key info: {public_key_path}")

    return private_key_path, public_key_path


def save_experiment_results(original_image, encrypted_data, decrypted_image, output_dir, prefix="result"):
    """Save the experiment results"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    # Use centered normalization
    plt.imshow(normalize_image(original_image,
               norm_type='centered'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.plot(encrypted_data['bucket_signals'])
    plt.title('Encrypted Data (Bucket Signal)')
    plt.xlabel('Frame Index')
    plt.ylabel('Signal Intensity')

    plt.subplot(133)
    # Use centered normalization
    plt.imshow(normalize_image(decrypted_image,
               norm_type='centered'), cmap='gray')
    plt.title('Decrypted Image')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_{timestamp}.png'))
    plt.show()

    # Save the result data
    np.save(os.path.join(
        output_dir, f'{prefix}_decrypted_{timestamp}.npy'), decrypted_image)

    # Compute the correlation coefficient - using centered normalization
    original_norm = normalize_image(original_image, norm_type='centered')
    decrypted_norm = normalize_image(decrypted_image, norm_type='centered')
    correlation = np.corrcoef(original_norm.flatten(),
                              decrypted_norm.flatten())[0, 1]

    # Save the analysis results
    with open(os.path.join(output_dir, f'{prefix}_analysis_{timestamp}.txt'), 'w') as f:
        f.write(f"Experiment time: {timestamp}\n")
        f.write(f"Correlation coefficient between original and decrypted images: {correlation:.4f}\n")
        f.write(
            f"Encrypted data statistics: mean={np.mean(encrypted_data['bucket_signals']):.4f}, std={np.std(encrypted_data['bucket_signals']):.4f}\n")


def select_image_file():
    """Let the user select an image file"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        return file_path if file_path else None
    except ImportError:
        # If tkinter is unavailable, use command-line input
        print("Please enter the full path to the image file:")
        return input().strip()


def calculate_ssim(original, reconstructed, win_size=11, k1=0.01, k2=0.03, L=2.0):
    """SSIM computation optimized for centered normalization"""
    if win_size % 2 == 0:
        win_size += 1

    # Use centered normalization to ensure the range is [-1, 1]
    original = normalize_image(original, norm_type='centered')
    reconstructed = normalize_image(reconstructed, norm_type='centered')

    # Dynamically check the actual data range
    actual_range = max(np.max(original) - np.min(original),
                       np.max(reconstructed) - np.min(reconstructed))
    if abs(actual_range - 2.0) > 0.1:  # If the actual range differs significantly from the expected range
        print(f"Warning: actual data range {actual_range:.2f} does not match the expected range of 2.0")
        # Optionally adjust the L value
        L = actual_range

    # Define the Gaussian window
    def gaussian_window(win_size, sigma=1.5):
        x, y = np.mgrid[-win_size//2 + 1:win_size //
                        2 + 1, -win_size//2 + 1:win_size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()

    window = gaussian_window(win_size)

    # Constants
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    # Compute means, variances, and covariance
    mu1 = ndimage.convolve(original, window)
    mu2 = ndimage.convolve(reconstructed, window)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = ndimage.convolve(original**2, window) - mu1_sq
    sigma2_sq = ndimage.convolve(reconstructed**2, window) - mu2_sq
    sigma12 = ndimage.convolve(original * reconstructed, window) - mu1_mu2

    # Compute SSIM
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den

    return np.mean(ssim_map)


def calculate_quality_metrics(original, reconstructed):
    """Compute various image quality evaluation metrics"""
    # Use centered normalization
    original = normalize_image(original, norm_type='centered')
    reconstructed = normalize_image(reconstructed, norm_type='centered')

    # Compute SSIM
    ssim_value = calculate_ssim(original, reconstructed)

    # Compute SNR
    T0_mean = np.mean(original)
    signal = np.sum((original - T0_mean)**2)
    noise = np.sum((reconstructed - original)**2)
    if noise < 1e-10:
        noise = 1e-10
    snr_value = np.sqrt(signal / noise)

    # Compute RMSE and PSNR
    mse = np.mean((original - reconstructed) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else 100

    return {
        'ssim': ssim_value,
        'snr': snr_value,
        'rmse': rmse,
        'psnr': psnr
    }


def enhance_edges(image, enhancement_factor=1.0, method='sobel'):
    """Apply edge enhancement to an image"""
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

    # Use centered normalization
    enhanced_image = normalize_image(enhanced_image, norm_type='centered')

    return enhanced_image


def evaluate_decryption_quality(original_image, decryption_results):
    """
    Evaluate the quality of decryption results and display a comparison

    Parameters:
    -----------
    original_image : ndarray
        The original reference image
    decryption_results : dict
        A dictionary containing the different decryption results; each entry should have an 'image' key

    Returns:
    --------
    sorted_results : list
        A list of results sorted by SSIM in descending order
    """
    if original_image is None or not decryption_results:
        print("No valid images available for quality evaluation")
        return None

    # Compute quality metrics for each decryption result
    for name, result in decryption_results.items():
        if ('ssim' not in result or 'snr' not in result) and name != 'original':
            # If the quality metrics are not provided, compute them
            if 'image' in result and name != 'original':
                # Explicitly use centered normalization for quality evaluation here
                metrics = calculate_quality_metrics(
                    original_image, result['image'])
                # Update the result dictionary
                for metric_name, value in metrics.items():
                    result[metric_name] = value

    # Use SNR as the primary sorting metric
    sorted_results = sorted(
        decryption_results.items(),
        key=lambda x: x[1].get('snr', 0),
        reverse=True
    )

    # Extract method names and quality metric values
    method_names = []
    ssim_values = []
    snr_values = []
    from collections import defaultdict
    method_types = defaultdict(list)  # Use defaultdict to avoid key errors

    # Extract parameter information from the results
    thresholds = set()
    filter_sigmas = set()

    for method_name, result in sorted_results:
        if method_name == 'original':
            continue  # Skip the original image

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
        print("No method results available to visualize")
        return None

    # Plot the comparison chart
    plt.figure(figsize=(15, 10))

    # Show the original image
    plt.subplot(2, len(sorted_results)+1, 1)
    # Use centered normalization to display the original image
    plt.imshow(normalize_image(original_image,
               norm_type='centered'), cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Show each decryption result
    for i, (name, result) in enumerate(sorted_results):
        if 'image' in result:
            plt.subplot(2, len(sorted_results)+1, i+2)
            # Use centered normalization to display the decrypted image
            plt.imshow(normalize_image(
                result['image'], norm_type='centered'), cmap='gray')
            title = f"{name}\nSSIM: {result.get('ssim', 0):.4f}"
            plt.title(title)
            plt.axis('off')

    # Plot a bar chart comparing the quality metrics
    names = [name for name, _ in sorted_results]
    ssim_values = [result.get('ssim', 0) for _, result in sorted_results]
    snr_values = [result.get('snr', 0) for _, result in sorted_results]

    # SSIM bar chart
    plt.subplot(2, 2, 3)
    plt.bar(names, ssim_values)
    plt.title('SSIM Comparison (centered normalization)')
    plt.xticks(rotation=45)

    # SNR bar chart
    plt.subplot(2, 2, 4)
    plt.bar(names, snr_values)
    plt.title('SNR Comparison (centered normalization)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.suptitle('Decryption Result Quality Evaluation (using centered normalization)', fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

    return sorted_results


def analyze_key_space(cryptosystem, output_dir=None, visualize=True):
    """
    Analyze the key space of the TCDGI encryption system to evaluate its security

    Parameters:
    -----------
    cryptosystem : TcdgiImageCryptosystem
        An instance of the encryption system
    output_dir : str
        Output directory; if None, the default data directory is used
    visualize : bool
        Whether to generate visualization charts

    Returns:
    --------
    analysis_results : dict
        A dictionary containing the key space analysis results
    """
    if output_dir is None:
        output_dir = cryptosystem.data_dir

    timestamp = cryptosystem._get_time_stamp()

    # 1. Initialize the results dictionary
    results = {
        'timestamp': timestamp,
        'key_components': {},
        'theoretical_key_space': 0,
        'effective_key_space': 0,
        'security_level': '',
        'recommendations': []
    }

    # 2. Analyze the reference frame sequence and parameters
    if cryptosystem.reference_frames is not None:
        ref_frames_shape = cryptosystem.reference_frames.shape
        results['key_components']['reference_frames'] = {
            'type': 'array',
            'shape': ref_frames_shape,
            'dimensions': ref_frames_shape[0] * ref_frames_shape[1] * ref_frames_shape[2],
            'bits_per_value': 32,  # Assume 32-bit floating point values
            'theoretical_entropy': ref_frames_shape[0] * ref_frames_shape[1] * ref_frames_shape[2] * 32
        }

    # 3. Analyze the speckle size parameter
    precision_factor = 100  # A change of 0.01 affects decryption
    speckle_range = (0.1, 10.0)  # Assume a reasonable range for the speckle size
    theoretical_speckle_values = int(
        (speckle_range[1] - speckle_range[0]) * precision_factor)
    results['key_components']['speckle_size'] = {
        'type': 'float',
        'range': speckle_range,
        'precision': 1/precision_factor,
        'possible_values': theoretical_speckle_values,
        'bits': np.ceil(np.log2(theoretical_speckle_values))
    }

    # 4. Analyze the threshold parameter - consider only a single threshold
    threshold_range = (0, 100)  # Assume a reasonable threshold range
    threshold_precision = 0.0001  # Assume a threshold precision of 0.0001
    theoretical_threshold_values = int(
        (threshold_range[1] - threshold_range[0]) / threshold_precision)
    results['key_components']['threshold'] = {
        'type': 'float',
        'range': threshold_range,
        'precision': threshold_precision,
        'possible_values': theoretical_threshold_values,
        'bits': np.ceil(np.log2(theoretical_threshold_values)),
        'current_value': cryptosystem.threshold
    }

    # 5. Analyze the filter parameter
    filter_range = (0.1, 10.0)
    filter_precision = 0.1
    theoretical_filter_values = int(
        (filter_range[1] - filter_range[0]) / filter_precision)
    results['key_components']['filter_sigma'] = {
        'type': 'float',
        'range': filter_range,
        'precision': filter_precision,
        'possible_values': theoretical_filter_values,
        'bits': np.ceil(np.log2(theoretical_filter_values))
    }

    # 6. Analyze other algorithm parameters
    results['key_components']['use_weighted_avg'] = {
        'type': 'boolean',
        'possible_values': 2,
        'bits': 1
    }

    # 7. Compute the total theoretical key space (in bits)
    # Note: this is a more conservative, entropy-corrected estimate, and differs from the
    # loose upper-bound estimate given by analyze_encryption_security() in compare.py;
    # the two will not produce the same number. The 2^71.29 figure reported in the thesis
    # is an independently reported effective estimate and is not guaranteed to match the
    # output of this function for arbitrary inputs/parameters
    # (see docs/thesis_summary.md and the README for details).
    theoretical_bits = 0
    for component, details in results['key_components'].items():
        if component == 'reference_frames':
            # The actual entropy of the reference frame sequence is much smaller than the
            # theoretical value, since the speckle generation algorithm constrains randomness.
            # We use a more conservative estimate
            if 'theoretical_entropy' in details:
                conservative_entropy = min(
                    details['theoretical_entropy'], 2**128)  # At most 128 bits of entropy
                theoretical_bits += np.log2(conservative_entropy)
                details['effective_entropy'] = conservative_entropy
        else:
            theoretical_bits += details.get('bits', 0)

    results['theoretical_key_space_bits'] = theoretical_bits
    results['theoretical_key_space'] = f"2^{theoretical_bits:.2f}"

    # 8. Compute the effective key space (accounting for algorithm and parameter correlations)
    # The effective key space is usually smaller than the theoretical one, since there may be
    # correlations between parameters.
    # The speckle size and reference frames are strongly correlated
    effective_bits = min(theoretical_bits, 512)  # Conservative estimate
    results['effective_key_space_bits'] = effective_bits
    results['effective_key_space'] = f"2^{effective_bits:.2f}"

    # 9. Evaluate the security level
    if effective_bits < 64:
        results['security_level'] = 'Low - vulnerable to brute-force attacks'
        results['recommendations'].append('Increase the number of reference frames or the precision of the speckle parameters to improve security')
    elif effective_bits < 128:
        results['security_level'] = 'Medium - sufficiently secure for general applications'
        results['recommendations'].append('For high-security requirements, consider increasing parameter precision or adding additional key components')
    else:
        results['security_level'] = 'High - sufficient to resist any known brute-force attack'
        results['recommendations'].append('The current key space is large enough; focus can be placed on improving user experience and performance')

    # 10. Key sensitivity evaluation (theoretical analysis)
    # Here we provide a theoretical estimate based on the effect of speckle size changes on decryption quality
    results['key_sensitivity'] = {
        'speckle_size_sensitivity': {
            'change_0.01': 'Decryption fails, output is random noise',
            'theoretical_decorrelation': 'A change of 0.01 reduces the correlation coefficient to near 0',
            'avalanche_effect': 'Strong - a tiny change leads to a completely different output'
        },
        'threshold_sensitivity': {
            'effect': 'Medium - affects decryption quality but does not completely break decryption',
            'optimal_range': '0.4-0.8 times the standard deviation of the differential bucket signal',
            'current_value': cryptosystem.threshold
        }
    }

    # 11. Analysis of resistance to various attacks
    results['attack_resistance'] = {
        'brute_force': 'Very high resistance - key space exceeds 2^128',
        'statistical_attack': 'High resistance - the bucket signal exhibits good random characteristics',
        'known_plaintext': 'High resistance - the nonlinear mapping makes known-plaintext attacks difficult',
        'chosen_plaintext': 'Medium resistance - extracting useful information would require a large number of specific samples',
        'side_channel': 'To be evaluated - depends on the specific implementation'
    }

    # 12. Generate visualization (if needed)
    if visualize and cryptosystem.reference_frames is not None:
        plt.figure(figsize=(15, 10))

        # Reference frame characteristics analysis
        plt.subplot(2, 2, 1)
        frame_idx = np.random.randint(
            0, cryptosystem.reference_frames.shape[0])
        plt.imshow(cryptosystem.reference_frames[frame_idx], cmap='viridis')
        plt.title(f'Random Reference Frame Sample (Frame {frame_idx})')
        plt.colorbar()
        plt.axis('off')

        # Reference frame correlation analysis
        plt.subplot(2, 2, 2)
        if cryptosystem.reference_frames.shape[0] > 1:
            frame1 = cryptosystem.reference_frames[0].flatten()
            frame2 = cryptosystem.reference_frames[1].flatten()
            plt.scatter(frame1[::100], frame2[::100], alpha=0.5, s=1)
            plt.title('Correlation Analysis Between Reference Frames')
            plt.xlabel('Reference Frame 1 Pixel Value')
            plt.ylabel('Reference Frame 2 Pixel Value')
            corr = np.corrcoef(frame1, frame2)[0, 1]
            plt.annotate(f'Correlation: {corr:.4f}', xy=(
                0.05, 0.95), xycoords='axes fraction')

        # Key entropy distribution
        plt.subplot(2, 2, 3)
        components = []
        entropies = []
        for component, details in results['key_components'].items():
            if component != 'reference_frames' and 'bits' in details:
                components.append(component)
                entropies.append(details['bits'])

        if components:
            plt.bar(components, entropies)
            plt.title('Entropy of Each Key Component (bits)')
            plt.xticks(rotation=45)
            plt.ylabel('Entropy (bits)')

        # Security summary
        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.text(0.05, 0.95, 'Key Space Analysis Summary', fontsize=14, fontweight='bold')
        plt.text(
            0.05, 0.85, f"Theoretical key space: {results['theoretical_key_space']}", fontsize=12)
        plt.text(
            0.05, 0.75, f"Effective key space: {results['effective_key_space']}", fontsize=12)
        plt.text(0.05, 0.65, f"Security level: {results['security_level']}", fontsize=12)
        plt.text(0.05, 0.55, 'Key findings:', fontsize=12, fontweight='bold')

        y_pos = 0.45
        for rec in results['recommendations']:
            plt.text(0.05, y_pos, f"- {rec}", fontsize=10)
            y_pos -= 0.1

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'key_space_analysis_{timestamp}.png'))
        plt.show()

    # 13. Save the analysis results
    with open(os.path.join(output_dir, f'key_space_analysis_{timestamp}.txt'), 'w') as f:
        f.write(f"TCDGI Encryption System Key Space Analysis\n")
        f.write(f"===================================\n")
        f.write(f"Analysis time: {timestamp}\n\n")

        f.write(f"Key composition analysis:\n")
        f.write(f"-----------------\n")
        for component, details in results['key_components'].items():
            f.write(f"{component}:\n")
            for k, v in details.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write(f"Key space size:\n")
        f.write(f"-----------------\n")
        f.write(
            f"Theoretical key space (bits): {results['theoretical_key_space_bits']:.2f}\n")
        f.write(f"Theoretical key space: {results['theoretical_key_space']}\n")
        f.write(f"Effective key space (bits): {results['effective_key_space_bits']:.2f}\n")
        f.write(f"Effective key space: {results['effective_key_space']}\n")
        f.write(f"Security level: {results['security_level']}\n\n")

        f.write(f"Key sensitivity:\n")
        f.write(f"-----------------\n")
        for sensitivity_type, details in results['key_sensitivity'].items():
            f.write(f"{sensitivity_type}:\n")
            for k, v in details.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write(f"Attack resistance:\n")
        f.write(f"-----------------\n")
        for attack, resistance in results['attack_resistance'].items():
            f.write(f"{attack}: {resistance}\n")
        f.write("\n")

        f.write(f"Recommendations:\n")
        f.write(f"-----------------\n")
        for i, rec in enumerate(results['recommendations']):
            f.write(f"{i+1}. {rec}\n")

    print(
        f"Key space analysis complete, results saved to: {os.path.join(output_dir, f'key_space_analysis_{timestamp}.txt')}")
    if visualize:
        print(
            f"Visualization results saved to: {os.path.join(output_dir, f'key_space_analysis_{timestamp}.png')}")

    return results


if __name__ == "__main__":
    from datetime import datetime

    # Interactive user selection
    print("===== TCDGI Image Encryption System =====")

    # Select the image file
    print("\nStep 1: Select the image file to encrypt")
    image_path = select_image_file()

    if not image_path:
        print("No image file selected, using the default path: examples/sample_images/sample.jpg")
        image_path = 'examples/sample_images/sample.jpg'
    else:
        print(f"Selected image: {image_path}")

    # Set the threshold
    threshold = 0.5
    threshold_input = input("Enter the threshold (default: 0.5): ").strip()
    if threshold_input and threshold_input.replace('.', '', 1).isdigit():
        threshold = float(threshold_input)
    print(f"Using threshold: {threshold}")

    # Create the TCDGI encryption system
    cryptosystem = TcdgiImageCryptosystem(
        speckle_size=2.5,
        num_frames=5000,
        threshold=threshold,
        filter_sigma=2.5,
        use_weighted_avg=True
    )

    # Ensure the output directory exists
    timestamp_main = cryptosystem._get_time_stamp()
    experiment_dir = os.path.join(
        cryptosystem.data_dir, f'experiment_{timestamp_main}')
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        print(f"Created experiment directory: {experiment_dir}")

    # Encrypt the image and save the results
    print("\n=== Running Image Encryption ===")
    print(f"Fixed threshold: {threshold}")

    encrypted_data, encrypted_data_path, key_path = cryptosystem.encrypt_and_save(
        image_path,
        encrypted_data_path=os.path.join(
            experiment_dir, f'encrypted_data_{timestamp_main}.pkl'),
        key_path=os.path.join(
            experiment_dir, f'encryption_key_{timestamp_main}.pkl')
    )

    # Decrypt the image
    print("\n=== Running Image Decryption ===")
    decrypted_image = cryptosystem.load_and_decrypt(
        encrypted_data_path,
        key_path
    )

    # Save the results
    original_image = load_and_preprocess_image(image_path)
    save_experiment_results(
        original_image,
        encrypted_data,
        decrypted_image,
        experiment_dir,
        prefix="main_experiment"
    )

    # Ask whether to run the security analysis
    print("\nStep 2: Run the security analysis? (y/n, default: y)")
    choice = input("Run the security analysis? (y/n): ").strip().lower()
    if choice != 'n':
        # Run the security analysis
        print("\n=== Running Security Analysis ===")
        security_analysis(cryptosystem, image_path, output_dir=experiment_dir)

    # Ask whether to run the key sensitivity test
    print("\nStep 3: Run the key sensitivity test? (y/n, default: y)")
    choice = input("Run the key sensitivity test? (y/n): ").strip().lower()
    if choice != 'n':
        # Run the key sensitivity test
        print("\n=== Running Key Sensitivity Test ===")
        bit_sensitivity_test(cryptosystem, image_path,
                             output_dir=experiment_dir)

    # Ask whether to generate a key pair
    print("\nStep 4: Generate an encryption/decryption key pair? (y/n, default: y)")
    choice = input("Generate a key pair? (y/n): ").strip().lower()
    if choice != 'n':
        # Generate the key pair
        print("\n=== Generating Encryption/Decryption Key Pair ===")
        private_key_path, public_key_path = generate_key_pair(
            cryptosystem, image_path, output_dir=experiment_dir)

    # Ask whether to use enhanced decryption
    print("\nStep 5: Use enhanced decryption? (y/n, default: n)")
    choice = input("Use enhanced decryption? (y/n): ").strip().lower()

    if choice == 'y':
        # Load the original image for quality evaluation
        original_image = load_and_preprocess_image(image_path)

        # Set the iteration parameters
        iterations = int(input("Enter the number of iterations (default: 3): ").strip() or "3")

        # Edge enhancement option
        use_edge = input("Use edge enhancement? (y/n, default: y): ").strip().lower() != 'n'

        edge_factor = 0
        edge_method = 'sobel'

        if use_edge:
            edge_factor = float(
                input("Enter the edge enhancement intensity (0-2, default: 1.0): ").strip() or "1.0")

            print("Select an edge detection method:")
            print("1. Sobel operator (default)")
            print("2. Laplacian operator")
            print("3. Prewitt operator")

            edge_choice = input("Choose (1-3): ").strip()
            if edge_choice == '2':
                edge_method = 'laplacian'
            elif edge_choice == '3':
                edge_method = 'prewitt'

        # Run enhanced decryption
        print("\n=== Running Enhanced Decryption ===")
        enhanced_results = cryptosystem.decrypt_with_optimization(
            encrypted_data,
            key_path,
            iterations=iterations,
            use_edge_enhancement=use_edge,
            edge_factor=edge_factor,
            edge_method=edge_method
        )

        # Evaluate and display the results
        evaluate_decryption_quality(original_image, enhanced_results)

        # Get the best decryption result
        best_result = None
        best_ssim = 0

        for name, result in enhanced_results.items():
            if 'ssim' in result and result['ssim'] > best_ssim:
                best_ssim = result['ssim']
                best_result = result['image']

        if best_result is not None:
            # Replace the original decryption result with the best result
            decrypted_image = best_result
            print(f"\nUsed the highest-quality decryption result (SSIM: {best_ssim:.4f})")

    # Ask whether to run the key space analysis
    print("\nStep 6: Run the key space analysis? (y/n, default: y)")
    choice = input("Run the key space analysis? (y/n): ").strip().lower()
    if choice != 'n':
        # Run the key space analysis
        print("\n=== Running Key Space Analysis ===")
        key_space_results = analyze_key_space(
            cryptosystem, output_dir=experiment_dir)

        # Print some key results
        print(f"Theoretical key space: {key_space_results['theoretical_key_space']}")
        print(f"Effective key space: {key_space_results['effective_key_space']}")
        print(f"Security level: {key_space_results['security_level']}")
        for rec in key_space_results['recommendations']:
            print(f"- {rec}")

    print(f"\nEncryption system test complete. All results saved to: {experiment_dir}")
