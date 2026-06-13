import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from scipy import signal, stats
from PIL import Image



def analyze_encrypted_data(encrypted_data_path, output_dir=None):
    """
    Analyze the statistical properties of an encrypted data file.

    Parameters
    -----------
    encrypted_data_path : str
        Path to the encrypted data file (.pkl)
    output_dir : str
        Output directory for the results; defaults to the same directory as the input file
    """
    # Load the encrypted data
    with open(encrypted_data_path, 'rb') as f:
        encrypted_data = pickle.load(f)

    bucket_signals = encrypted_data['bucket_signals']

    # Set the output directory
    if output_dir is None:
        output_dir = os.path.dirname(encrypted_data_path)

    # Extract the filename as an identifier
    file_id = os.path.basename(encrypted_data_path).split('.')[0]

    # Create the figure
    plt.figure(figsize=(18, 12))

    # 1. Signal waveform
    plt.subplot(2, 3, 1)
    plt.plot(bucket_signals)
    plt.title('Bucket Signal Waveform')
    plt.xlabel('Frame Index')
    plt.ylabel('Signal Intensity')

    # 2. Signal histogram
    plt.subplot(2, 3, 2)
    sns.histplot(bucket_signals, kde=True)
    plt.title('Bucket Signal Distribution')
    plt.xlabel('Signal Value')
    plt.ylabel('Frequency')

    # 3. Autocorrelation analysis
    plt.subplot(2, 3, 3)
    autocorr = signal.correlate(bucket_signals - np.mean(bucket_signals),
                                bucket_signals - np.mean(bucket_signals), mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr /= autocorr[0]  # Normalize
    plt.plot(autocorr[:100])  # Show only the first 100 points
    plt.title('Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('Correlation Coefficient')

    # 4. Power spectral density
    plt.subplot(2, 3, 4)
    f, psd = signal.welch(bucket_signals, fs=1.0, nperseg=256)
    plt.semilogy(f, psd)
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency')
    plt.ylabel('Power/Frequency (dB/Hz)')

    # 5. Cumulative distribution function
    plt.subplot(2, 3, 5)
    counts, bin_edges = np.histogram(bucket_signals, bins=50, density=True)
    cdf = np.cumsum(counts) * (bin_edges[1] - bin_edges[0])
    plt.plot(bin_edges[1:], cdf)
    plt.title('Cumulative Distribution Function')
    plt.xlabel('Signal Value')
    plt.ylabel('Probability')

    # 6. Signal randomness test results
    plt.subplot(2, 3, 6)
    plt.axis('off')

    # Randomness statistics
    stats_info = [
        f"Mean: {np.mean(bucket_signals):.4f}",
        f"Std Dev: {np.std(bucket_signals):.4f}",
        f"Median: {np.median(bucket_signals):.4f}",
        f"Skewness: {stats.skew(bucket_signals):.4f}",
        f"Kurtosis: {stats.kurtosis(bucket_signals):.4f}",
        f"Min: {np.min(bucket_signals):.4f}",
        f"Max: {np.max(bucket_signals):.4f}",
        f"Signal Length: {len(bucket_signals)}",
        f"Signal Entropy: {stats.entropy(np.histogram(bucket_signals, bins=50)[0]):.4f}"
    ]

    # Perform a runs test to evaluate randomness
    median = np.median(bucket_signals)
    binary_seq = (bucket_signals > median).astype(int)
    runs = np.diff(binary_seq) != 0
    num_runs = np.sum(runs) + 1
    n1 = np.sum(binary_seq)
    n2 = len(binary_seq) - n1

    # Compute the expected number of runs and the standard deviation
    r_exp = (2 * n1 * n2) / (n1 + n2) + 1
    s_r = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
                  ((n1 + n2)**2 * (n1 + n2 - 1)))

    # Compute the Z statistic
    z = (num_runs - r_exp) / s_r
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    stats_info.append(f"Runs Test Z-value: {z:.4f}")
    stats_info.append(f"Runs Test p-value: {p_value:.4f}")
    stats_info.append(f"Randomness: {'Good' if p_value > 0.05 else 'Possibly insufficient'}")

    # Display the statistics on the figure
    plt.text(0.1, 0.9, '\n'.join(stats_info), fontsize=10,
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.title('Signal Statistical Properties')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_id}_analysis.png'), dpi=300)
    plt.show()

    # Save the analysis results to a text file
    with open(os.path.join(output_dir, f'{file_id}_stats.txt'), 'w') as f:
        f.write("Encrypted Data Statistical Analysis\n")
        f.write("="*40 + "\n")
        f.write(f"File: {encrypted_data_path}\n")
        f.write("="*40 + "\n\n")
        f.write('\n'.join(stats_info))

    print(f"Analysis complete, results saved to {output_dir}")

    return stats_info
