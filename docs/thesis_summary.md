# Thesis Summary

> BSc thesis (Electronic Information Engineering). Method and results condensed for this repository. All figures are taken from the final thesis.

## Problem

Optical image encryption is a promising route to securing image data, and ghost imaging (GI) is an attractive primitive because the secret key can be encoded in the random reference patterns. Plain GI, however, gives low reconstruction quality at a fixed frame budget. This work studies **Time-Correspondence Differential Ghost Imaging (TCDGI)** as the encryption primitive and builds a complete encrypt/decrypt system around it.

## Method

TCDGI reconstructs the object by combining temporal correspondence selection with differential processing:

1. **Signal processing**: compute the reference integral signal and the differential bucket signal, then its fluctuation around the mean.
2. **Adaptive threshold selection**: split reference frames into positive/negative subsets by the sign of the fluctuation, and keep only frames whose fluctuation exceeds a threshold `k`. `k` is set adaptively from the standard deviation of the differential bucket signal (the `0.5σ–0.7σ` band is near-optimal); batch threshold testing locates the best value per target.
3. **Image reconstruction**: average the selected positive/negative subsets and take their difference; apply **weighted averaging** and **Gaussian filtering** to suppress high-frequency noise while preserving edges, followed by centered normalization to `[-1, 1]`.
4. **Encryption system**: the random reference sequence, speckle size, threshold, and Gaussian parameters jointly form a composite key; only the exact key reconstructs the plaintext image.

## Key Results

Test setup: 256×256 binary target, 5000 frames, speckle size 2.5, weighted-average reconstruction (Intel i7-10700K / 16 GB).

**Reconstruction quality (TCDGI vs. GI/DGI, best threshold k = 9.6429):**

| Metric | GI / DGI | TCDGI | Improvement |
|--------|----------|-------|-------------|
| SNR    | 0.9590   | 1.2581 | **+31.19%** |
| SSIM   | 0.4680   | 0.6376 | **+36.24%** |

TCDGI reaches its best reconstruction using only about half the frames, and remains stable across the `k ≈ 9.64–10.71` band (SNR > 1.24, SSIM > 0.63), indicating useful fault tolerance in parameter selection. TCDGI also retains a quality advantage under added noise (e.g. SSIM ≈ 0.4830 at high noise, ~59% above GI/DGI).

**Security:**

- **Key space**: theoretical and effective key space both ≈ **2^71.29** (threshold ~20 bits, reference sequence ~10 bits, Gaussian ~7 bits). Larger than DES (2^56), below AES-128 (2^128); sufficient against brute force under conventional compute. This is the thesis-reported effective estimate for the reported configuration — the key-space analysis functions in `src/` use different estimation methodologies and will generally produce different numbers on other inputs/parameters (see README).
- **Key sensitivity (avalanche)**: a 0.01 change in speckle size (2.5 → 2.51) collapses the decryption correlation toward zero; a 1% reference-frame occlusion drops SNR 1.27 → 1.21, and a 3% occlusion drops SNR to 0.62 (error rate > 75%). TCDGI shows the highest avalanche value (0.0355) among the three methods.
- **Attack resistance**: known-plaintext: high; chosen-plaintext and differential analysis: medium-high (the nonlinear threshold-selection step blurs plaintext features in the bucket signal, beating GI/DGI's medium level).

## Takeaway

Adding adaptive threshold selection plus weighted-average and Gaussian post-processing to differential ghost imaging measurably improves reconstruction quality (SNR/SSIM) and computational efficiency without weakening the cryptographic properties, while the threshold step adds nonlinearity that strengthens resistance to chosen-plaintext and differential attacks.
