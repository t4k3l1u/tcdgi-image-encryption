# Design of Image Encryption System Based On Correspondence Ghost Imaging
BSc graduation project

## Overview

This project designs and evaluates an image encryption system based on
**Time-Correspondence Differential Ghost Imaging (TCDGI)**, an extension of
ghost imaging (GI) and differential ghost imaging (DGI) that combines
adaptive threshold selection, weighted averaging, and Gaussian filtering
to improve reconstruction quality and encryption performance.

The speckle pattern sequence and threshold parameters act as the encryption
key; an attacker without the correct key/threshold cannot reconstruct the
original image from the bucket-detector signal.

## Method summary

1. Generate a sequence of random reference speckle frames and corresponding bucket-detector signals (simulated ghost imaging acquisition).
2. Apply differential processing between bucket signals and integrated reference signals, then select frames via an **adaptive threshold** (computed from the signal distribution rather than fixed a priori).
3. Reconstruct the image using **weighted averaging** over selected frames, with Gaussian filtering for denoising.
4. Evaluate reconstruction quality (decryption side) and security properties (key space, key sensitivity / avalanche effect, statistical attack resistance) against GI and DGI baselines.

## Results

Compared with standard GI and DGI on the same test setup:

| Metric    | Improvement over GI/DGI baseline |
| --------- | -------------------------------- |
| SNR       | **+31.19%**                      |
| SSIM      | **+36.24%**                      |
| Key space | **2^71.29** (thesis-reported effective estimate, see `docs/thesis_summary.md`) |

TCDGI consistently ranked best across SSIM / SNR / PSNR in repeated comparison runs (see `results/` for sample comparison reports and figures). Sensitivity analysis shows the system is highly responsive to small perturbations in key parameters (threshold, speckle size), which underlies its resistance to brute-force and statistical attacks — full derivations and the complete metric set (PSNR, RMSE, correlation coefficient, avalanche effect, etc.) are documented in the accompanying dissertation (`docs/thesis_summary.md`, English summary of the original Chinese thesis).

## Repository structure

src/

TCDGI.py # Core TCDGI algorithm + adaptive threshold selection

cryptosystem.py # TcdgiImageCryptosystem: encrypt/decrypt pipeline

compare.py # GI / DGI / TCDGI comparison harness

metrics.py # Decryption quality metrics (SSIM, PSNR, entropy, edge preservation, ...)

analysis.py # Encrypted-data analysis and visualization

run_pipeline.py # End-to-end analysis entry point

examples/

sample_images/ # Small set of test images

results/

sample_comparison/ # Selected comparison figures + metric reports

docs/

thesis_summary.md # English summary of method and findings


## Running it

Quick demo (default parameters, ~1000 frames):
```bash
pip install -r requirements.txt
python src/run_pipeline.py --input examples/sample_images/sample.jpg
```

Thesis-style run (parameters matching the headline result in `docs/thesis_summary.md`):
```bash
python src/run_pipeline.py --input examples/sample_images/sample.jpg \
    --frames 5000 --speckle 2.5 --threshold 9.6429 --filter-sigma 2.5
```

Add `--seed <int>` to either command to make the main speckle sequence (and hence the
encryption key) reproducible. This covers the encrypt/decrypt/quality-evaluation steps
only; the key-sensitivity test in the pipeline derives a second key from `seed + 1`, and
the multi-threshold comparison reuses the main key, so the full report is reproducible
end-to-end given the same `--seed`, but `--seed` does not pin down a single global RNG
state for every sub-step independently.

Both commands run the full encrypt -> decrypt -> quality evaluation -> security
analysis pipeline and write results (figures, metric reports) to a
timestamped output directory.

## Notes

This is graduation-project research code, cleaned up and reorganised for publication. Earlier iteration branches have been removed in favour of the version that produced the reported results.
Test images and result samples here are a representative subset; the full experimental sweep (multiple thresholds, speckle sizes, and test images) is summarised in docs/thesis_summary.md.

The files in `results/sample_comparison/` are a demo export and are not intended to exactly
reproduce the thesis headline metrics in the table above; differences come from the input
image, preprocessing, random speckle generation (no fixed seed was used for that run), and
metric normalization. See `results/sample_comparison/summary_report.txt` for the exact
parameters used to produce that sample.

Key-space figures reported by the code itself are estimates, not a single fixed quantity:
`compare.py`'s `analyze_encryption_security()` computes a loose upper bound from the raw
reference-frame bit count, while `cryptosystem.py`'s `analyze_key_space()` computes a more
conservative, entropy-capped estimate. The **2^71.29** figure in the Results table above is
the thesis-reported effective estimate (see `docs/thesis_summary.md`); running either function
on a different image/parameter set will generally produce a different number.

## License
MIT
