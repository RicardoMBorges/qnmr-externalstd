````md
# qNMR External Calibration App — Complete User Guide

## Overview

The qNMR External Calibration App is a Streamlit-based platform for:

- quantitative NMR (qNMR)
- external calibration
- PULCON-style workflows
- Bruker data processing
- spectral integration
- pseudo-Voigt fitting
- batch quantification
- validation-oriented diagnostics

The application supports:

- CSV spectra
- Bruker processed spectra
- raw Bruker FIDs
- single-spectrum workflows
- batch processing workflows

The app was designed to prioritize:

- transparency
- traceability
- validation
- diagnostic interpretation
- user control

rather than fully automated “black-box” quantification.

The source code is implemented in:

`app_pulcon.py`

---

# Installation

## Clone repository

```bash
git clone <repository_url>
cd <repository_name>
````

## Create environment

```bash
conda create -n qnmr python=3.11
conda activate qnmr
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run app

```bash
streamlit run app_pulcon.py
```

---

# Main Dependencies

* streamlit
* numpy
* pandas
* scipy
* plotly
* nmrglue

---

# Main Features

## Data Input

Supports:

* CSV spectra
* Bruker processed spectra
* raw Bruker FID processing

---

## Integration Methods

### Trapezoidal Integration

Experimental direct numerical integration.

### Single pseudo-Voigt Fitting

One fitted pseudo-Voigt model.

### Multiplet pseudo-Voigt Fitting

Constrained multi-component fitting.

---

## Baseline Correction

Available modes:

* None
* Linear baseline
* Local minimum subtraction

---

## Mathematical Diagnostics

The app exports:

* Fit/Sum ratio
* Residual area
* R²
* RMSE
* SNR
* Numerical vs analytical area ratio
* FWHM
* dx resolution diagnostics

---

## Batch Quantification

Supports:

* one reference
* multiple samples
* multiple integration regions
* paired or combinatorial region comparison

---

# Application Architecture

The app is divided into seven tabs:

| Tab               | Purpose                               |
| ----------------- | ------------------------------------- |
| Bruker Processing | Import and process Bruker experiments |
| Data              | Inspect spectra                       |
| Reference         | Define external reference             |
| Sample            | Define analyte/sample parameters      |
| Integration       | Configure integration/fitting         |
| Quantification    | Run calculations                      |
| Report            | Export results                        |

---

# 1. Bruker Processing Tab

This tab handles Bruker data import and processing.

---

## Supported Workflows

### Single Experiment Mode

Used for:

* one reference spectrum
* one sample spectrum

Workflow:

1. Upload ZIP
2. Process spectrum
3. Save as Reference or Sample

---

### Batch Bruker Mode

Used for:

* many experiments
* one reference
* multiple samples

Workflow:

1. Upload ZIP containing multiple experiments
2. Detect experiments
3. Assign:

   * Reference
   * Sample
   * Ignore
4. Process spectra
5. Run batch quantification

---

## Supported Bruker Structures

The app detects:

```text
experiment/
    fid
    acqus
    pdata/1/1r
```

or:

```text
experiment/
    pdata/1/1r
```

---

## Processing Modes

### Read Processed Spectrum

Reads:

```text
pdata/1/1r
```

Recommended when spectra were already processed in TopSpin.

---

### Process Raw FID

Applies:

* digital filter removal
* exponential multiplication
* zero filling
* Fourier transform
* optional automatic phase correction

Recommended for:

* raw FID validation
* experimental processing comparison

---

## Processing Parameters

| Parameter  | Meaning                    |
| ---------- | -------------------------- |
| LB         | line broadening            |
| ZF factor  | zero filling               |
| Auto phase | automatic phase correction |

---

# Appendix — Mathematical Validation

## Introduction

This document describes the mathematical and numerical validation strategy used in the qNMR External Calibration application implemented in `app_pulcon.py`.

The goal is to verify whether differences between:

* experimental integration (`Trapezoidal/Sum`)
* pseudo-Voigt fitted integration
* external calibration concentration results

originate from:

1. Mathematical implementation errors
2. Numerical discretization issues
3. Peak-model limitations
4. Baseline artifacts
5. Spectral overlap
6. Inappropriate peak-shape assumptions

---

# Core Mathematical Validation Checklist

## 1. ppm vs Hz Conversion

### Why this matters

NMR spectra are digitized in ppm, but:

* scalar couplings are defined in Hz
* linewidths are often discussed in Hz
* fitting models may internally mix ppm and Hz

A wrong conversion factor propagates directly into:

* peak spacing
* linewidth
* fitted area
* pseudo-Voigt geometry

---

### Current implementation

```python
spacing_ppm = spacing_hz / spectrometer_mhz
```

and:

```python
dx_hz = dx_ppm * spectrometer_mhz
```

---

## 2. dx Used in Numerical Integration

### Why this matters

Numerical integration depends on:

```text
Area ≈ Σ y_i * dx
```

If dx is inconsistent or too coarse:

* area errors occur
* fitted area differs from analytical area
* pseudo-Voigt tails become undersampled

---

### Current implementation

```python
np.trapezoid(y, x)
```

or fallback:

```python
np.trapz(y, x)
```

---

## 3. Definition of Pseudo-Voigt Width

### Current implementation

```python
pseudo_voigt(
    x,
    amplitude,
    center,
    sigma,
    gamma,
    eta
)
```

Where:

| Parameter | Meaning         |
| --------- | --------------- |
| sigma     | Gaussian σ      |
| gamma     | Lorentzian HWHM |
| eta       | mixing factor   |

---

## 4. Amplitude vs Height

The pseudo-Voigt is defined as:

```python
return amplitude * (
    eta * lorentzian
    + (1 - eta) * gaussian
)
```

Therefore:

```text
amplitude = peak height
```

NOT total area.

---

## 5. FWHM vs Sigma

Gaussian FWHM:

```python
fwhm_gaussian = 2 * sqrt(2 * ln(2)) * sigma
```

Lorentzian FWHM:

```python
fwhm_lorentzian = 2 * gamma
```

---

## 6. Analytical Area of the Pseudo-Voigt

Gaussian area:

```python
A_gaussian = amplitude * sigma * sqrt(2π)
```

Lorentzian area:

```python
A_lorentzian = amplitude * π * gamma
```

Mixed pseudo-Voigt area:

```python
A = eta * A_lorentzian + (1 - eta) * A_gaussian
```

---

## 7. Internal Normalization

The app does NOT use internally area-normalized functions.

Instead:

```text
max(gaussian) = 1
max(lorentzian) = 1
```

Therefore:

```text
amplitude = physical peak height
```

---

# Recommended Validation Workflow

## Step 1

Start with:

```text
Trapezoidal
```

---

## Step 2

Run:

```text
Single pseudo-Voigt fit
```

Inspect:

* Fit/Sum ratio
* residual area
* R²
* RMSE
* FWHM
* SNR

---

## Step 3

Interpret Fit/Sum ratio:

| Ratio     | Interpretation           |
| --------- | ------------------------ |
| 0.95–1.05 | very good                |
| 0.80–0.95 | acceptable               |
| <0.80     | model losing area        |
| >1.10     | artificial area creation |

---

# Final Recommendation

For quantitative reporting:

```text
Use Trapezoidal integration as the primary quantitative result.
```

Use pseudo-Voigt fitting mainly for:

* diagnostics
* overlap evaluation
* model validation
* line-shape assessment

rather than direct replacement of experimental area.

```
```
