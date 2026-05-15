
## qNMR External Calibration App — Complete User Guide

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

`app_pulcon.py` :contentReference[oaicite:0]{index=0}

---

## Main Features

#### Data Input

Supports:

- CSV spectra
- Bruker processed spectra
- raw Bruker FID processing

---

#### Integration Methods

###### Trapezoidal integration

Experimental direct numerical integration.

---

###### Single pseudo-Voigt fitting

One fitted pseudo-Voigt model.

---

###### Multiplet pseudo-Voigt fitting

Constrained multi-component fitting.

---

#### Baseline Correction

Available modes:

- None
- Linear baseline
- Local minimum subtraction

---

#### Mathematical Diagnostics

The app exports:

- Fit/Sum ratio
- Residual area
- R²
- RMSE
- SNR
- Numerical vs analytical area ratio
- FWHM
- dx resolution diagnostics

---

#### Batch Quantification

Supports:

- one reference
- multiple samples
- multiple integration regions
- paired or combinatorial region comparison

---

## Application Architecture

The app is divided into seven tabs:

| Tab | Purpose |
|---|---|
| Bruker Processing | Import and process Bruker experiments |
| Data | Inspect spectra |
| Reference | Define external reference |
| Sample | Define analyte/sample parameters |
| Integration | Configure integration/fitting |
| Quantification | Run calculations |
| Report | Export results |

---

## 1. Bruker Processing Tab

This tab handles Bruker data import and processing.

---

## Supported Workflows

#### Single Experiment Mode

Used for:

- one reference spectrum
- one sample spectrum

Workflow:

1. Upload ZIP
2. Process spectrum
3. Save as Reference or Sample

---

#### Batch Bruker Mode

Used for:

- many experiments
- one reference
- multiple samples

Workflow:

1. Upload ZIP containing multiple experiments
2. Detect experiments
3. Assign:
   - Reference
   - Sample
   - Ignore
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
````

or:

```text
experiment/
    pdata/1/1r
```

---

## Processing Modes

---

#### Read Processed Spectrum

Reads:

```text
pdata/1/1r
```

Recommended when spectra were already processed in TopSpin.

---

#### Process Raw FID

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

## Important Recommendation

For validated qNMR:

#### Prefer

```text
Processed TopSpin spectra
```

because manual phasing and baseline correction are usually more reliable.

---

## 2. Data Tab

This tab displays:

* reference spectrum
* sample spectrum

with:

* ppm axis
* intensity
* interactive zoom
* reversed ppm axis

---

## Purpose

Use this tab to:

* verify successful import
* inspect spectral quality
* check noise
* inspect phasing
* inspect baseline

before quantification.

---

## 3. Reference Tab

Defines the external reference.

---

## Two Input Modes

---

#### A. Mass-Based Mode

The app calculates concentration automatically.

Required inputs:

| Parameter       | Meaning          |
| --------------- | ---------------- |
| Reference mass  | mass of standard |
| Solvent mass    | solvent weight   |
| Solvent density | density          |
| MW              | molecular weight |

---

## Internal Formula

```text
C_ref (mM)
=
(reference_mass × density)
/
(MW × solvent_mass)
```

---

#### B. Direct mM Mode

The user directly provides:

```text
C_ref
```

---

## Additional Parameters

| Parameter | Meaning           |
| --------- | ----------------- |
| nH        | number of protons |
| NS        | number of scans   |
| P1        | pulse length      |
| RG        | receiver gain     |
| Volume    | NMR tube volume   |
| DF        | dilution factor   |

---

## Reference Regions

Supports:

* one region
* multiple regions

Format:

```text
7.19, 7.39
3.85, 3.95
```

---

## Important Recommendation

Choose:

* isolated
* stable
* non-overlapping
* symmetric

signals whenever possible.

---

## 4. Sample Tab

Defines:

* analyte signal
* sample acquisition parameters

---

## Required Parameters

| Parameter | Meaning              |
| --------- | -------------------- |
| nH        | analyte proton count |
| NS        | number of scans      |
| P1        | pulse length         |
| RG        | receiver gain        |
| Volume    | NMR volume           |
| DF        | dilution factor      |

---

## Optional

```text
Analyte MW
```

If provided, the app calculates:

```text
mg/mL
```

in addition to:

```text
mM
```

---

## Sample Regions

Supports:

* one region
* multiple regions

---

## Multiple Region Logic

The app can:

* pair regions by index
* reuse one reference region
* generate all combinations

depending on region counts.

---

## 5. Integration Tab

This is the core quantitative section.

---

## Available Integration Methods

---

## A. Trapezoidal

Experimental direct integration.

---

#### Advantages

* closest to classical qNMR
* no peak-shape assumptions
* robust
* validation-friendly

---

#### Recommended Use

Primary quantitative method.

---

## B. Single pseudo-Voigt Fit

Fits:

```text
1 pseudo-Voigt peak
```

---

#### Advantages

* line-shape analysis
* overlap diagnostics
* model comparison

---

#### Limitations

* may lose tails
* may underfit overlap
* assumes idealized peak shape

---

## C. Multiplet pseudo-Voigt Fit

Fits:

```text
multiple constrained peaks
```

---

#### Recommended for

* doublets
* triplets
* quartets
* structured multiplets

---

## Baseline Modes

---

## None

No baseline correction.

Best for:

* flat baselines
* validation
* direct experimental integration

---

## Linear Baseline

Interpolates baseline between integration limits.

Best for:

* routine qNMR
* slight slopes

Recommended default.

---

## Local Minimum

Subtracts minimum intensity inside region.

Best for:

* exploratory metabolomics
* offset correction

Use carefully.

---

## Important Recommendation

For validated qNMR:

#### Prefer

```text
None
```

or:

```text
Linear baseline
```

---

## Integration Diagnostics

The app exports:

| Metric        | Meaning                     |
| ------------- | --------------------------- |
| Fit/Sum ratio | fitted vs experimental area |
| Residual area | unexplained signal          |
| R²            | fit agreement               |
| RMSE          | fitting error               |
| SNR           | local signal-to-noise       |
| FWHM          | linewidth                   |
| dx_ppm        | digital resolution          |

---

## Fit/Sum Ratio Interpretation

| Ratio     | Interpretation           |
| --------- | ------------------------ |
| 0.95–1.05 | very good                |
| 0.80–0.95 | acceptable               |
| <0.80     | model losing area        |
| >1.10     | artificial area creation |

---

## Important Scientific Interpretation

If:

```text
fit_numeric_analytic_ratio ≈ 1
```

but:

```text
fit_sum_ratio << 1
```

then:

* mathematics are correct
* the model is insufficient

Usually caused by:

* overlap
* asymmetry
* tails
* baseline distortion

---

## 6. Quantification Tab

Runs external calibration calculations.

---

## Equation Used

The app uses a practical external calibration equation inspired by PULCON workflows.

The calculation includes:

* signal area
* proton count
* scans
* pulse length
* receiver gain
* volume
* dilution

---

## Batch Quantification

Supports:

* one reference
* many samples
* many regions

---

## Result Types

---

## region_result

One individual quantification result.

---

## sample_summary

Summary statistics:

* mean concentration
* standard deviation
* RSD

---

## RSD Interpretation

| RSD   | Interpretation |
| ----- | -------------- |
| <5%   | excellent      |
| 5–10% | acceptable     |
| >10%  | investigate    |

---

## Possible Causes of High RSD

* overlap
* poor baseline
* poor phasing
* incorrect proton count
* relaxation mismatch
* low SNR

---

## 7. Report Tab

Exports:

* all quantitative results
* diagnostics
* integration metadata
* fitting metadata

in CSV format.

---

## Recommended Validation Workflow

---

## Step 1

Inspect spectra.

---

## Step 2

Start with:

```text
Trapezoidal
```

---

## Step 3

Use:

```text
pseudo-Voigt
```

as diagnostic support.

---

## Step 4

Inspect:

* Fit/Sum ratio
* residual area
* R²
* FWHM
* SNR

---

## Step 5

Investigate problematic fits manually.

---

## Best Practices

---

## For validated qNMR

#### Prefer:

* isolated signals
* linear baseline
* trapezoidal integration
* experimental area

---

## Use fitting mainly for:

* diagnostics
* overlap assessment
* line-shape analysis

---

## Avoid

* aggressive baseline correction
* blind automated fitting
* overfitting multiplets
* narrow integration windows

---

## Current Limitations

The app intentionally avoids:

* fully automated deconvolution
* aggressive baseline removal
* black-box quantification

because the goal is:

```text
transparent validation-oriented qNMR
```

---

## Future Possible Features

Potential future developments:

* polynomial baseline
* ALS baseline
* rubber-band baseline
* automatic overlap detection
* Bayesian fitting
* probabilistic quantification
* Lorentzian/Gaussian adaptive models
* uncertainty propagation
* automatic QC scoring

---

## Final Recommendation

For quantitative reporting:

#### Use

```text
Trapezoidal integration
```

as the primary quantitative result.

Use pseudo-Voigt fitting mainly as:

* diagnostic information
* overlap evaluation
* model validation

rather than direct replacement of experimental area.

```
```


## qNMR External Calibration — Mathematical Validation Tutorial

#### Introduction

This document describes the mathematical and numerical validation strategy used in the qNMR External Calibration application implemented in `app_pulcon.py`. :contentReference[oaicite:0]{index=0}

The goal is to verify whether differences between:

- experimental integration (`Trapezoidal/Sum`)
- pseudo-Voigt fitted integration
- external calibration concentration results

originate from:

1. Mathematical implementation errors
2. Numerical discretization issues
3. Peak-model limitations
4. Baseline artifacts
5. Spectral overlap
6. Inappropriate peak-shape assumptions

---

## Core Mathematical Validation Checklist

The following items should be systematically verified.

---

## 1. ppm vs Hz Conversion

#### Why this matters

NMR spectra are digitized in ppm, but:

- scalar couplings are defined in Hz
- linewidths are often discussed in Hz
- fitting models may internally mix ppm and Hz

A wrong conversion factor propagates directly into:

- peak spacing
- linewidth
- fitted area
- pseudo-Voigt geometry

---

#### Current implementation

```python
spacing_ppm = spacing_hz / spectrometer_mhz
