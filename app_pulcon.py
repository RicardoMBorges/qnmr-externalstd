import os
import zipfile
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit

try:
    import nmrglue as ng
    NMRGLUE_AVAILABLE = True
except Exception:
    NMRGLUE_AVAILABLE = False


st.set_page_config(
    page_title="qNMR External Calibration",
    page_icon="🧪",
    layout="wide"
)


# =========================================================
# Session State
# =========================================================

if "processed_reference" not in st.session_state:
    st.session_state.processed_reference = None

if "processed_sample" not in st.session_state:
    st.session_state.processed_sample = None

if "quant_results" not in st.session_state:
    st.session_state.quant_results = None

if "batch_spectra" not in st.session_state:
    st.session_state.batch_spectra = {}

if "batch_reference_name" not in st.session_state:
    st.session_state.batch_reference_name = None

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

if "selected_batch_sample_labels" not in st.session_state:
    st.session_state.selected_batch_sample_labels = []

if "reference_table" not in st.session_state:
    st.session_state.reference_table = pd.DataFrame([{
        "Reference compound": "Caffeine",
        "Input mode": "Mass-based",
        "Reference mass (mg)": 0.7356,
        "Solvent mass (mg)": 994.0,
        "Solvent density (g/mL)": 1.478,
        "Concentration (mM)": 6.50,
        "Signal protons (nH)": 1.0,
        "NS": 16.0,
        "P1 (µs)": 10.0,
        "RG": 1.0,
        "Volume (µL)": 600.0,
        "Dilution factor": 1.0,
        "Molecular weight (g/mol)": 194.0804,
        "Region start (ppm)": 7.19,
        "Region end (ppm)": 7.39,
        "Notes": "Caffeine aromatic signal"
    }])

# Apply pending widget updates before widgets are instantiated.
# Streamlit does not allow changing st.session_state values for widget keys
# after the widgets have already been created in the same rerun.
if "pending_reference_update" in st.session_state:
    pending_update = st.session_state.pop("pending_reference_update")
    for key, value in pending_update.items():
        st.session_state[key] = value


# =========================================================
# Helper Functions
# =========================================================

def load_csv(file):
    """Load a CSV spectrum file with ppm and intensity columns."""
    return pd.read_csv(file)


def detect_ppm_intensity_columns(df):
    """Try to infer ppm and intensity columns from a dataframe."""
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}

    ppm_candidates = ["ppm", "chemical_shift", "chemical shift", "delta", "x"]
    intensity_candidates = ["intensity", "signal", "y", "real", "amplitude"]

    ppm_col = None
    intensity_col = None

    for c in ppm_candidates:
        if c in lower:
            ppm_col = lower[c]
            break

    for c in intensity_candidates:
        if c in lower:
            intensity_col = lower[c]
            break

    if ppm_col is None:
        ppm_col = cols[0]
    if intensity_col is None:
        intensity_col = cols[1]

    return ppm_col, intensity_col


def clean_spectrum_df(df, ppm_col, intensity_col):
    """Return a standardized spectrum dataframe."""
    out = df[[ppm_col, intensity_col]].copy()
    out.columns = ["ppm", "intensity"]
    out = out.dropna()
    out["ppm"] = pd.to_numeric(out["ppm"], errors="coerce")
    out["intensity"] = pd.to_numeric(out["intensity"], errors="coerce")
    out = out.dropna()
    return out.sort_values("ppm", ascending=True).reset_index(drop=True)


def integrate_region(x, y, xmin, xmax, absolute=True, baseline="none"):
    """Integrate a spectral region with boundary interpolation.

    baseline:
    - "none": raw trapezoidal area
    - "linear": subtract straight baseline between region limits
    - "minimum": subtract local minimum inside the region
    """

    low, high = sorted([xmin, xmax])

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    mask = (x_sorted >= low) & (x_sorted <= high)

    if mask.sum() < 2:
        return np.nan

    x_inside = x_sorted[mask]
    y_inside = y_sorted[mask]

    y_low = np.interp(low, x_sorted, y_sorted)
    y_high = np.interp(high, x_sorted, y_sorted)

    x_int = np.concatenate([[low], x_inside, [high]])
    y_int = np.concatenate([[y_low], y_inside, [y_high]])

    unique_idx = np.unique(x_int, return_index=True)[1]
    unique_idx = sorted(unique_idx)

    x_int = x_int[unique_idx]
    y_int = y_int[unique_idx]

    if baseline == "linear":
        baseline_line = np.interp(x_int, [low, high], [y_low, y_high])
        y_int = y_int - baseline_line

    elif baseline == "minimum":
        y_int = y_int - np.nanmin(y_int)

    area = float(
        np.trapezoid(y_int, x_int)
        if hasattr(np, "trapezoid")
        else np.trapz(y_int, x_int)
    )

    return abs(area) if absolute else area

def parse_regions(region_text):
    """Parse one or more spectral regions from text.

    Expected format, one region per line:
    7.19, 7.39
    3.20, 3.35

    Returns
    -------
    list[tuple[float, float]]
        List of (start_ppm, end_ppm) regions.
    """
    regions = []
    if region_text is None:
        return regions

    for line in str(region_text).splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            a, b = [float(x.strip()) for x in line.split(",")]
            regions.append((a, b))
        except Exception:
            continue

    return regions


def format_regions(regions):
    """Format a list of regions as multiline text."""
    if not regions:
        return ""
    return "\n".join([f"{a}, {b}" for a, b in regions])


def ensure_regions(value, default=(7.19, 7.39)):
    """Normalize a tuple or list of tuples to a list of regions."""
    if value is None:
        return [default]
    if isinstance(value, tuple) and len(value) == 2:
        return [value]
    if isinstance(value, list):
        if len(value) == 0:
            return [default]
        if isinstance(value[0], tuple):
            return value
    return [default]


def baseline_subtract_minimum(x, y, xmin, xmax):
    """Simple local baseline correction by subtracting the minimum intensity inside the selected region."""
    low, high = sorted([xmin, xmax])
    mask = (x >= low) & (x <= high)
    y_corr = y.copy()
    if mask.sum() > 0:
        y_corr = y_corr - np.nanmin(y_corr[mask])
    return y_corr

def sum_integrate_region(
    x,
    y,
    xmin,
    xmax,
    absolute=True,
    baseline="none",
    scale_mode="dx"
):
    """Sum-based integration.

    scale_mode:
    - "dx": returns sum(y) * median dx, comparable to ppm-scaled area.
    - "raw": returns sum(y), comparable to raw point-sum area.
    """

    low, high = sorted([xmin, xmax])

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (x >= low) & (x <= high)

    if mask.sum() < 2:
        return np.nan, {}

    x_region = x[mask].copy()
    y_region = y[mask].copy()

    order = np.argsort(x_region)
    x_region = x_region[order]
    y_region = y_region[order]

    if baseline == "minimum":
        y_region = y_region - np.nanmin(y_region)

    elif baseline == "linear":
        y_low = np.interp(low, x_region, y_region)
        y_high = np.interp(high, x_region, y_region)

        baseline_line = np.interp(
            x_region,
            [low, high],
            [y_low, y_high]
        )

        y_region = y_region - baseline_line

    dx = (
        float(np.nanmedian(np.abs(np.diff(x_region))))
        if len(x_region) > 2
        else np.nan
    )

    raw_sum = float(np.nansum(y_region))
    dx_scaled_sum = raw_sum * dx if np.isfinite(dx) else np.nan

    if scale_mode == "raw":
        selected_area = raw_sum
    else:
        selected_area = dx_scaled_sum

    if absolute:
        selected_area = abs(selected_area)
        raw_sum = abs(raw_sum)
        dx_scaled_sum = abs(dx_scaled_sum)

    diagnostics = {
        "sum_raw_points": raw_sum,
        "sum_dx_scaled": dx_scaled_sum,
        "sum_dx_ppm": dx,
        "sum_n_points": int(len(x_region)),
        "sum_scale_mode": scale_mode,
    }

    return float(selected_area), diagnostics
def pseudo_voigt(x, amplitude, center, sigma, gamma, eta):
    """Single pseudo-Voigt line shape."""
    sigma = max(abs(sigma), 1e-12)
    gamma = max(abs(gamma), 1e-12)
    eta = np.clip(eta, 0, 1)

    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    lorentzian = gamma**2 / ((x - center) ** 2 + gamma**2)
    return amplitude * (eta * lorentzian + (1 - eta) * gaussian)

def gaussian_peak(x, amplitude, center, sigma):
    """Single Gaussian peak. Amplitude is peak height."""
    sigma = max(abs(sigma), 1e-12)
    return amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))


def lorentzian_peak(x, amplitude, center, gamma):
    """Single Lorentzian peak. Amplitude is peak height and gamma is HWHM."""
    gamma = max(abs(gamma), 1e-12)
    return amplitude * (gamma**2 / ((x - center) ** 2 + gamma**2))


def gaussian_area_analytic(amplitude, sigma):
    return float(amplitude * abs(sigma) * np.sqrt(2 * np.pi))


def lorentzian_area_analytic(amplitude, gamma):
    return float(amplitude * np.pi * abs(gamma))

def multiplet_pseudo_voigt(x, center, sigma, gamma, eta, amplitude, spacing_hz, ratio_1, ratio_2, ratio_3, ratio_4, spectrometer_mhz, peak_count):
    """Pseudo-Voigt multiplet with up to four components.

    The spacing is defined in Hz and converted to ppm using spectrometer MHz.
    Ratios are relative peak intensities.
    """
    ratios = np.array([ratio_1, ratio_2, ratio_3, ratio_4], dtype=float)[:peak_count]
    ratios = np.maximum(ratios, 0)
    if ratios.sum() == 0:
        ratios = np.ones(peak_count)
    ratios = ratios / ratios.max()

    spacing_ppm = spacing_hz / spectrometer_mhz
    positions = np.arange(peak_count) - (peak_count - 1) / 2
    signal = np.zeros_like(x, dtype=float)

    for pos, ratio in zip(positions, ratios):
        signal += pseudo_voigt(x, amplitude * ratio, center + pos * spacing_ppm, sigma, gamma, eta)

    return signal

def pseudo_voigt_area_analytic(amplitude, sigma, gamma, eta):
    """Approximate analytical area of the pseudo-Voigt model in ppm units."""
    sigma = abs(float(sigma))
    gamma = abs(float(gamma))
    eta = float(np.clip(eta, 0, 1))

    gaussian_area = amplitude * sigma * np.sqrt(2 * np.pi)
    lorentzian_area = amplitude * np.pi * gamma

    return eta * lorentzian_area + (1 - eta) * gaussian_area


def pseudo_voigt_fwhm_estimate(sigma, gamma, eta):
    """Approximate pseudo-Voigt FWHM from Gaussian and Lorentzian components."""
    sigma = abs(float(sigma))
    gamma = abs(float(gamma))
    eta = float(np.clip(eta, 0, 1))

    fwhm_gaussian = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_lorentzian = 2 * gamma

    return eta * fwhm_lorentzian + (1 - eta) * fwhm_gaussian

def fit_single_peak(x, y):
    """Fit one pseudo-Voigt peak."""
    amplitude0 = float(np.nanmax(y))
    center0 = float(x[np.nanargmax(y)])
    width0 = max((np.nanmax(x) - np.nanmin(x)) / 20, 0.001)
    p0 = [amplitude0, center0, width0, width0, 0.5]

    bounds = (
        [0, np.nanmin(x), 1e-6, 1e-6, 0],
        [np.inf, np.nanmax(x), np.inf, np.inf, 1]
    )

    try:
        popt, _ = curve_fit(
            pseudo_voigt,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )

        fitted = pseudo_voigt(x, *popt)

        area_numeric_ppm = float(
            np.trapezoid(fitted, x)
            if hasattr(np, "trapezoid")
            else np.trapz(fitted, x)
        )

        amplitude, center, sigma, gamma, eta = popt

        area_analytic_ppm = pseudo_voigt_area_analytic(
            amplitude=amplitude,
            sigma=sigma,
            gamma=gamma,
            eta=eta
        )

        fwhm_ppm = pseudo_voigt_fwhm_estimate(
            sigma=sigma,
            gamma=gamma,
            eta=eta
        )

        fit_params = {
            "amplitude": amplitude,
            "center_ppm": center,
            "sigma_ppm": sigma,
            "gamma_ppm": gamma,
            "eta": eta,
            "fwhm_ppm": fwhm_ppm,
            "area_numeric_ppm": area_numeric_ppm,
            "area_analytic_ppm": area_analytic_ppm,
        }

        return fit_params, fitted, area_numeric_ppm

    except Exception:
        return None, None, np.nan

def fit_single_gaussian_peak(x, y):
    """Fit one Gaussian peak."""
    amplitude0 = float(np.nanmax(y))
    center0 = float(x[np.nanargmax(y)])
    sigma0 = max((np.nanmax(x) - np.nanmin(x)) / 20, 0.001)

    p0 = [amplitude0, center0, sigma0]

    bounds = (
        [0, np.nanmin(x), 1e-6],
        [np.inf, np.nanmax(x), np.inf]
    )

    try:
        popt, _ = curve_fit(
            gaussian_peak,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )

        fitted = gaussian_peak(x, *popt)
        fit_area = float(
            np.trapezoid(fitted, x)
            if hasattr(np, "trapezoid")
            else np.trapz(fitted, x)
        )

        amplitude, center, sigma = popt
        area_analytic = gaussian_area_analytic(amplitude, sigma)
        fwhm_ppm = 2 * np.sqrt(2 * np.log(2)) * abs(sigma)

        fit_params = {
            "model": "gaussian",
            "amplitude": amplitude,
            "center_ppm": center,
            "sigma_ppm": sigma,
            "gamma_ppm": np.nan,
            "eta": np.nan,
            "fwhm_ppm": fwhm_ppm,
            "area_numeric_ppm": fit_area,
            "area_analytic_ppm": area_analytic,
        }

        return fit_params, fitted, fit_area

    except Exception:
        return None, None, np.nan


def fit_single_lorentzian_peak(x, y):
    """Fit one Lorentzian peak."""
    amplitude0 = float(np.nanmax(y))
    center0 = float(x[np.nanargmax(y)])
    gamma0 = max((np.nanmax(x) - np.nanmin(x)) / 20, 0.001)

    p0 = [amplitude0, center0, gamma0]

    bounds = (
        [0, np.nanmin(x), 1e-6],
        [np.inf, np.nanmax(x), np.inf]
    )

    try:
        popt, _ = curve_fit(
            lorentzian_peak,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=20000
        )

        fitted = lorentzian_peak(x, *popt)
        fit_area = float(
            np.trapezoid(fitted, x)
            if hasattr(np, "trapezoid")
            else np.trapz(fitted, x)
        )

        amplitude, center, gamma = popt
        area_analytic = lorentzian_area_analytic(amplitude, gamma)
        fwhm_ppm = 2 * abs(gamma)

        fit_params = {
            "model": "lorentzian",
            "amplitude": amplitude,
            "center_ppm": center,
            "sigma_ppm": np.nan,
            "gamma_ppm": gamma,
            "eta": np.nan,
            "fwhm_ppm": fwhm_ppm,
            "area_numeric_ppm": fit_area,
            "area_analytic_ppm": area_analytic,
        }

        return fit_params, fitted, fit_area

    except Exception:
        return None, None, np.nan


def fit_multiplet_peak(x, y, peak_count, spectrometer_mhz):
    """Fit a constrained pseudo-Voigt multiplet with up to four peaks."""
    amplitude0 = float(np.nanmax(y))
    center0 = float(x[np.nanargmax(y)])
    width0 = max((np.nanmax(x) - np.nanmin(x)) / 40, 0.001)
    spacing0 = 7.0

    default_ratios = {
        1: [1, 1, 1, 1],
        2: [1, 1, 1, 1],
        3: [1, 2, 1, 1],
        4: [1, 3, 3, 1]
    }[peak_count]

    p0 = [
        center0, width0, width0, 0.5, amplitude0, spacing0,
        default_ratios[0], default_ratios[1], default_ratios[2], default_ratios[3]
    ]

    low_bounds = [np.nanmin(x), 1e-6, 1e-6, 0, 0, 0.0, 0, 0, 0, 0]
    high_bounds = [np.nanmax(x), np.inf, np.inf, 1, np.inf, 30.0, 10, 10, 10, 10]

    def model(xx, center, sigma, gamma, eta, amplitude, spacing_hz, r1, r2, r3, r4):
        return multiplet_pseudo_voigt(
            xx, center, sigma, gamma, eta, amplitude, spacing_hz,
            r1, r2, r3, r4, spectrometer_mhz, peak_count
        )

    try:
        popt, _ = curve_fit(
            model,
            x,
            y,
            p0=p0,
            bounds=(low_bounds, high_bounds),
            maxfev=30000
        )
        fitted = model(x, *popt)
        area = float(np.trapezoid(fitted, x) if hasattr(np, 'trapezoid') else np.trapz(fitted, x))
        return popt, fitted, area
    except Exception:
        return None, None, np.nan


def calculate_external_calibration(
    I_sample,
    I_ref,
    n_sample,
    n_ref,
    C_ref,
    NS_sample,
    NS_ref,
    P1_sample,
    P1_ref,
    RG_sample,
    RG_ref,
    volume_sample,
    volume_ref,
    dilution_factor_sample,
    dilution_factor_ref,
    correction_factor,
):
    """External calibration/PULCON-like concentration calculation.

    This is a practical implementation for app validation:
    C_sample = C_ref * (I_sample/I_ref) * (n_ref/n_sample)
               * (NS_ref/NS_sample)
               * (P1_sample/P1_ref)
               * (RG_ref/RG_sample)
               * (V_ref/V_sample)
               * (DF_sample/DF_ref)
               * correction_factor

    The pulse term is included as a simplified probe damping/pulse correction placeholder.
    For a strict implementation, this should be validated using the local instrument protocol.
    """
    if I_ref == 0 or n_sample == 0 or NS_sample == 0 or P1_ref == 0 or RG_sample == 0 or volume_sample == 0 or dilution_factor_ref == 0:
        return np.nan

    return (
        C_ref
        * (I_sample / I_ref)
        * (n_ref / n_sample)
        * (NS_ref / NS_sample)
        * (P1_sample / P1_ref)
        * (RG_ref / RG_sample)
        * (volume_ref / volume_sample)
        * (dilution_factor_sample / dilution_factor_ref)
        * correction_factor
    )


def calculate_reference_mM_from_masses(reference_mass_mg, solvent_mass_mg, solvent_density_g_ml, molecular_weight_g_mol):
    """Calculate reference concentration in mM from reference mass and solvent mass.

    Parameters
    ----------
    reference_mass_mg : float
        Mass of the reference compound in mg.
    solvent_mass_mg : float
        Mass of the solvent in mg.
    solvent_density_g_ml : float
        Solvent density in g/mL.
    molecular_weight_g_mol : float
        Molecular weight of the reference compound in g/mol.

    Formula
    -------
    volume_mL = (solvent_mass_mg / 1000) / solvent_density_g_ml
    moles = (reference_mass_mg / 1000) / molecular_weight_g_mol
    concentration_mM = (moles / volume_L) * 1000

    Simplified:
    concentration_mM = reference_mass_mg * 1_000_000 * solvent_density_g_ml / (molecular_weight_g_mol * solvent_mass_mg)
    """
    if reference_mass_mg <= 0 or solvent_mass_mg <= 0 or solvent_density_g_ml <= 0 or molecular_weight_g_mol <= 0:
        return np.nan

    return (
        reference_mass_mg
        * 1_000_000.0
        * solvent_density_g_ml
        / (molecular_weight_g_mol * solvent_mass_mg)
    )


def plot_spectrum(df, title, xmin=None, xmax=None, regions=None):
    """Plot a spectrum using Plotly.

    Parameters
    ----------
    df : pandas.DataFrame
        Spectrum dataframe with ppm and intensity columns.
    title : str
        Plot title.
    xmin, xmax : float, optional
        Single region to highlight. Kept for backward compatibility.
    regions : list[tuple[float, float]], optional
        Multiple regions to highlight.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["ppm"], y=df["intensity"], mode="lines", name=title))
    fig.update_layout(
        title=title,
        xaxis_title="Chemical shift (ppm)",
        yaxis_title="Intensity",
        xaxis=dict(autorange="reversed"),
        height=450,
    )

    regions_to_plot = []
    if regions is not None:
        regions_to_plot = ensure_regions(regions)
    elif xmin is not None and xmax is not None:
        regions_to_plot = [(xmin, xmax)]

    for idx, region in enumerate(regions_to_plot, start=1):
        low, high = sorted(region)
        fig.add_vrect(
            x0=low,
            x1=high,
            fillcolor="LightSalmon",
            opacity=0.25,
            line_width=0,
            annotation_text=f"R{idx}",
            annotation_position="top left"
        )

    return fig


def extract_zip_to_temp(uploaded_zip):
    """Extract uploaded ZIP to a temporary directory and return the path."""
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, "bruker_upload.zip")
    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmpdir)
    return tmpdir


def find_bruker_experiment_dirs(root_dir):
    """Find folders that look like Bruker experiment directories."""
    root = Path(root_dir)
    candidates = []
    for path in root.rglob("*"):
        if path.is_dir():
            if (path / "fid").exists() and (path / "acqus").exists():
                candidates.append(path)
            elif (path / "pdata" / "1" / "1r").exists() and (path / "acqus").exists():
                candidates.append(path)
    return sorted(list(set(candidates)))


def read_bruker_processed(exp_dir, procno="1"):
    """Read Bruker processed data from pdata/procno using nmrglue."""
    dic, data = ng.bruker.read_pdata(str(Path(exp_dir) / "pdata" / procno))
    data = np.real(data)

    # Generate ppm scale from processed parameters.
    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm = uc.ppm_scale()

    return pd.DataFrame({"ppm": ppm, "intensity": data})


def process_bruker_fid(exp_dir, lb=0.3, zf_factor=2, auto_phase=False):
    """Basic processing of Bruker 1D FID using nmrglue.

    Steps:
    - read Bruker FID
    - remove Bruker digital filter
    - exponential multiplication
    - zero filling
    - Fourier transform
    - optional automatic phase correction
    - generate ppm axis
    """
    dic, data = ng.bruker.read(str(exp_dir))
    data = ng.bruker.remove_digital_filter(dic, data)

    data = ng.proc_base.em(data, lb=lb)
    target_size = int(2 ** np.ceil(np.log2(data.shape[-1] * zf_factor)))
    data = ng.proc_base.zf_size(data, target_size)
    data = ng.proc_base.fft(data)
    data = ng.proc_base.ps(data, p0=0.0, p1=0.0)

    if auto_phase:
        try:
            data = ng.proc_autophase.autops(data, "acme")
        except Exception:
            pass

    data = np.real(data)

    udic = ng.bruker.guess_udic(dic, data)
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm = uc.ppm_scale()

    return pd.DataFrame({"ppm": ppm, "intensity": data})


def make_experiment_label(exp_dir, root_dir=None):
    """Create a readable label for a Bruker experiment folder."""
    exp_path = Path(exp_dir)
    if root_dir is not None:
        try:
            return str(exp_path.relative_to(Path(root_dir)))
        except Exception:
            pass
    return exp_path.name


def process_one_bruker_experiment(exp_dir, mode="Read processed spectrum", procno="1", lb=0.3, zf_factor=2, auto_phase=False):
    """Process or read one Bruker experiment and return a standardized spectrum dataframe."""
    if mode == "Read processed spectrum":
        spectrum_df = read_bruker_processed(exp_dir, procno=procno)
    else:
        spectrum_df = process_bruker_fid(exp_dir, lb=lb, zf_factor=zf_factor, auto_phase=auto_phase)

    return spectrum_df.dropna().sort_values("ppm", ascending=True).reset_index(drop=True)

def calculate_fit_quality_metrics(x_region, y_region, fit, sum_area_selected, fit_area):
    """Calculate local fit quality metrics for one fitted region."""

    x_region = np.asarray(x_region, dtype=float)
    y_region = np.asarray(y_region, dtype=float)

    metrics = {
        "r2_local": np.nan,
        "rmse": np.nan,
        "residual_area": np.nan,
        "fit_sum_ratio": np.nan,
        "snr_local": np.nan,
        "n_points_region": int(len(x_region)),
        "n_points_per_peak": np.nan,
        "fit_quality_label": "not_evaluated",
    }

    if fit is None or len(x_region) < 3:
        metrics["fit_quality_label"] = "fit_failed_or_too_few_points"
        return metrics

    residual = y_region - fit

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_region - np.mean(y_region)) ** 2))

    metrics["r2_local"] = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    metrics["rmse"] = float(np.sqrt(np.mean(residual ** 2)))

    metrics["residual_area"] = abs(float(
        np.trapezoid(residual, x_region)
        if hasattr(np, "trapezoid")
        else np.trapz(residual, x_region)
    ))

    metrics["fit_sum_ratio"] = (
        fit_area / sum_area_selected
        if np.isfinite(fit_area) and np.isfinite(sum_area_selected) and sum_area_selected != 0
        else np.nan
    )

    noise_estimate = np.nanstd(residual)
    signal_estimate = np.nanmax(np.abs(fit))
    metrics["snr_local"] = (
        signal_estimate / noise_estimate
        if np.isfinite(noise_estimate) and noise_estimate > 0
        else np.nan
    )

    if np.isfinite(metrics["fit_sum_ratio"]):
        ratio = metrics["fit_sum_ratio"]
        if 0.95 <= ratio <= 1.05:
            metrics["fit_quality_label"] = "good_fit_sum_agreement"
        elif 0.80 <= ratio < 0.95:
            metrics["fit_quality_label"] = "acceptable_fit_lower_than_sum"
        elif 1.05 < ratio <= 1.10:
            metrics["fit_quality_label"] = "acceptable_fit_higher_than_sum"
        elif ratio < 0.80:
            metrics["fit_quality_label"] = "poor_fit_losing_area"
        elif ratio > 1.10:
            metrics["fit_quality_label"] = "poor_fit_creating_area"

    return metrics

def calculate_integral_for_method(
    df,
    region,
    method,
    baseline_mode,
    peak_count=1,
    spectrometer_mhz=500.0
):
    """Calculate the integral of one spectrum region using the selected integration method."""

    x = df["ppm"].to_numpy(dtype=float)
    y = df["intensity"].to_numpy(dtype=float)

    if baseline_mode in ["None", "None (raw experimental signal)"]:
        baseline_for_integral = "none"
    elif baseline_mode in ["Linear between limits", "Linear baseline between limits"]:
        baseline_for_integral = "linear"
    elif baseline_mode in ["Subtract local minimum", "Subtract local minimum (local offset)"]:
        baseline_for_integral = "minimum"
    else:
        baseline_for_integral = "none"

    low, high = sorted(region)
    mask = (x >= low) & (x <= high)

    if mask.sum() < 2:
        return np.nan, None

    sum_area_selected = integrate_region(
        x,
        y,
        region[0],
        region[1],
        absolute=True,
        baseline=baseline_for_integral
    )

    y_used = y.copy()

    if baseline_for_integral == "minimum":
        y_used = baseline_subtract_minimum(
            x,
            y_used,
            region[0],
            region[1]
        )

    elif baseline_for_integral == "linear":
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        y_low = np.interp(low, x_sorted, y_sorted)
        y_high = np.interp(high, x_sorted, y_sorted)
        baseline_line = np.interp(x, [low, high], [y_low, y_high])
        y_used = y_used - baseline_line

    dx_ppm = float(np.nanmedian(np.abs(np.diff(x[mask])))) if mask.sum() > 2 else np.nan
    dx_hz = dx_ppm * spectrometer_mhz if np.isfinite(dx_ppm) else np.nan

    if method in ["Sum Integration", "Raw Point Sum"]:

        scale_mode = "raw" if method == "Raw Point Sum" else "dx"

        area_sum, sum_diagnostics = sum_integrate_region(
            x,
            y,
            region[0],
            region[1],
            absolute=True,
            baseline=baseline_for_integral,
            scale_mode=scale_mode
        )

        return area_sum, {
            "selected_area_type": (
                "raw_point_sum"
                if scale_mode == "raw"
                else "sum_integration_dx_scaled"
            ),
            "sum_area_selected": area_sum,
            "sum_raw_points": sum_diagnostics.get("sum_raw_points", np.nan),
            "sum_dx_scaled": sum_diagnostics.get("sum_dx_scaled", np.nan),
            "sum_dx_ppm": sum_diagnostics.get("sum_dx_ppm", np.nan),
            "sum_n_points": sum_diagnostics.get("sum_n_points", np.nan),
            "sum_scale_mode": sum_diagnostics.get("sum_scale_mode", ""),
            "fit_area": np.nan,
            "residual_area": np.nan,
            "fit_sum_ratio": np.nan,
            "fit_status": "not_applicable",
            "integration_warning": (
                "Experimental Sum Integration selected. "
                "Both raw point-sum and dx-scaled sum were calculated for scale diagnostics."
            ),
            "dx_ppm": dx_ppm,
            "dx_hz": dx_hz,
        }

    if method == "Trapezoidal":
        area_none = integrate_region(
            x,
            y,
            region[0],
            region[1],
            absolute=True,
            baseline="none"
        )
        area_linear = integrate_region(
            x,
            y,
            region[0],
            region[1],
            absolute=True,
            baseline="linear"
        )
        area_minimum = integrate_region(
            x,
            y,
            region[0],
            region[1],
            absolute=True,
            baseline="minimum"
        )

        if baseline_for_integral == "linear":
            selected_area = area_linear
        elif baseline_for_integral == "minimum":
            selected_area = area_minimum
        else:
            selected_area = area_none

        return selected_area, {
            "selected_area_type": "sum_area",
            "area_none": area_none,
            "area_linear": area_linear,
            "area_minimum": area_minimum,
            "sum_area_selected": selected_area,
            "fit_area": np.nan,
            "residual_area": np.nan,
            "fit_sum_ratio": np.nan,
            "fit_status": "not_applicable",
            "integration_warning": "Sum/Trapezoidal area selected. No model-derived fit area was used.",
            "dx_ppm": dx_ppm,
            "dx_hz": dx_hz,
        }

    if method in ["Single pseudo-Voigt", "Single pseudo-Voigt fit"]:
        fit_params, fit, fit_area_raw = fit_single_peak(
            x[mask],
            y_used[mask]
        )

        fit_area = abs(fit_area_raw) if np.isfinite(fit_area_raw) else np.nan

        area_analytic_ppm = (
            fit_params.get("area_analytic_ppm", np.nan)
            if isinstance(fit_params, dict)
            else np.nan
        )

        fit_numeric_analytic_ratio = (
            fit_area / area_analytic_ppm
            if np.isfinite(fit_area)
            and np.isfinite(area_analytic_ppm)
            and area_analytic_ppm != 0
            else np.nan
        )

        quality = calculate_fit_quality_metrics(
            x_region=x[mask],
            y_region=y_used[mask],
            fit=fit,
            sum_area_selected=sum_area_selected,
            fit_area=fit_area,
        )

        quality["n_points_per_peak"] = (
            quality["n_points_region"] / peak_count
            if peak_count > 0
            else np.nan
        )

        fit_sum_ratio = quality["fit_sum_ratio"]
        residual_area = quality["residual_area"]

        if not np.isfinite(fit_area):
            warning = "Fit failed."
        elif np.isfinite(fit_sum_ratio) and fit_sum_ratio < 0.85:
            warning = "Fit area is much lower than Sum area. The model may be losing tails, overlap, or real signal."
        elif np.isfinite(fit_sum_ratio) and fit_sum_ratio > 1.15:
            warning = "Fit area is higher than Sum area. Possible overfitting or baseline problem."
        else:
            warning = "Fit and Sum are reasonably consistent."

        return fit_area, {
            "selected_area_type": "fit_area_single_pseudovoigt",
            "sum_area_selected": sum_area_selected,
            "fit_area": fit_area,
            "residual_area": residual_area,
            "fit_sum_ratio": fit_sum_ratio,
            "fit_status": "ok" if np.isfinite(fit_area) else "failed",
            "integration_warning": warning,
            "r2_local": quality["r2_local"],
            "rmse": quality["rmse"],
            "snr_local": quality["snr_local"],
            "n_points_region": quality["n_points_region"],
            "n_points_per_peak": quality["n_points_per_peak"],
            "fit_quality_label": quality["fit_quality_label"],
            "dx_ppm": dx_ppm,
            "dx_hz": dx_hz,
            "fit_model": "pseudo_voigt",
            "fit_amplitude": fit_params.get("amplitude", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_center_ppm": fit_params.get("center_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_sigma_ppm": fit_params.get("sigma_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_gamma_ppm": fit_params.get("gamma_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_eta": fit_params.get("eta", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_fwhm_ppm": fit_params.get("fwhm_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_fwhm_hz": (
                fit_params.get("fwhm_ppm", np.nan) * spectrometer_mhz
                if isinstance(fit_params, dict)
                else np.nan
            ),
            "fit_area_analytic_ppm": area_analytic_ppm,
            "fit_numeric_analytic_ratio": fit_numeric_analytic_ratio,
        }

    if method in ["Single Gaussian fit", "Single Lorentzian fit"]:

        if method == "Single Gaussian fit":
            fit_params, fit, fit_area_raw = fit_single_gaussian_peak(
                x[mask],
                y_used[mask]
            )
            selected_area_type = "fit_area_single_gaussian"
        else:
            fit_params, fit, fit_area_raw = fit_single_lorentzian_peak(
                x[mask],
                y_used[mask]
            )
            selected_area_type = "fit_area_single_lorentzian"

        fit_area = abs(fit_area_raw) if np.isfinite(fit_area_raw) else np.nan

        area_analytic_ppm = (
            fit_params.get("area_analytic_ppm", np.nan)
            if isinstance(fit_params, dict)
            else np.nan
        )

        fit_numeric_analytic_ratio = (
            fit_area / area_analytic_ppm
            if np.isfinite(fit_area)
            and np.isfinite(area_analytic_ppm)
            and area_analytic_ppm != 0
            else np.nan
        )

        quality = calculate_fit_quality_metrics(
            x_region=x[mask],
            y_region=y_used[mask],
            fit=fit,
            sum_area_selected=sum_area_selected,
            fit_area=fit_area,
        )

        quality["n_points_per_peak"] = quality["n_points_region"]

        fit_sum_ratio = quality["fit_sum_ratio"]
        residual_area = quality["residual_area"]

        if not np.isfinite(fit_area):
            warning = "Fit failed."
        elif np.isfinite(fit_sum_ratio) and fit_sum_ratio < 0.85:
            warning = f"{method} area is much lower than Sum area. The model may be losing tails, overlap, or real signal."
        elif np.isfinite(fit_sum_ratio) and fit_sum_ratio > 1.15:
            warning = f"{method} area is higher than Sum area. Possible overfitting or baseline problem."
        else:
            warning = f"{method} and Sum are reasonably consistent."

        return fit_area, {
            "selected_area_type": selected_area_type,
            "sum_area_selected": sum_area_selected,
            "fit_area": fit_area,
            "residual_area": residual_area,
            "fit_sum_ratio": fit_sum_ratio,
            "fit_status": "ok" if np.isfinite(fit_area) else "failed",
            "integration_warning": warning,
            "r2_local": quality["r2_local"],
            "rmse": quality["rmse"],
            "snr_local": quality["snr_local"],
            "n_points_region": quality["n_points_region"],
            "n_points_per_peak": quality["n_points_per_peak"],
            "fit_quality_label": quality["fit_quality_label"],
            "dx_ppm": dx_ppm,
            "dx_hz": dx_hz,
            "fit_model": (
                fit_params.get("model", "")
                if isinstance(fit_params, dict)
                else ""
            ),
            "fit_amplitude": fit_params.get("amplitude", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_center_ppm": fit_params.get("center_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_sigma_ppm": fit_params.get("sigma_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_gamma_ppm": fit_params.get("gamma_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_eta": np.nan,
            "fit_fwhm_ppm": fit_params.get("fwhm_ppm", np.nan) if isinstance(fit_params, dict) else np.nan,
            "fit_fwhm_hz": (
                fit_params.get("fwhm_ppm", np.nan) * spectrometer_mhz
                if isinstance(fit_params, dict)
                else np.nan
            ),
            "fit_area_analytic_ppm": area_analytic_ppm,
            "fit_numeric_analytic_ratio": fit_numeric_analytic_ratio,
        }

    if method in ["Multiplet pseudo-Voigt", "Multiplet pseudo-Voigt fit"]:
        fit_params, fit, fit_area_raw = fit_multiplet_peak(
            x[mask],
            y_used[mask],
            peak_count,
            spectrometer_mhz
        )

        fit_area = abs(fit_area_raw) if np.isfinite(fit_area_raw) else np.nan

        quality = calculate_fit_quality_metrics(
            x_region=x[mask],
            y_region=y_used[mask],
            fit=fit,
            sum_area_selected=sum_area_selected,
            fit_area=fit_area,
        )

        quality["n_points_per_peak"] = (
            quality["n_points_region"] / peak_count
            if peak_count > 0
            else np.nan
        )

        fit_sum_ratio = quality["fit_sum_ratio"]
        residual_area = quality["residual_area"]

        if not np.isfinite(fit_area):
            warning = "Fit failed."
        elif np.isfinite(fit_sum_ratio) and fit_sum_ratio < 0.85:
            warning = "Fit area is much lower than Sum area. The model may be losing tails, overlap, or real signal."
        elif np.isfinite(fit_sum_ratio) and fit_sum_ratio > 1.15:
            warning = "Fit area is higher than Sum area. Possible overfitting or baseline problem."
        else:
            warning = "Fit and Sum are reasonably consistent."

        return fit_area, {
            "selected_area_type": "fit_area_multiplet_pseudovoigt",
            "sum_area_selected": sum_area_selected,
            "fit_area": fit_area,
            "residual_area": residual_area,
            "fit_sum_ratio": fit_sum_ratio,
            "fit_status": "ok" if np.isfinite(fit_area) else "failed",
            "integration_warning": warning,
            "r2_local": quality["r2_local"],
            "rmse": quality["rmse"],
            "snr_local": quality["snr_local"],
            "n_points_region": quality["n_points_region"],
            "n_points_per_peak": quality["n_points_per_peak"],
            "fit_quality_label": quality["fit_quality_label"],
            "dx_ppm": dx_ppm,
            "dx_hz": dx_hz,
            "fit_model": "multiplet_pseudo_voigt",
        }

    return np.nan, None

def calculate_integrals_for_regions(df, regions, method, baseline_mode, peak_count=1, spectrometer_mhz=500.0):
    """Calculate integrals for multiple regions and include diagnostic integration values."""

    rows = []

    for idx, region in enumerate(regions, start=1):
        area, diagnostic = calculate_integral_for_method(
            df,
            region,
            method,
            baseline_mode,
            peak_count=peak_count,
            spectrometer_mhz=spectrometer_mhz,
        )

        row = {
            "region_index": idx,
            "region_start_ppm": min(region),
            "region_end_ppm": max(region),
            "integral": area,
        }

        if isinstance(diagnostic, dict):
            row.update(diagnostic)

        rows.append(row)

    return pd.DataFrame(rows)


def run_batch_quantification(batch_spectra, reference_label, sample_labels, ref_region, sample_region, integration_method, baseline_mode, peak_count, spectrometer_mhz, C_ref, n_ref, n_sample, NS_ref, NS_sample, P1_ref, P1_sample, RG_ref, RG_sample, volume_ref, volume_sample, dilution_factor_ref, dilution_factor_sample, correction_factor, analyte_mw=0.0):
    """Run external calibration quantification for multiple sample spectra against one reference spectrum.

    Supports one or multiple reference/sample regions. If the number of reference and sample
    regions is equal, regions are paired by index. If only one reference region is provided,
    it is reused for all sample regions. Otherwise all combinations are calculated.
    """
    if reference_label not in batch_spectra:
        return pd.DataFrame()

    ref_regions = ensure_regions(ref_region, default=(7.19, 7.39))
    sample_regions = ensure_regions(sample_region, default=(3.90, 4.05))

    ref_df_local = batch_spectra[reference_label]
    ref_integrals = calculate_integrals_for_regions(
        ref_df_local,
        ref_regions,
        integration_method,
        baseline_mode,
        peak_count,
        spectrometer_mhz,
    )

    rows = []
    for sample_label in sample_labels:
        if sample_label not in batch_spectra or sample_label == reference_label:
            continue

        sample_df_local = batch_spectra[sample_label]
        sample_integrals = calculate_integrals_for_regions(
            sample_df_local,
            sample_regions,
            integration_method,
            baseline_mode,
            peak_count,
            spectrometer_mhz,
        )

        if len(ref_integrals) == len(sample_integrals):
            pairs = list(zip(ref_integrals.to_dict("records"), sample_integrals.to_dict("records")))
            pairing_mode = "paired_by_index"
        elif len(ref_integrals) == 1:
            ref_record = ref_integrals.to_dict("records")[0]
            pairs = [(ref_record, sample_record) for sample_record in sample_integrals.to_dict("records")]
            pairing_mode = "single_reference_region_reused"
        else:
            pairs = [
                (ref_record, sample_record)
                for ref_record in ref_integrals.to_dict("records")
                for sample_record in sample_integrals.to_dict("records")
            ]
            pairing_mode = "all_combinations"

        concentrations = []
        for ref_record, sample_record in pairs:
            I_ref = ref_record["integral"]
            I_sample = sample_record["integral"]

            C_sample_mM = calculate_external_calibration(
                I_sample=I_sample,
                I_ref=I_ref,
                n_sample=n_sample,
                n_ref=n_ref,
                C_ref=C_ref,
                NS_sample=NS_sample,
                NS_ref=NS_ref,
                P1_sample=P1_sample,
                P1_ref=P1_ref,
                RG_sample=RG_sample,
                RG_ref=RG_ref,
                volume_sample=volume_sample,
                volume_ref=volume_ref,
                dilution_factor_sample=dilution_factor_sample,
                dilution_factor_ref=dilution_factor_ref,
                correction_factor=correction_factor,
            )

            C_sample_mg_ml = np.nan
            if analyte_mw > 0 and np.isfinite(C_sample_mM):
                C_sample_mg_ml = C_sample_mM * analyte_mw / 1000.0

            concentrations.append(C_sample_mM)

            rows.append({
                "result_type": "region_result",
                "sample_name": sample_label,
                "reference_name": reference_label,
                "pairing_mode": pairing_mode,
                "integration_method": integration_method,
                "reference_region_index": ref_record["region_index"],
                "sample_region_index": sample_record["region_index"],
                "reference_region_ppm": f"{ref_record['region_start_ppm']:.4f}-{ref_record['region_end_ppm']:.4f}",
                "sample_region_ppm": f"{sample_record['region_start_ppm']:.4f}-{sample_record['region_end_ppm']:.4f}",
                "I_reference": I_ref,
                "I_sample": I_sample,
                "C_sample_mM": C_sample_mM,
                "C_sample_mg_mL": C_sample_mg_ml,
                "reference_sum_area": ref_record.get("sum_area_selected", np.nan),
                "sample_sum_area": sample_record.get("sum_area_selected", np.nan),
                "reference_fit_area": ref_record.get("fit_area", np.nan),
                "sample_fit_area": sample_record.get("fit_area", np.nan),
                "reference_fit_sum_ratio": ref_record.get("fit_sum_ratio", np.nan),
                "sample_fit_sum_ratio": sample_record.get("fit_sum_ratio", np.nan),
                "reference_residual_area": ref_record.get("residual_area", np.nan),
                "sample_residual_area": sample_record.get("residual_area", np.nan),
                "reference_integration_warning": ref_record.get("integration_warning", ""),
                "sample_integration_warning": sample_record.get("integration_warning", ""),
                "reference_r2_local": ref_record.get("r2_local", np.nan),
                "sample_r2_local": sample_record.get("r2_local", np.nan),
                "reference_rmse": ref_record.get("rmse", np.nan),
                "sample_rmse": sample_record.get("rmse", np.nan),
                "reference_snr_local": ref_record.get("snr_local", np.nan),
                "sample_snr_local": sample_record.get("snr_local", np.nan),
                "reference_n_points_region": ref_record.get("n_points_region", np.nan),
                "sample_n_points_region": sample_record.get("n_points_region", np.nan),
                "reference_n_points_per_peak": ref_record.get("n_points_per_peak", np.nan),
                "sample_n_points_per_peak": sample_record.get("n_points_per_peak", np.nan),
                "reference_fit_quality_label": ref_record.get("fit_quality_label", ""),
                "sample_fit_quality_label": sample_record.get("fit_quality_label", ""),
                "reference_dx_ppm": ref_record.get("dx_ppm", np.nan),
                "sample_dx_ppm": sample_record.get("dx_ppm", np.nan),
                "reference_dx_hz": ref_record.get("dx_hz", np.nan),
                "sample_dx_hz": sample_record.get("dx_hz", np.nan),

                "reference_fit_amplitude": ref_record.get("fit_amplitude", np.nan),
                "sample_fit_amplitude": sample_record.get("fit_amplitude", np.nan),
                "reference_fit_center_ppm": ref_record.get("fit_center_ppm", np.nan),
                "sample_fit_center_ppm": sample_record.get("fit_center_ppm", np.nan),

                "reference_fit_sigma_ppm": ref_record.get("fit_sigma_ppm", np.nan),
                "sample_fit_sigma_ppm": sample_record.get("fit_sigma_ppm", np.nan),
                "reference_fit_gamma_ppm": ref_record.get("fit_gamma_ppm", np.nan),
                "sample_fit_gamma_ppm": sample_record.get("fit_gamma_ppm", np.nan),

                "reference_fit_eta": ref_record.get("fit_eta", np.nan),
                "sample_fit_eta": sample_record.get("fit_eta", np.nan),

                "reference_fit_fwhm_ppm": ref_record.get("fit_fwhm_ppm", np.nan),
                "sample_fit_fwhm_ppm": sample_record.get("fit_fwhm_ppm", np.nan),
                "reference_fit_fwhm_hz": ref_record.get("fit_fwhm_hz", np.nan),
                "sample_fit_fwhm_hz": sample_record.get("fit_fwhm_hz", np.nan),

                "reference_fit_area_analytic_ppm": ref_record.get("fit_area_analytic_ppm", np.nan),
                "sample_fit_area_analytic_ppm": sample_record.get("fit_area_analytic_ppm", np.nan),
                "reference_fit_numeric_analytic_ratio": ref_record.get("fit_numeric_analytic_ratio", np.nan),
                "sample_fit_numeric_analytic_ratio": sample_record.get("fit_numeric_analytic_ratio", np.nan),
            })

        valid_conc = np.array([x for x in concentrations if np.isfinite(x)], dtype=float)
        if valid_conc.size > 0:
            mean_mM = float(np.mean(valid_conc))
            std_mM = float(np.std(valid_conc, ddof=1)) if valid_conc.size > 1 else 0.0
            rsd_percent = float((std_mM / mean_mM) * 100.0) if mean_mM != 0 else np.nan
            mean_mg_ml = mean_mM * analyte_mw / 1000.0 if analyte_mw > 0 else np.nan
        else:
            mean_mM = np.nan
            std_mM = np.nan
            rsd_percent = np.nan
            mean_mg_ml = np.nan

        rows.append({
            "result_type": "sample_summary",
            "sample_name": sample_label,
            "reference_name": reference_label,
            "pairing_mode": pairing_mode if len(pairs) > 0 else "none",
            "integration_method": integration_method,
            "reference_region_index": np.nan,
            "sample_region_index": np.nan,
            "reference_region_ppm": "; ".join([f"{min(r):.4f}-{max(r):.4f}" for r in ref_regions]),
            "sample_region_ppm": "; ".join([f"{min(r):.4f}-{max(r):.4f}" for r in sample_regions]),
            "I_reference": np.nan,
            "I_sample": np.nan,
            "C_sample_mM": mean_mM,
            "C_sample_mg_mL": mean_mg_ml,
            "C_sample_std_mM": std_mM,
            "C_sample_RSD_percent": rsd_percent,
            "n_region_results": int(valid_conc.size),
        })

    return pd.DataFrame(rows)


def get_spectrum_from_source(source_label, uploaded_csv, processed_key):
    """Return spectrum dataframe from CSV upload or session state processed Bruker data."""
    processed_df = st.session_state.get(processed_key)

    if source_label == "Auto (prefer Bruker)":
        if processed_df is not None:
            return processed_df
        if uploaded_csv is not None:
            df = load_csv(uploaded_csv)
            ppm_col, intensity_col = detect_ppm_intensity_columns(df)
            return clean_spectrum_df(df, ppm_col, intensity_col)
        return None

    if source_label == "Uploaded CSV":
        if uploaded_csv is None:
            return None
        df = load_csv(uploaded_csv)
        ppm_col, intensity_col = detect_ppm_intensity_columns(df)
        return clean_spectrum_df(df, ppm_col, intensity_col)

    if source_label == "Processed Bruker spectrum":
        return processed_df

    return None


def get_current_source_label(source_label, uploaded_csv, processed_key):
    """Return a human-readable label for the currently active spectrum source."""
    processed_df = st.session_state.get(processed_key)

    if source_label == "Auto (prefer Bruker)":
        if processed_df is not None:
            return "Bruker (processed)"
        if uploaded_csv is not None:
            return "CSV"
        return "Not loaded"

    if source_label == "Uploaded CSV":
        return "CSV" if uploaded_csv is not None else "CSV selected, not loaded"

    if source_label == "Processed Bruker spectrum":
        return "Bruker (processed)" if processed_df is not None else "Bruker selected, not loaded"

    return "Not loaded"


# =========================================================
# Sidebar
# =========================================================
# LOGOs (optional)
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["qNMRSTD_logo.png","LAABio.png", "inmetro.png"]: 
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.sidebar.image(Image.open(p), use_container_width=True)
    except Exception:
        pass

st.sidebar.link_button(
    "📘 Tutorial / Documentation",
    "https://github.com/RicardoMBorges/qnmr-externalstd/blob/main/README.md",
    use_container_width=True
)

st.sidebar.divider()


st.sidebar.title("qNMR External Calibration")
st.sidebar.caption("External Calibration and Probe Damping Correction with External Reference")

with st.sidebar.expander("General tutorial", expanded=False):
    st.markdown(
        """
        This app is designed for qNMR quantification using an external reference.

        Recommended workflow:
        1. Upload or process the Bruker spectra.
        2. Define the external reference parameters.
        3. Define the sample signal and experimental parameters.
        4. Select integration regions.
        5. Run quantification using the button.

        The first version is intentionally semi-automated. Visual inspection is still required.
        """
    )

with st.sidebar.expander("Expected CSV format", expanded=False):
    st.markdown(
        """
        CSV files should contain at least two columns:

        - `ppm`: chemical shift axis
        - `intensity`: signal intensity

        The app also tries to infer column names such as `x`, `y`, `signal`, or `real`.
        """
    )

st.sidebar.subheader("Input source")

st.sidebar.markdown("**Important:** You can use CSV OR Bruker processed data. If both are present, priority can be selected below.")

reference_source = st.sidebar.selectbox(
    "Reference spectrum source",
    ["Auto (prefer Bruker)", "Uploaded CSV", "Processed Bruker spectrum"],
    help="Choose how the reference spectrum is selected. Auto will use Bruker if available."
)

sample_source = st.sidebar.selectbox(
    "Sample spectrum source",
    ["Auto (prefer Bruker)", "Uploaded CSV", "Processed Bruker spectrum"],
    help="Choose how the sample spectrum is selected. Auto will use Bruker if available."
)

ref_file = None
sample_file = None

if reference_source in ["Uploaded CSV", "Auto (prefer Bruker)"]:
    ref_file = st.sidebar.file_uploader("Upload reference spectrum CSV", type="csv")

if sample_source in ["Uploaded CSV", "Auto (prefer Bruker)"]:
    sample_file = st.sidebar.file_uploader("Upload sample spectrum CSV", type="csv")

# =========================================================
# Default quantification parameters
# =========================================================

# Sidebar is intentionally kept for input source/import only.
# Quantification parameters are defined in the Reference, Sample, and Integration tabs.
st.session_state.setdefault("reference_name", "Caffeine")
st.session_state.setdefault("reference_input_mode", "Mass-based")
st.session_state.setdefault("ref_mw", 194.0804)
st.session_state.setdefault("reference_mass_mg", 0.7356)
st.session_state.setdefault("solvent_mass_mg", 994.0)
st.session_state.setdefault("solvent_density_g_ml", 1.478)
st.session_state.setdefault("C_ref", 5.6357)
st.session_state.setdefault("n_ref", 1.0)
st.session_state.setdefault("NS_ref", 16.0)
st.session_state.setdefault("P1_ref", 10.0)
st.session_state.setdefault("RG_ref", 1.0)
st.session_state.setdefault("volume_ref", 600.0)
st.session_state.setdefault("dilution_factor_ref", 1.0)
st.session_state.setdefault("ref_region", [(7.19, 7.39)])

st.session_state.setdefault("sample_name", "Sample_001")
st.session_state.setdefault("analyte_name", "Target signal")
st.session_state.setdefault("n_sample", 1.0)
st.session_state.setdefault("NS_sample", 16.0)
st.session_state.setdefault("P1_sample", 10.0)
st.session_state.setdefault("RG_sample", 1.0)
st.session_state.setdefault("volume_sample", 600.0)
st.session_state.setdefault("dilution_factor_sample", 1.0)
st.session_state.setdefault("analyte_mw", 0.0)
st.session_state.setdefault("sample_region", [(3.90, 4.05)])

st.session_state.setdefault("correction_factor", 1.0)
st.session_state.setdefault("spectrometer_mhz", 500.0)
st.session_state.setdefault("peak_count", 1)
st.session_state.setdefault("integration_method", "Trapezoidal")
st.session_state.setdefault("baseline_mode", "None")

run_quantification = False


# =========================================================
# Prepare spectra and show current sources
# =========================================================

ref_df = get_spectrum_from_source(reference_source, ref_file, "processed_reference")
sample_df = get_spectrum_from_source(sample_source, sample_file, "processed_sample")

current_reference_source = get_current_source_label(reference_source, ref_file, "processed_reference")
current_sample_source = get_current_source_label(sample_source, sample_file, "processed_sample")

source_col1, source_col2 = st.columns(2)
with source_col1:
    st.info(f"Current Reference Source: **{current_reference_source}**")
with source_col2:
    st.info(f"Current Sample Source: **{current_sample_source}**")


# =========================================================
# Main Tabs
# =========================================================

tabs = st.tabs([
    "Bruker Processing",
    "Data",
    "Reference",
    "Sample",
    "Integration",
    "Quantification",
    "Report"
])


# =========================================================
# Tab 1: Bruker Processing
# =========================================================

with tabs[0]:
    st.header("Bruker Processing")
    st.markdown(
        """
        This tab supports two workflows:

        1. **Single experiment mode**: upload one Bruker ZIP, process it, and save it as Reference or Sample.
        2. **Batch Bruker Mode**: upload one ZIP containing many Bruker experiments, select one as reference, and process all samples together.
        """
    )

    with st.expander("Tutorial: Bruker ZIP structure", expanded=False):
        st.markdown(
            """
            The ZIP file may contain one or many Bruker experiments. Each experiment should contain either raw data or processed data.

            Example with multiple experiments:

            ```text
            project_folder/
                reference_caffeine/
                    1/
                        fid
                        acqus
                        pdata/1/1r
                sample_001/
                    1/
                        fid
                        acqus
                        pdata/1/1r
                sample_002/
                    1/
                        fid
                        acqus
                        pdata/1/1r
            ```

            Use **Read processed spectrum** if the spectra were already processed in TopSpin.
            Use **Process raw FID** for basic Python processing.
            """
        )

    if not NMRGLUE_AVAILABLE:
        st.error("nmrglue is not installed. Add `nmrglue` to requirements.txt before using this tab.")
    else:
        bruker_workflow = st.radio(
            "Bruker workflow",
            ["Batch Bruker Mode", "Single experiment mode"],
            horizontal=True,
            help="Use batch mode when one ZIP contains many Bruker experiments."
        )

        if bruker_workflow == "Single experiment mode":
            st.subheader("Single experiment mode")
            st.markdown(
                "Upload one ZIP file containing one complete Bruker experiment folder. Process the reference first and save it as **Reference**. Then replace the ZIP with the sample experiment, select **Sample**, and process again."
            )

            bruker_zip = st.file_uploader("Upload Bruker experiment ZIP", type="zip", key="single_bruker_zip")

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                bruker_mode = st.selectbox(
                    "Processing mode",
                    ["Read processed spectrum", "Process raw FID"],
                    help="Processed mode reads pdata/1/1r. Raw FID mode applies basic Python processing.",
                    key="single_bruker_mode"
                )
            with col_b2:
                destination = st.selectbox(
                    "Send result as",
                    ["Reference", "Sample"],
                    help="Process the reference ZIP first and save as Reference. Then upload the sample ZIP and save as Sample.",
                    key="single_destination"
                )
            with col_b3:
                procno = st.text_input("Processed data number", value="1", help="Usually 1 for pdata/1.", key="single_procno")

            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                lb = st.number_input("Line broadening (Hz)", value=0.3, min_value=0.0, step=0.1, key="single_lb")
            with col_p2:
                zf_factor = st.number_input("Zero filling factor", value=2, min_value=1, max_value=8, step=1, key="single_zf")
            with col_p3:
                auto_phase = st.checkbox("Try automatic phase correction", value=False, key="single_autophase")

            run_bruker_processing = st.button("Run Bruker processing", key="run_single_bruker")

            if run_bruker_processing:
                if bruker_zip is None:
                    st.warning("Upload a Bruker ZIP file first.")
                else:
                    try:
                        tmpdir = extract_zip_to_temp(bruker_zip)
                        exp_dirs = find_bruker_experiment_dirs(tmpdir)

                        if len(exp_dirs) == 0:
                            st.error("No Bruker experiment folder was found inside the ZIP.")
                        else:
                            exp_dir = exp_dirs[0]
                            st.info(f"Using experiment folder: {exp_dir}")

                            spectrum_df = process_one_bruker_experiment(
                                exp_dir,
                                mode=bruker_mode,
                                procno=procno,
                                lb=lb,
                                zf_factor=zf_factor,
                                auto_phase=auto_phase,
                            )

                            if destination == "Reference":
                                st.session_state.processed_reference = spectrum_df
                            else:
                                st.session_state.processed_sample = spectrum_df

                            st.success(f"Processed spectrum saved as {destination} spectrum.")
                            st.plotly_chart(
                                plot_spectrum(spectrum_df, f"Processed Bruker spectrum ({destination})"),
                                use_container_width=True,
                                key=f"single_processed_{destination}"
                            )
                            st.dataframe(spectrum_df.head())

                            csv_data = spectrum_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download processed spectrum CSV",
                                data=csv_data,
                                file_name=f"processed_bruker_{destination.lower()}.csv",
                                mime="text/csv"
                            )

                    except Exception as e:
                        st.error(f"Bruker processing failed: {e}")

        else:
            st.subheader("Batch Bruker Mode")
            st.markdown(
                "Upload one ZIP containing many Bruker experiments. The app will detect all experiment folders, let you choose the reference, and process selected samples in batch."
            )

            batch_zip = st.file_uploader("Upload batch Bruker ZIP", type="zip", key="batch_bruker_zip")

            col_batch1, col_batch2, col_batch3 = st.columns(3)
            with col_batch1:
                batch_mode = st.selectbox(
                    "Batch processing mode",
                    ["Read processed spectrum", "Process raw FID"],
                    key="batch_mode",
                    help="Use processed mode for pdata/1/1r files."
                )
            with col_batch2:
                batch_procno = st.text_input("Processed data number", value="1", key="batch_procno")
            with col_batch3:
                auto_guess_reference = st.checkbox(
                    "Auto-guess reference name",
                    value=True,
                    help="Looks for folder names containing ref, reference, std, standard, caffeine, or cafeine.",
                    key="auto_guess_reference"
                )

            col_batch4, col_batch5, col_batch6 = st.columns(3)
            with col_batch4:
                batch_lb = st.number_input("Batch line broadening (Hz)", value=0.3, min_value=0.0, step=0.1, key="batch_lb")
            with col_batch5:
                batch_zf = st.number_input("Batch zero filling factor", value=2, min_value=1, max_value=8, step=1, key="batch_zf")
            with col_batch6:
                batch_autophase = st.checkbox("Try automatic phase correction", value=False, key="batch_autophase")

            detect_batch = st.button("Detect experiments in ZIP", key="detect_batch")

            if detect_batch:
                if batch_zip is None:
                    st.warning("Upload a batch Bruker ZIP first.")
                else:
                    try:
                        tmpdir = extract_zip_to_temp(batch_zip)
                        exp_dirs = find_bruker_experiment_dirs(tmpdir)
                        if len(exp_dirs) == 0:
                            st.error("No Bruker experiment folders were detected.")
                        else:
                            exp_table = []
                            for exp_dir in exp_dirs:
                                label = make_experiment_label(exp_dir, tmpdir)
                                exp_table.append({"label": label, "path": str(exp_dir)})
                            st.session_state.batch_exp_table = pd.DataFrame(exp_table)
                            st.session_state.batch_tmpdir = tmpdir
                            st.success(f"Detected {len(exp_table)} Bruker experiment(s).")
                    except Exception as e:
                        st.error(f"Batch detection failed: {e}")

            if "batch_exp_table" in st.session_state and st.session_state.batch_exp_table is not None:
                exp_table = st.session_state.batch_exp_table

                st.markdown("### Assign labels and roles")
                st.markdown(
                    """
                    Bruker experiment folders often use numeric labels. Rename each experiment and assign its role.

                    - Use exactly **one** `Reference`.
                    - Use one or more `Sample` rows.
                    - Use `Ignore` for spectra that should not be processed.
                    """
                )

                role_table = exp_table.copy()
                if "original_label" not in role_table.columns:
                    role_table["original_label"] = role_table["label"]
                role_table = role_table[["original_label", "label", "path"]]
                role_table["role"] = "Sample"

                if auto_guess_reference:
                    ref_keywords = ["ref", "reference", "std", "standard", "caffeine", "cafeine", "cafeína"]
                    for idx, row in role_table.iterrows():
                        label_text = str(row["label"]).lower()
                        if any(k.lower() in label_text for k in ref_keywords):
                            role_table.loc[idx, "role"] = "Reference"
                            break

                edited_role_table = st.data_editor(
                    role_table,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "original_label": st.column_config.TextColumn(
                            "Original Bruker label",
                            disabled=True,
                            help="Detected folder label from the Bruker ZIP."
                        ),
                        "label": st.column_config.TextColumn(
                            "User label",
                            help="Rename this experiment, for example ref_caffeine, sample_001, sample_002."
                        ),
                        "role": st.column_config.SelectboxColumn(
                            "Role",
                            options=["Reference", "Sample", "Ignore"],
                            required=True,
                            help="Choose one Reference, one or more Samples, or Ignore."
                        ),
                        "path": st.column_config.TextColumn(
                            "Path",
                            disabled=True,
                            help="Internal path of the detected Bruker experiment."
                        ),
                    },
                    key="batch_role_editor"
                )

                reference_rows = edited_role_table[edited_role_table["role"] == "Reference"]
                sample_rows = edited_role_table[edited_role_table["role"] == "Sample"]
                ignored_rows = edited_role_table[edited_role_table["role"] == "Ignore"]

                if len(reference_rows) == 0:
                    st.error("Select exactly one experiment as Reference.")
                elif len(reference_rows) > 1:
                    st.error("Only one experiment can be assigned as Reference.")
                else:
                    st.success(f"Reference selected: {reference_rows.iloc[0]['label']}")

                st.info(f"Samples selected: {len(sample_rows)} | Ignored: {len(ignored_rows)}")

                reference_label = reference_rows.iloc[0]["label"] if len(reference_rows) == 1 else None
                sample_labels = sample_rows["label"].tolist()
                exp_table = edited_role_table.copy()

                run_batch_processing = st.button("Process selected batch spectra", key="run_batch_processing")

                if run_batch_processing:
                    processed = {}
                    progress = st.progress(0)
                    status = st.empty()

                    path_map = dict(zip(exp_table["label"], exp_table["path"]))
                    selected_labels = [reference_label] + [x for x in sample_labels if x != reference_label]

                    for i, label in enumerate(selected_labels):
                        try:
                            status.info(f"Processing {label}")
                            spectrum_df = process_one_bruker_experiment(
                                Path(path_map[label]),
                                mode=batch_mode,
                                procno=batch_procno,
                                lb=batch_lb,
                                zf_factor=batch_zf,
                                auto_phase=batch_autophase,
                            )
                            processed[label] = spectrum_df
                        except Exception as e:
                            st.warning(f"Failed to process {label}: {e}")
                        progress.progress((i + 1) / len(selected_labels))

                    st.session_state.batch_spectra = processed
                    st.session_state.batch_reference_name = reference_label
                    st.session_state.selected_batch_sample_labels = [x for x in sample_labels if x in processed]

                    if reference_label in processed:
                        st.session_state.processed_reference = processed[reference_label]
                    if len(sample_labels) > 0 and sample_labels[0] in processed:
                        st.session_state.processed_sample = processed[sample_labels[0]]

                    status.success(f"Processed {len(processed)} spectrum/spectra.")

                if len(st.session_state.batch_spectra) > 0:
                    st.success(f"Batch spectra available: {len(st.session_state.batch_spectra)}")
                    preview_label = st.selectbox(
                        "Preview processed spectrum",
                        list(st.session_state.batch_spectra.keys()),
                        key="batch_preview_label"
                    )
                    preview_df = st.session_state.batch_spectra[preview_label]
                    st.plotly_chart(
                        plot_spectrum(preview_df, f"Batch preview: {preview_label}"),
                        use_container_width=True,
                        key=f"batch_preview_{preview_label}"
                    )

                    st.download_button(
                        "Download all processed spectra as one CSV",
                        data=pd.concat(
                            [df.assign(spectrum_name=name) for name, df in st.session_state.batch_spectra.items()],
                            ignore_index=True
                        ).to_csv(index=False).encode("utf-8"),
                        file_name="batch_processed_bruker_spectra.csv",
                        mime="text/csv"
                    )
                    st.info("Batch spectra are ready. Go to the Reference, Sample, and Integration tabs to adjust parameters, then run quantification in the Quantification tab.")


# =========================================================
# Tab 2: Data
# =========================================================

with tabs[1]:
    st.header("Data")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Reference spectrum")
        if ref_df is not None:
            st.dataframe(ref_df.head())
            st.plotly_chart(plot_spectrum(ref_df, "Reference spectrum"), use_container_width=True, key="data_reference_spectrum")
        else:
            st.warning("No reference spectrum loaded.")

    with c2:
        st.subheader("Sample spectrum")
        if sample_df is not None:
            st.dataframe(sample_df.head())
            st.plotly_chart(plot_spectrum(sample_df, "Sample spectrum"), use_container_width=True, key="data_sample_spectrum")
        else:
            st.warning("No sample spectrum loaded.")


# =========================================================
# Tab 3: Reference Parameters
# =========================================================

with tabs[2]:
    st.header("Reference Parameters")

    reference_name = st.text_input("Reference compound", value=st.session_state.reference_name, key="reference_name")
    reference_input_mode = st.selectbox("Input mode", ["Mass-based", "Direct mM"], index=0 if st.session_state.reference_input_mode=="Mass-based" else 1, key="reference_input_mode")
    ref_mw = st.number_input("MW (g/mol)", value=st.session_state.ref_mw, key="ref_mw")

    if reference_input_mode == "Mass-based":
        reference_mass_mg = st.number_input("Reference mass (mg)", value=st.session_state.reference_mass_mg, key="reference_mass_mg")
        solvent_mass_mg = st.number_input("Solvent mass (mg)", value=st.session_state.solvent_mass_mg, key="solvent_mass_mg")
        solvent_density_g_ml = st.number_input("Solvent density (g/mL)", value=st.session_state.solvent_density_g_ml, key="solvent_density_g_ml")
        C_ref = calculate_reference_mM_from_masses(reference_mass_mg, solvent_mass_mg, solvent_density_g_ml, ref_mw)
        st.metric("C_ref (mM)", f"{C_ref:.6g}" if np.isfinite(C_ref) else "Invalid")
        st.session_state.C_ref = C_ref
    else:
        C_ref = st.number_input("C_ref (mM)", value=st.session_state.C_ref, key="C_ref")

    n_ref = st.number_input("nH (ref)", value=st.session_state.n_ref, key="n_ref")
    NS_ref = st.number_input("NS (ref)", value=st.session_state.NS_ref, key="NS_ref")
    P1_ref = st.number_input("P1 (µs, ref)", value=st.session_state.P1_ref, key="P1_ref")
    RG_ref = st.number_input("RG (ref)", value=st.session_state.RG_ref, key="RG_ref")
    volume_ref = st.number_input("Volume (µL, ref)", value=st.session_state.volume_ref, key="volume_ref")
    dilution_factor_ref = st.number_input("DF (ref)", value=st.session_state.dilution_factor_ref, key="dilution_factor_ref")

    ref_region_text = st.text_area(
        "Reference regions (ppm)",
        value=format_regions(ensure_regions(st.session_state.ref_region, default=(7.19, 7.39))),
        help="Enter one region per line, for example:\n7.19, 7.39\n3.85, 3.95"
    )
    parsed_ref_regions = parse_regions(ref_region_text)
    if parsed_ref_regions:
        st.session_state.ref_region = parsed_ref_regions
    else:
        st.warning("Invalid reference region format. Use one region per line: start, end")
    if ref_df is not None:
        st.plotly_chart(
            plot_spectrum(ref_df, "Reference integration regions", regions=ensure_regions(st.session_state.ref_region, default=(7.19, 7.39))),
            use_container_width=True,
            key="reference_tab_region_plot"
        )


# =========================================================
# Tab 4: Sample Parameters
# =========================================================

with tabs[3]:
    st.header("Sample Parameters")

    sample_name = st.text_input("Sample name", value=st.session_state.sample_name, key="sample_name")
    analyte_name = st.text_input("Analyte label", value=st.session_state.analyte_name, key="analyte_name")
    n_sample = st.number_input("nH (sample)", value=st.session_state.n_sample, key="n_sample")
    NS_sample = st.number_input("NS (sample)", value=st.session_state.NS_sample, key="NS_sample")
    P1_sample = st.number_input("P1 (µs, sample)", value=st.session_state.P1_sample, key="P1_sample")
    RG_sample = st.number_input("RG (sample)", value=st.session_state.RG_sample, key="RG_sample")
    volume_sample = st.number_input("Volume (µL, sample)", value=st.session_state.volume_sample, key="volume_sample")
    dilution_factor_sample = st.number_input("DF (sample)", value=st.session_state.dilution_factor_sample, key="dilution_factor_sample")
    analyte_mw = st.number_input("Analyte MW", value=st.session_state.analyte_mw, key="analyte_mw")

    sample_region_text = st.text_area(
        "Sample regions (ppm)",
        value=format_regions(ensure_regions(st.session_state.sample_region, default=(3.90, 4.05))),
        help="Enter one region per line, for example:\n3.90, 4.05\n6.10, 6.25"
    )
    parsed_sample_regions = parse_regions(sample_region_text)
    if parsed_sample_regions:
        st.session_state.sample_region = parsed_sample_regions
    else:
        st.warning("Invalid sample region format. Use one region per line: start, end")
    if sample_df is not None:
        st.plotly_chart(
            plot_spectrum(sample_df, "Sample integration regions", regions=ensure_regions(st.session_state.sample_region, default=(3.90, 4.05))),
            use_container_width=True,
            key="sample_tab_region_plot"
        )


# =========================================================
# Tab 5: Integration
# =========================================================
with tabs[4]:
    st.header("Integration")

    with st.expander("How to choose the integration method", expanded=False):
        st.markdown(
            """
### Recommended choice

For most qNMR applications, start with **Trapezoidal** integration.

### 1. Trapezoidal

Use when the signal is reasonably isolated and the baseline is acceptable.

This is the closest option to classical Sum integration. It integrates the experimental signal directly, without assuming a mathematical peak shape.

Best for:
- clean qNMR signals
- external calibration
- validation work
- reporting quantitative values

Interpretation:
- this should usually be your primary quantitative area.

---

### 2. Single pseudo-Voigt fit

Use only when the signal behaves like one simple peak.

This method fits one idealized peak shape and calculates the area from the model.

Best for:
- isolated singlets
- checking line shape
- diagnostic comparison with Trapezoidal area

Be careful:
- real NMR peaks are often not perfect pseudo-Voigt peaks
- tails, asymmetry, and overlap may be lost
- always inspect Fit/Sum ratio

---

### 3. Multiplet pseudo-Voigt fit

Use when the selected signal is a known multiplet.

This method fits several constrained pseudo-Voigt components.

Best for:
- doublets
- triplets
- quartets
- partially structured multiplets

Be careful:
- poor peak-count choice can produce misleading areas
- always compare with Trapezoidal area

---

### Practical rule

Use Trapezoidal as the main quantitative method.

Use pseudo-Voigt methods as diagnostic/model-based methods.
"""
        )

    integration_method = st.selectbox(
        "Method",
        [
            "Sum Integration",
            "Raw Point Sum",            
            "Trapezoidal",
            "Single pseudo-Voigt fit",
            "Multiplet pseudo-Voigt fit",
            "Single Lorentzian fit",
            "Single Gaussian fit"
        ],
        key="integration_method",
        help="For qNMR, Trapezoidal/Sum is usually the primary quantitative area."
    )

    baseline_mode = st.selectbox(
        "Baseline correction",
        [
            "None (raw experimental signal)",
            "Linear baseline between limits",
            "Subtract local minimum (local offset)"
        ],
        key="baseline_mode"
    )

    with st.expander("How to choose baseline correction", expanded=False):
        st.markdown(
            """
### Baseline correction overview

Baseline correction strongly affects qNMR integration.

Aggressive baseline correction may remove real signal area.

---

### 1. None (raw experimental signal)

No baseline correction is applied.

Best for:
- flat baselines
- validation
- direct experimental integration

---

### 2. Linear baseline between limits

A straight baseline is interpolated between the integration limits.

Best for:
- routine qNMR
- slight baseline slopes

Usually the safest default option.

---

### 3. Subtract local minimum (local offset)

Subtracts the minimum intensity inside the region.

Best for:
- exploratory metabolomics
- local offset correction

Be careful:
- may underestimate broad signals
- sensitive to noise
"""
        )

    correction_factor = st.number_input(
        "Empirical correction factor",
        value=st.session_state.correction_factor,
        min_value=0.0,
        help="Use 1.0 initially.",
        key="correction_factor"
    )

    spectrometer_mhz = st.number_input(
        "MHz",
        value=st.session_state.spectrometer_mhz,
        key="spectrometer_mhz"
    )

    peak_count = st.selectbox(
        "Peaks",
        [1, 2, 3, 4],
        index=st.session_state.peak_count - 1,
        key="peak_count"
    )

    if ref_df is not None and sample_df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_spectrum(
                    ref_df,
                    "Reference regions",
                    regions=ensure_regions(
                        st.session_state.ref_region,
                        default=(7.19, 7.39)
                    )
                ),
                use_container_width=True,
                key="integration_reference_region_plot"
            )

        with col2:
            st.plotly_chart(
                plot_spectrum(
                    sample_df,
                    "Sample regions",
                    regions=ensure_regions(
                        st.session_state.sample_region,
                        default=(3.90, 4.05)
                    )
                ),
                use_container_width=True,
                key="integration_sample_region_plot"
            )

    else:
        st.warning("Load both reference and sample spectra to inspect integration regions.")


# =========================================================
# Tab 6: Quantification
# =========================================================

with tabs[5]:
    st.header("Quantification")

    with st.expander("Method, equation, definitions, and references", expanded=False):
        st.markdown(
            r"""
            ## External calibration with probe damping correction

            This app implements a practical external calibration approach inspired by PULCON-style qNMR workflows. The sample and the external reference are measured separately, and the analyte concentration is calculated from the ratio between the corrected signal area of the sample and the corrected signal area of the reference.

            ### Working equation used in this app

            $$
            C_{sample} = C_{ref}
            	imes rac{I_{sample}}{I_{ref}}
            	imes rac{N_{ref}}{N_{sample}}
            	imes rac{NS_{ref}}{NS_{sample}}
            	imes rac{P1_{sample}}{P1_{ref}}
            	imes rac{RG_{ref}}{RG_{sample}}
            	imes rac{V_{ref}}{V_{sample}}
            	imes rac{DF_{sample}}{DF_{ref}}
            	imes CF
            $$

            ### Definitions

            | Symbol | Meaning |
            |---|---|
            | $C_{sample}$ | Calculated analyte concentration in the sample |
            | $C_{ref}$ | Known concentration of the external reference |
            | $I_{sample}$ | Integrated signal area of the selected sample resonance |
            | $I_{ref}$ | Integrated signal area of the selected reference resonance |
            | $N_{sample}$ | Number of protons represented by the sample signal |
            | $N_{ref}$ | Number of protons represented by the reference signal |
            | $NS_{sample}$ | Number of scans used for the sample spectrum |
            | $NS_{ref}$ | Number of scans used for the reference spectrum |
            | $P1_{sample}$ | 90° pulse length used for the sample spectrum |
            | $P1_{ref}$ | 90° pulse length used for the reference spectrum |
            | $RG_{sample}$ | Receiver gain used for the sample spectrum |
            | $RG_{ref}$ | Receiver gain used for the reference spectrum |
            | $V_{sample}$ | Final NMR volume of the sample |
            | $V_{ref}$ | Final NMR volume of the reference |
            | $DF_{sample}$ | Dilution factor applied to the sample |
            | $DF_{ref}$ | Dilution factor applied to the reference |
            | $CF$ | Empirical correction factor for validated systematic effects |

            ### Important notes

            - This implementation is intended as a transparent and editable validation workflow.
            - The pulse correction term is included as a practical placeholder for probe damping / pulse-length correction.
            - For strict qNMR validation, relaxation delay, receiver gain, pulse calibration, baseline correction, phase correction, and signal overlap must be experimentally controlled.
            - The external reference must be measured under comparable conditions and should have a stable, isolated signal.
            - The correction factor $CF$ should initially be kept as 1.0 and later replaced by an experimentally validated correction factor when needed.

            ### References

            1. Wider, G.; Dreier, L. Measuring Protein Concentrations by NMR Spectroscopy. *Journal of the American Chemical Society* **2006**, 128, 2571–2576. DOI: 10.1021/ja055336t.
            2. Monakhova, Y. B.; Kohl-Himmelseher, M.; Kuballa, T.; Lachenmeier, D. W. Determination of the purity of pharmaceutical reference materials by ¹H NMR using the standardless PULCON methodology. *Journal of Pharmaceutical and Biomedical Analysis* **2014**, 100, 381–386.
            3. Milbert, S.; Laumeyer, J. T.; Müller, T. M.; Krenz, O.; Fuchs, J.; Lehmann, L.; Seifert, S. T. Quantification of Ingredients and Additives of Beer and Beer-Based Mixed Beverages by ¹H NMR Spectroscopy and Automated PULCON Analysis. *Analytical Chemistry* **2026**.
            4. Malz, F.; Jancke, H. Validation of quantitative NMR. *Journal of Pharmaceutical and Biomedical Analysis* **2005**, 38, 813–823.
            """
        )

    if len(st.session_state.batch_spectra) == 0:
        st.warning("No batch spectra available. First process a batch in the Bruker Processing tab.")
    elif st.session_state.batch_reference_name is None:
        st.warning("No reference spectrum has been assigned. Go to Bruker Processing and assign one Reference role.")
    else:
        st.subheader("Run batch quantification")
        st.info(
            "Review Reference, Sample, and Integration settings before running. "
            "This button uses the current global settings and the processed batch spectra."
        )

        batch_reference = st.session_state.batch_reference_name
        available_samples = [x for x in st.session_state.batch_spectra.keys() if x != batch_reference]
        selected_samples_for_quant = st.multiselect(
            "Samples to quantify",
            available_samples,
            default=st.session_state.get("selected_batch_sample_labels", available_samples),
            help="Select which processed sample spectra should be quantified against the assigned reference."
        )

        run_batch_quant = st.button("Run batch quantification", type="primary", key="run_batch_quantification_from_quant_tab")

        if run_batch_quant:
            ref_region_safe = ensure_regions(st.session_state.get("ref_region", [(7.19, 7.39)]), default=(7.19, 7.39))
            sample_region_safe = ensure_regions(st.session_state.get("sample_region", [(3.90, 4.05)]), default=(3.90, 4.05))
            peak_count_safe = st.session_state.get("peak_count", 1)
            spectrometer_mhz_safe = st.session_state.get("spectrometer_mhz", 500.0)
            analyte_mw_safe = st.session_state.get("analyte_mw", 0.0)

            batch_results = run_batch_quantification(
                batch_spectra=st.session_state.batch_spectra,
                reference_label=batch_reference,
                sample_labels=selected_samples_for_quant,
                ref_region=ref_region_safe,
                sample_region=sample_region_safe,
                integration_method=st.session_state.integration_method,
                baseline_mode=st.session_state.baseline_mode,
                peak_count=peak_count_safe,
                spectrometer_mhz=spectrometer_mhz_safe,
                C_ref=st.session_state.C_ref,
                n_ref=st.session_state.n_ref,
                n_sample=st.session_state.n_sample,
                NS_ref=st.session_state.NS_ref,
                NS_sample=st.session_state.NS_sample,
                P1_ref=st.session_state.P1_ref,
                P1_sample=st.session_state.P1_sample,
                RG_ref=st.session_state.RG_ref,
                RG_sample=st.session_state.RG_sample,
                volume_ref=st.session_state.volume_ref,
                volume_sample=st.session_state.volume_sample,
                dilution_factor_ref=st.session_state.dilution_factor_ref,
                dilution_factor_sample=st.session_state.dilution_factor_sample,
                correction_factor=st.session_state.correction_factor,
                analyte_mw=analyte_mw_safe,
            )
            st.session_state.batch_results = batch_results
            st.session_state.quant_results = batch_results
            st.success("Batch quantification completed.")

        if st.session_state.batch_results is not None:
            st.subheader("Batch quantification results")

            with st.expander("Tutorial: how to interpret the batch quantification results table", expanded=False):
                st.markdown(
                    """
                    This table contains two types of rows:

                    ### 1. `region_result`
                    Each `region_result` row corresponds to one individual quantification result calculated from a specific reference/sample region pair.

                    Key columns:
                    - `sample_name`: sample being quantified.
                    - `reference_name`: reference spectrum used for calibration.
                    - `reference_region_index`: index of the reference integration region.
                    - `sample_region_index`: index of the sample integration region.
                    - `reference_region_ppm`: ppm interval used in the reference spectrum.
                    - `sample_region_ppm`: ppm interval used in the sample spectrum.
                    - `I_reference`: integrated area of the reference signal.
                    - `I_sample`: integrated area of the sample signal.
                    - `C_sample_mM`: concentration calculated from that region pair.
                    - `C_sample_mg_mL`: concentration in mg/mL, if analyte MW was provided.

                    ### 2. `sample_summary`
                    Each `sample_summary` row summarizes all valid region-level results for one sample.

                    Key columns:
                    - `C_sample_mM`: mean concentration across valid region results.
                    - `C_sample_std_mM`: standard deviation across region results.
                    - `C_sample_RSD_percent`: relative standard deviation, useful as an internal validation metric.
                    - `n_region_results`: number of valid region-pair calculations used in the summary.

                    ### Validation interpretation
                    Low RSD means that the selected regions agree well. High RSD suggests possible overlap, baseline distortion, poor phasing, relaxation differences, or incorrect proton assignment.

                    Suggested early interpretation:
                    - RSD < 5%: very good agreement.
                    - RSD 5–10%: acceptable, but inspect spectra.
                    - RSD > 10%: investigate selected regions and acquisition parameters.
                    """
                )

            st.dataframe(st.session_state.batch_results, use_container_width=True)
            st.download_button(
                "Download batch quantification CSV",
                data=st.session_state.batch_results.to_csv(index=False).encode("utf-8"),
                file_name="batch_qnmr_external_calibration_results.csv",
                mime="text/csv"
            )


# =========================================================
# Tab 7: Report
# =========================================================

with tabs[6]:
    st.header("Report")

    with st.expander("Tutorial: reporting", expanded=False):
        st.markdown(
            """
            The report table stores the calculation inputs and outputs used for the current quantification.
            Export this table together with the processed spectra to ensure traceability.
            """
        )

    if st.session_state.quant_results is not None:
        st.dataframe(st.session_state.quant_results)
        csv = st.session_state.quant_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download quantification report CSV",
            data=csv,
            file_name="qnmr_external_calibration_report.csv",
            mime="text/csv"
        )
    else:
        st.info("No report available yet. Run quantification first.")