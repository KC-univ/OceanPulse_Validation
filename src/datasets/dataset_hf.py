from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor         # noqa: F401 - reserved for future MICE pass
from sklearn.metrics import root_mean_squared_error        # noqa: F401 - reserved for imputation validation
from typing import Literal, Optional, List, Dict, Final
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 - required side-effect import

# Custom imports
from src.utils.hf_dataset_utils import create_distribution_plots, plot_data_loss_analysis

# ============================================================
#                 LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
#                 HELPERS
# ============================================================
def _is_direction_col(col_name: str) -> bool:
    """Return True if *col_name* looks like a circular angular signal (degrees)."""
    keywords = ["DIR", "dir", "direction", "Heading", "heading", "ANGLE", "angle"]
    return any(k in col_name for k in keywords)


def _save_plot(fig, path: str) -> None:
    """Tight-layout, save at 300 dpi, then close the figure to free memory."""
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ============================================================
#                 SIGNAL DEFINITIONS
# ============================================================
class SignalNames:
    """Central registry of column-name constants used across the pipeline.

    Having constants here avoids scattered magic strings and makes
    renaming a signal a single-line change.
    """

    # Identifiers
    IMO       = "IMO"
    TIMESTAMP = "TIMESTAMP"

    # Navigation
    HEADING      = "HEADING"
    HEADING_STD  = "HEADING_STD"
    RUDDER_ANGLE = "RUDDER_ANGLE"

    SOG      = "SPEED_OVER_GROUND"
    COG      = "COURSE_OVER_GROUND"
    STW      = "SPEED_THROUGH_WATER"
    STW_LONG = "SPEED_THROUGH_WATER_LONG"
    STW_TRANS= "SPEED_THROUGH_WATER_TRANS"
    VSTW     = "VIRTUAL_SPEED_THROUGH_WATER"

    DRIFT = "DRIFT_ANGLE"

    # Ship geometry (constants per vessel)
    LPP            = "LBP"
    BEAM           = "BEAM"
    PROPELLER_DIAM = "PROPELLER_DIAM"

    TRIM       = "TRIM_AT_PP"
    DRAFT_AFT  = "DRAFT_AFT"
    DRAFT_MID  = "DRAFT_MID"
    DRAFT_FORE = "DRAFT_FORE"
    DRAFT_MEAN = "DRAFT_MEAN"

    # Propulsion
    SHAFT_POWER      = "SHAFT_POWER"
    SHAFT_SPEED      = "SHAFT_SPEED"
    SHAFT_TORQUE     = "SHAFT_TORQUE"
    SHAFT_LOAD       = "SHAFT_LOAD"
    SHAFT_SPEED_PERC = "SHAFT_SPEED_PERC"

    # Fuel
    ME_FO_CONS   = "ME_FUEL_OIL_CONSUMPT_LCV_CORR"
    ME_GAS_CONS  = "ME_FUEL_GAS_CONSUMPT"
    SFOC         = "ME_SPECIFIC_FUEL_OIL_CONSUMPT_LCV_CORR"
    SGC          = "ME_SPECIFIC_FUEL_GAS_CONSUMPT"
    SFC          = "ME_SPECIFIC_FUEL_CONSUMPT_LCV_CORR"
    ME_FUEL_MODE = "ME_FUEL_MODE"

    # Environment - depth
    WATER_DEPTH         = "WATER_DEPTH"
    DEPTH_FROUDE_NUMBER = "DEPTH_FROUDE_NB"
    DEPTH_OVER_DRAFT    = "DEPTH_OVER_DRAFT_RATIO"

    # Wind
    TRUE_WIND_SPEED = "TRUE_WIND_SPEED_10M"
    TRUE_WIND_DIR   = "TRUE_WIND_DIR_10M"
    REL_WIND_SPEED  = "REL_WIND_SPEED_10M"
    REL_WIND_DIR    = "REL_WIND_DIR_10M"

    # Current
    TRUE_CURRENT_SPEED = "TRUE_CURRENT_SPEED"

    # Waves (primary)
    WAVE_DIRECTION   = "TOTAL_WAVE_MEAN_DIR"
    MEAN_WAVE_PERIOD = "TOTAL_WAVE_MEAN_PERIOD"
    HS_TOTAL         = "TOTAL_WAVE_SIGNIFICANT_HEIGHT"

    # Waves (decomposed)
    HS_WIND_WAVES            = "WIND_WAVES_SIGNIFICANT_HEIGHT"
    HS_SWELL                 = "TOTAL_SWELL_SIGNIFICANT_HEIGHT"
    WIND_WAVES_MEAN_PERIOD   = "WIND_WAVES_MEAN_PERIOD"
    WIND_WAVES_REL_MEAN_DIR  = "WIND_WAVES_REL_MEAN_DIR"
    TOTAL_SWELL_MEAN_PERIOD  = "TOTAL_SWELL_MEAN_PERIOD"
    TOTAL_SWELL_REL_MEAN_DIR = "TOTAL_SWELL_REL_MEAN_DIR"


def get_ocean_pulse_signals() -> List[str]:
    """Return the ordered list of signals required by the OceanPulse model."""
    return [
        # Vessel identifiers / geometry (static per voyage)
        SignalNames.IMO,
        SignalNames.LPP,
        SignalNames.BEAM,
        SignalNames.TIMESTAMP,
        SignalNames.PROPELLER_DIAM,

        # Speed
        #SignalNames.STW,
        SignalNames.STW_LONG,
        #SignalNames.STW_TRANS,

        # Draft
        SignalNames.DRAFT_MEAN,
        # SignalNames.DRAFT_MID,
        # SignalNames.DRAFT_FORE,
        # SignalNames.DRAFT_AFT,

        # Wind
        # SignalNames.HEADING,
        SignalNames.REL_WIND_SPEED,
        SignalNames.REL_WIND_DIR,
        # SignalNames.TRUE_WIND_SPEED,
        # SignalNames.TRUE_WIND_DIR,

        # Waves
        SignalNames.HS_TOTAL,
        SignalNames.MEAN_WAVE_PERIOD,
        SignalNames.WAVE_DIRECTION,

        # Propulsion
        SignalNames.SHAFT_POWER,
        # SignalNames.SHAFT_LOAD,
        SignalNames.SHAFT_SPEED,
        # SignalNames.SHAFT_TORQUE,
        # SignalNames.SHAFT_SPEED_PERC,
    ]


def abbreviate_signal_names(columns: List[str]) -> List[str]:
    """Return display-friendly short names for long column identifiers (used in plots)."""
    abbrev_map: Dict[str, str] = {
        "ME_FUEL_OIL_CONSUMPT_LCV_CORR":             "ME_FO_CONS",
        "ME_FUEL_GAS_CONSUMPT":                      "ME_GAS_CONS",
        "ME_SPECIFIC_FUEL_OIL_CONSUMPT_LCV_CORR":    "SFOC",
        "ME_SPECIFIC_FUEL_GAS_CONSUMPT":             "SGC",
        "ME_SPECIFIC_FUEL_CONSUMPT_LCV_CORR":        "SFC",
        "SPEED_THROUGH_WATER_LONG":                  "STW",
        "VIRTUAL_SPEED_THROUGH_WATER":               "VSTW",
        "SPEED_OVER_GROUND":                         "SOG",
        "COURSE_OVER_GROUND":                        "COG",
        "TRUE_WIND_SPEED_10M":                       "TWS_10M",
        "TRUE_WIND_DIR_10M":                         "TWD_10M",
        "REL_WIND_SPEED_10M":                        "RWS_10M",
        "REL_WIND_DIR_10M":                          "RWD_10M",
        "WIND_WAVES_SIGNIFICANT_HEIGHT":             "HS_WIND",
        "TOTAL_SWELL_SIGNIFICANT_HEIGHT":            "HS_SWELL",
        "TOTAL_WAVE_SIGNIFICANT_HEIGHT":             "HS_TOTAL",
        "WIND_WAVES_MEAN_PERIOD":                    "WW_PERIOD",
        "WIND_WAVES_REL_MEAN_DIR":                   "WW_DIR",
        "TOTAL_SWELL_MEAN_PERIOD":                   "SWELL_PERIOD",
        "TOTAL_SWELL_REL_MEAN_DIR":                  "SWELL_DIR",
        "TOTAL_WAVE_MEAN_PERIOD":                    "WAVE_PERIOD",
        "TOTAL_WAVE_MEAN_DIR":                       "WAVE_DIR",
        "TRUE_CURRENT_SPEED":                        "CURR_SPEED",
        "DEPTH_FROUDE_NB":                           "DEPTH_FN",
        "DEPTH_OVER_DRAFT_RATIO":                    "DEPTH/DRAFT",
    }
    return [abbrev_map.get(col, col) for col in columns]


# ============================================================
#                 FEATURE CORRELATION PLOT
# ============================================================
def plot_feature_correlation(df: pd.DataFrame, save_path: str, filename: str, imo: str,
                            method: Literal["pearson", "kendall", "spearman"] = "pearson",
                        ) -> None:
    """Generate and save a lower-triangle feature-correlation heatmap.

    Args:
        df:        DataFrame of numeric features (non-numeric columns are ignored).
        save_path: Directory where the PNG is written.
        filename:  Source CSV filename - used to build the output filename.
        imo:       Vessel IMO number for file labelling.
        method:    Correlation method passed to ``DataFrame.corr``.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    # Drop constant columns - they carry no correlation information.
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]

    if numeric_df.shape[1] < 2:
        logger.warning("Not enough numeric columns for correlation plot - skipping.")
        return

    corr_matrix = numeric_df.corr(method=method)

    # Shorten column labels so the heatmap stays readable.
    abbrev_cols = abbreviate_signal_names(list(corr_matrix.columns))
    corr_matrix.columns = abbrev_cols
    corr_matrix.index   = abbrev_cols  # type: ignore[assignment]

    num_vars = len(corr_matrix.columns)
    fig_size = max(16, num_vars * 0.8)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size))

    # Enforce annotating the values inside each square
    annot      = True
    annot_size = max(8, 14 - num_vars // 4)

    sns.heatmap(
        corr_matrix, ax=ax,
        square=True, cbar=True,
        cmap="BrBG", center=0, vmin=-1, vmax=1,
        annot=annot, fmt=".2f", annot_kws={"size": annot_size},
    )

    # Ensure the heatmap is fully scaled and not cut off on y-axis
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)

    ax.set_title(
        f"Feature Correlation Matrix ({method.capitalize()})",
        fontsize=16, fontweight="bold", pad=20,
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0,  fontsize=10)

    out_file = os.path.join(
        save_path, f"feature_corr_{imo}_{filename.replace('.csv', '')}.png"
    )
    _save_plot(fig, out_file)
    logger.info(f"Saved feature correlation heatmap -> {out_file}")


# ============================================================
#                 CFD CONDITIONS FILTER
# ============================================================
def apply_cfd_conditions(df: pd.DataFrame, cfd_conditions: Dict[str, float], ) -> pd.DataFrame:
    """Filter rows to those that satisfy CFD simulation validity conditions.

    Each condition is applied sequentially; missing columns are skipped with
    a warning rather than raising an exception.

    Supported keys in *cfd_conditions*:
        min_speed       - minimum STW (m/s)
        max_speed       - maximum STW (m/s)
        min_draft       - minimum draft (m)  - uses DRAFT_MEAN, falls back to DRAFT_AFT
        max_draft       - maximum draft (m)
        max_drift_angle - maximum |drift angle| (deg)
        max_rudder_angle- maximum |rudder angle| (deg)

    Returns:
        Filtered copy of *df*.
    """
    df_filtered   = df.copy()
    initial_total = len(df_filtered)
    logger.info(f"CFD filtering - initial rows: {initial_total:,}")

    def _log_step(label: str, before: int) -> None:
        removed = before - len(df_filtered)
        pct     = removed / before * 100 if before > 0 else 0.0
        logger.info(f"  [{label}] removed {removed:,} rows ({pct:.1f}%) "
                    f"-> remaining {len(df_filtered):,}"
                )

    # --- Speed ---------------------------------------------------------------
    if (v := cfd_conditions.get("min_speed")) is not None:
        if SignalNames.STW_LONG in df_filtered.columns:
            n = len(df_filtered)
            df_filtered = df_filtered[df_filtered[SignalNames.STW_LONG] >= v]
            _log_step(f"min_speed >= {v:.2f} m/s", n)
        else:
            logger.warning(f"CFD [min_speed]: column {SignalNames.STW_LONG} not found - skipped")

    if (v := cfd_conditions.get("max_speed")) is not None:
        if SignalNames.STW_LONG in df_filtered.columns:
            n = len(df_filtered)
            df_filtered = df_filtered[df_filtered[SignalNames.STW_LONG] <= v]
            _log_step(f"max_speed <= {v:.2f} m/s", n)
        else:
            logger.warning(f"CFD [max_speed]: column {SignalNames.STW_LONG} not found - skipped")

    # --- Draft (prefer DRAFT_MEAN, fall back to DRAFT_AFT) -------------------
    draft_col: Optional[str] = None
    if SignalNames.DRAFT_MEAN in df_filtered.columns:
        draft_col = SignalNames.DRAFT_MEAN
    elif SignalNames.DRAFT_AFT in df_filtered.columns:
        draft_col = SignalNames.DRAFT_AFT
        logger.info("Draft filtering: using DRAFT_AFT (DRAFT_MEAN not present)")
    else:
        logger.warning("CFD draft filtering: no draft column found - skipped")

    if (v := cfd_conditions.get("min_draft")) is not None and draft_col is not None:
        n = len(df_filtered)
        df_filtered = df_filtered[df_filtered[draft_col] >= v]
        _log_step(f"min_draft >= {v:.2f} m ({draft_col})", n)

    if (v := cfd_conditions.get("max_draft")) is not None and draft_col is not None:
        n = len(df_filtered)
        df_filtered = df_filtered[df_filtered[draft_col] <= v]
        _log_step(f"max_draft <= {v:.2f} m ({draft_col})", n)

    # --- Angular conditions (absolute value) ---------------------------------
    if (v := cfd_conditions.get("max_drift_angle")) is not None:
        if SignalNames.DRIFT in df_filtered.columns:
            n = len(df_filtered)
            df_filtered = df_filtered[df_filtered[SignalNames.DRIFT].abs() <= v]
            _log_step(f"|drift| <= {v:.2f}deg", n)
        else:
            logger.warning("CFD [max_drift_angle]: column DRIFT not found - skipped")

    if (v := cfd_conditions.get("max_rudder_angle")) is not None:
        if SignalNames.RUDDER_ANGLE in df_filtered.columns:
            n = len(df_filtered)
            df_filtered = df_filtered[df_filtered[SignalNames.RUDDER_ANGLE].abs() <= v]
            _log_step(f"|rudder| <= {v:.2f}deg", n)
        else:
            logger.warning("CFD [max_rudder_angle]: column RUDDER_ANGLE not found - skipped")

    # --- Summary -------------------------------------------------------------
    total_removed = initial_total - len(df_filtered)
    pct = total_removed / initial_total * 100 if initial_total > 0 else 0.0
    logger.info(f"CFD filter complete - removed {total_removed:,} / {initial_total:,} rows "
                f"({pct:.1f}%) - "
                f"final: {len(df_filtered):,} rows"
            )
    return df_filtered


# ============================================================
#                 DATA DESCRIPTION REPORT
# ============================================================
def save_data_description(data: pd.DataFrame, df: pd.DataFrame, output_path: str, ) -> None:
    """Write a plain-text quality report comparing *data* (raw) and *df* (cleaned).

    Args:
        data:        DataFrame **before** any cleaning / filtering.
        df:          DataFrame **after** cleaning / filtering.
        output_path: Destination .txt file path.
    """
    from io import StringIO
    
    total_columns      = len(data.columns)
    columns_with_data  = data.notna().any().sum()
    usability_pct      = columns_with_data / total_columns * 100 if total_columns > 0 else 0.0
    total_rows         = len(data)
    rows_with_data     = data.notna().any(axis=1).sum()
    is_usable          = not data.empty and total_columns > 0 and data.notna().any().any()
    
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("DATA QUALITY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("USABILITY SUMMARY\n" + "-" * 80 + "\n")
        f.write(f"Status:              {'USABLE' if is_usable else 'NOT USABLE'}\n")
        f.write(f"Total columns:       {total_columns}\n")
        f.write(f"Columns with data:   {columns_with_data} ({usability_pct:.1f}%)\n")
        f.write(f"Total rows:          {total_rows}\n")
        f.write(f"Rows with data:      {rows_with_data}\n")
        
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("STATISTICAL SUMMARY\n" + "=" * 80 + "\n\n")
        f.write(data.describe().to_string())
        
        for title, frame in [
            ("DATASET INFORMATION BEFORE CLEANING", data),
            ("DATASET INFORMATION AFTER CLEANING",  df),
        ]:
            f.write("\n\n" + "=" * 80 + "\n" + title + "\n" + "=" * 80 + "\n\n")
            buf = StringIO()
            frame.info(buf=buf)
            f.write(buf.getvalue())
            
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("MISSING DATA ANALYSIS (before cleaning)\n" + "=" * 80 + "\n\n")
        missing = data.isnull().sum()
        for col in missing[missing > 0].index:
            pct = missing[col] / len(data) * 100
            f.write(f"  {col}: {missing[col]:,} ({pct:.1f}%)\n")
            
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("CONCLUSION\n" + "=" * 80 + "\n\n")
        if is_usable:
            f.write("The dataframe is USABLE for analysis.\n")
            f.write(f"  - {usability_pct:.1f}% of columns contain data\n")
            f.write(f"  - {rows_with_data:,} / {total_rows:,} rows contain data\n")
        else:
            f.write("The dataframe is NOT USABLE for analysis.\n")
            if data.empty:
                f.write("  - Reason: DataFrame is empty\n")
            elif total_columns == 0:
                f.write("  - Reason: No columns present\n")
            else:
                f.write("  - Reason: All values are null\n")
                
    logger.info(f"Data quality report saved -> {output_path}")

# ============================================================
#                 DOMAIN-AWARE IMPUTATION
# ============================================================
def data_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using first-principles physical relationships.

    Only fills gaps that can be *reconstructed* from known signals -
    no statistical guessing.  Each imputation block logs the column
    name and the number of rows it filled.

    Imputation rules
    ----------------
    1. **STW** - from longitudinal + transverse speed components.
    2. **Wave period (Tp)** - from ITTC approximation Tp ≈ alpha.sqrt(Hs) when Hs > 0.
        Calm sea (Hs = 0) forces Tp = 0 and direction = 0.
    3. **Wave direction** - angular-safe unwrap -> interpolate -> mod 2pi.
    4. **Shaft power** - in order of preference:
            a. torque * angular speed
            b. shaft load fraction * rated power
            c. cubic RPM-percentage law
    5. **Relative wind** - exact kinematic vector subtraction of ship
        velocity from true-wind vector, rotated into ship frame.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame.  Modified columns are returned in a copy.

    Returns
    -------
    pd.DataFrame
        Frame with imputed values.  Original rows are never dropped.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Physical constants (specific to 23 k-TEU ULCS)
    # ------------------------------------------------------------------
    P_RATED_W: Final[float] = 80e6  # rated shaft power  [W]
    ALPHA_TP:  Final[float] = 4.0   # ITTC Tp = Tp ≈ alpha.sqrt(Hs) coefficient
    
    def _log_fill(col: str, n_filled: int) -> None:
        """Uniform log line so every imputation block is easy to grep."""
        if n_filled > 0:
            logger.info(f"[impute] {col}: filled {n_filled:,} rows")
        else:
            logger.debug(f"[impute] {col}: nothing to fill")
            
    # ------------------------------------------------------------------
    # 1. Speed through water - magnitude from components
    # ------------------------------------------------------------------
    stw_col   = "SPEED_THROUGH_WATER"
    stw_long  = "SPEED_THROUGH_WATER_LONG"
    stw_trans = "SPEED_THROUGH_WATER_TRANS"
    
    if stw_col in df.columns:
        mask = df[stw_col].isna()
        if mask.any() and {stw_long, stw_trans} <= set(df.columns):
            fill_mask = mask & df[stw_long].notna() & df[stw_trans].notna()
            df.loc[fill_mask, stw_col] = np.sqrt(
                df.loc[fill_mask, stw_long] ** 2
                + df.loc[fill_mask, stw_trans] ** 2
            )
            _log_fill(stw_col, int(fill_mask.sum()))
            
    # ------------------------------------------------------------------
    # 2. Wave parameters
    # ------------------------------------------------------------------
    hs_col   = "TOTAL_WAVE_SIGNIFICANT_HEIGHT"
    tp_col   = "TOTAL_WAVE_MEAN_PERIOD"
    dir_col  = "TOTAL_WAVE_REL_MEAN_DIR"
    
    if {hs_col, tp_col, dir_col} <= set(df.columns):
        Hs   = df[hs_col]
        Tp   = df[tp_col].copy()
        beta = df[dir_col].copy()
        
        # Calm sea: period and direction are physically zero.
        calm = Hs == 0.0
        n_calm_tp   = int((calm & Tp.notna() & (Tp != 0)).sum())
        n_calm_dir  = int((calm & beta.notna() & (beta != 0)).sum())
        Tp.loc[calm]   = 0.0
        beta.loc[calm] = 0.0
        if n_calm_tp or n_calm_dir:
            logger.info(
                f"  [impute] calm-sea zero-fill -> "
                f"{tp_col}: {n_calm_tp:,} rows, {dir_col}: {n_calm_dir:,} rows"
            )
            
        # ITTC approximation: Tp ≈ Tp ≈ alpha.sqrt(Hs) for non-calm missing periods.
        mask_tp = Hs.notna() & Tp.isna() & (Hs > 0)
        Tp.loc[mask_tp] = ALPHA_TP * np.sqrt(Hs.loc[mask_tp])
        _log_fill(tp_col, int(mask_tp.sum()))
        
        # Angular-safe interpolation for wave direction.
        n_dir_before = int(beta.isna().sum())
        if n_dir_before > 0:
            beta_unwrapped = np.unwrap(beta.to_numpy())
            beta_interp = (
                pd.Series(beta_unwrapped, index=df.index)
                .interpolate(method="index", limit_direction="both")
            )
            beta = np.mod(beta_interp, 2 * np.pi)
            n_dir_filled = n_dir_before - int(pd.Series(beta).isna().sum())
            _log_fill(dir_col, n_dir_filled)
            
        df[tp_col]  = Tp
        df[dir_col] = beta
        
    # ------------------------------------------------------------------
    # 3. Shaft power - three fallback sources in order of accuracy
    # ------------------------------------------------------------------
    pwr_col    = "SHAFT_POWER"
    torque_col = "SHAFT_TORQUE"
    spd_col    = "SHAFT_SPEED"
    load_col   = "SHAFT_LOAD"
    perc_col   = "SHAFT_SPEED_PERC"
    
    if pwr_col in df.columns:
        P = df[pwr_col].copy()
        
        # (a) P = torque * angular-speed  [most accurate]
        if {torque_col, spd_col} <= set(df.columns):
            mask = P.isna() & df[torque_col].notna() & df[spd_col].notna()
            P.loc[mask] = df.loc[mask, torque_col] * df.loc[mask, spd_col]
            _log_fill(f"{pwr_col} via torque*speed", int(mask.sum()))
            
        # (b) P = (load% / 100) * P_rated
        if load_col in df.columns:
            mask = P.isna() & df[load_col].notna()
            P.loc[mask] = (df.loc[mask, load_col] / 100.0) * P_RATED_W
            _log_fill(f"{pwr_col} via shaft-load", int(mask.sum()))
            
        # (c) P = P_rated * (RPM% / 100)^3  [cubic law approximation]
        if perc_col in df.columns:
            mask = P.isna() & df[perc_col].notna()
            P.loc[mask] = P_RATED_W * (df.loc[mask, perc_col] / 100.0) ** 3
            _log_fill(f"{pwr_col} via RPM%-cubic-law", int(mask.sum()))
            
        df[pwr_col] = P
        
    # ------------------------------------------------------------------
    # 4. Relative wind - exact kinematic vector decomposition
    # ------------------------------------------------------------------
    tws_col = "TRUE_WIND_SPEED_10M"
    twd_col = "TRUE_WIND_DIR_10M"
    rws_col = "REL_WIND_SPEED_10M"
    rwd_col = "REL_WIND_DIR_10M"
    hdg_col = "HEADING"
    
    required_wind = {tws_col, twd_col, stw_col, hdg_col}
    if required_wind <= set(df.columns):
        mask = (
            df[tws_col].notna()
            & df[twd_col].notna()
            & df[stw_col].notna()
            & df[hdg_col].notna()
        )
        if mask.any():
            Uw      = df.loc[mask, tws_col]
            theta_w = df.loc[mask, twd_col]
            Vs      = df.loc[mask, stw_col]
            psi     = df.loc[mask, hdg_col]
            
            # True-wind components (from North convention, into ship frame)
            Vwx = -Uw * np.sin(theta_w)
            Vwy = -Uw * np.cos(theta_w)
            
            # Ship velocity components
            Vsx = Vs * np.sin(psi)
            Vsy = Vs * np.cos(psi)
            
            # Relative wind in earth frame -> rotated into ship frame
            Vrx = Vwx - Vsx
            Vry = Vwy - Vsy
            
            df.loc[mask, rws_col] = np.sqrt(Vrx ** 2 + Vry ** 2)
            theta_rel = np.arctan2(-Vrx, -Vry)
            df.loc[mask, rwd_col] = (theta_rel - psi) % (2 * np.pi)
            
            _log_fill(rws_col, int(mask.sum()))
            _log_fill(rwd_col, int(mask.sum()))
            
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_remaining_nan = int(df.isna().sum().sum())
    logger.info(f"data_imputation complete - "
                f"remaining NaN values across all columns: {total_remaining_nan:,}")
    return df
# ============================================================
# 
# ============================================================
def detect_cruising_state(df,
                        speed_col="SPEED_THROUGH_WATER",
                        heading_col="HEADING",
                        draft_col="DRAFT_MEAN",
                        window="30min"
                    ):
    """
    Returns a boolean Series indicating cruising state.
    """
    if speed_col not in df.columns:
        return pd.Series(False, index=df.index)

    # Rolling statistics
    speed_mean = df[speed_col].rolling(window, min_periods=3).mean()
    speed_std  = df[speed_col].rolling(window, min_periods=3).std()

    cruising = (speed_mean > 3.0) & (speed_std < 0.5)   # >6 kn ≈ underway, not maneuvering & low speed fluctuation

    if heading_col and heading_col in df.columns:
        heading_std = df[heading_col].rolling(window, min_periods=3).std()
        cruising = cruising & (heading_std < 5.0)       # no significant turning

    if draft_col and draft_col in df.columns:
        draft_rate  = df[draft_col].diff().abs() / (10 / 60)  # m/hour (10-min data)
        cruising = cruising & (draft_rate < 0.05)       # <5 cm/hour → no ops

    return cruising.fillna(False).astype(bool)

# ============================================================
#                 CORE EXTRACTION FUNCTION
# ============================================================
def hf_vars_extractor(high_frequency_folder: str,
                    analysis_save_path: str,
                    preprocessed_save_path: str,
                    perform_imputation: bool = True,
                    speed_filter_threshold: Optional[float] = 1.0,
                    power_filter_threshold: Optional[float] = 500.0,
                    cfd_conditions: Optional[Dict[str, float]] = None,
                    return_df: str = "final",
                ) -> bool:
    """Extract, clean, (optionally impute) and save high-frequency vessel data.

    Processing pipeline
    -------------------
    1. Read all CSVs in *high_frequency_folder*.
    2. Select OceanPulse signals that are actually present.
    3. Parse + sort by TIMESTAMP; set as DatetimeIndex.
    4. Coerce object columns to numeric (except IMO).
    5. Plot feature-correlation heatmap for diagnostics.
    6. Either:
        - ``perform_imputation=True``  -> forward-fill short gaps, time-interpolate,
            then ``data_imputation`` (physics-based).
        - ``perform_imputation=False`` -> sequential speed / power / CFD filters
            followed by ``dropna``.
    7. Save quality report (.txt), feather file, and distribution plots.

    Args:
        high_frequency_folder:  Folder containing raw CSV files.
        analysis_save_path:     Destination for diagnostic plots and reports.
        preprocessed_save_path: Destination for preprocessed feather files.
        perform_imputation:     Whether to run physics-based imputation (True)
                                or hard-filter + dropna (False).
        speed_filter_threshold: Minimum STW [m/s] applied before power filter.
        power_filter_threshold: Minimum shaft power [W] applied after speed filter.
        cfd_conditions:         Optional CFD condition dict (see
                                ``apply_cfd_conditions`` for keys).
        return_df:              Which filtering stage to persist when
                                ``perform_imputation=False``.
                                One of: ``"original"``, ``"speed"``, ``"power"``,
                                ``"cfd"``, ``"final"``.
    Returns:
        True  if at least one CSV was successfully processed.
        False if no files were processed (bad folder, no CSVs, parse failures).
    """
    try:
        os.makedirs(analysis_save_path,    exist_ok=True)
        os.makedirs(preprocessed_save_path, exist_ok=True)
    except Exception:
        logger.exception("Failed to create output directories.")
        return False
    
    ocean_pulse_signals = get_ocean_pulse_signals()
    processed_files     = 0
    
    for filename in os.listdir(high_frequency_folder):
        if not filename.lower().endswith(".csv"):
            continue
        
        file_path = os.path.join(high_frequency_folder, filename)
        
        # ------------------------------------------------------------------
        # Load CSV - try common encodings in order of likelihood
        # ------------------------------------------------------------------
        try:
            logger.info(f"Loading: {filename}")
            data: Optional[pd.DataFrame] = None
            for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
                try:
                    data = pd.read_csv(file_path, low_memory=False, encoding=enc)
                    if enc != "utf-8":
                        logger.info(f"  Loaded with encoding: {enc}")
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
                
            if data is None:
                raise ValueError(f"Could not decode {filename} with any supported encoding")
            
            logger.info(f"  Rows loaded: {len(data):,}")
            
        except Exception:
            logger.exception(f"Failed to load {filename} - skipping")
            continue
        
        # ------------------------------------------------------------------
        # Validate required columns
        # ------------------------------------------------------------------
        required_cols = ["TIMESTAMP", "IMO", "LBP", "BEAM"]
        missing_req   = [c for c in required_cols if c not in data.columns]
        if missing_req:
            logger.warning(f"{filename}: missing required columns {missing_req} - skipping")
            continue
        
        # ------------------------------------------------------------------
        # Restrict to OceanPulse signals present in this file
        # ------------------------------------------------------------------
        available_signals = [s for s in ocean_pulse_signals if s in data.columns]
        for r in required_cols:
            if r in data.columns and r not in available_signals:
                available_signals.insert(0, r)
                
        if not available_signals:
            logger.warning(f"{filename}: no OceanPulse signals found - skipping")
            continue
        
        # Guard against duplicate column names (can happen in malformed CSVs).
        data_filtered = (data[available_signals].loc[:, ~data[available_signals].columns.duplicated()].copy())
        
        # ------------------------------------------------------------------
        # Parse TIMESTAMP, sort, set as index
        # ------------------------------------------------------------------
        try:
            data_filtered["TIMESTAMP"] = pd.to_datetime(data_filtered["TIMESTAMP"], errors="coerce")
            bad_ts = data_filtered["TIMESTAMP"].isna().sum()
            if bad_ts > 0.5 * len(data_filtered):
                logger.warning(f"  {bad_ts:,} TIMESTAMP values failed to parse - check format")
                
            data_filtered = (data_filtered.sort_values("TIMESTAMP").reset_index(drop=True).set_index("TIMESTAMP"))
            
        except Exception:
            logger.exception("Failed to parse TIMESTAMP - continuing without time index")
            
        df = data_filtered.copy()
        
        # Coerce object columns to numeric so we can compute statistics.
        # IMO is intentionally kept as a string identifier.
        for col in df.columns:
            if col != "IMO" and df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # ------------------------------------------------------------------
        # Extract IMO for file naming (use first non-null value)
        # ------------------------------------------------------------------
        imo_series = df[SignalNames.IMO].dropna()
        imo        = str(imo_series.iloc[0]) if not imo_series.empty else "unknownIMO"
        
        # ------------------------------------------------------------------
        # Diagnostics - correlation heatmap
        # ------------------------------------------------------------------
        try:
            plot_feature_correlation(df[numeric_cols] if numeric_cols else df,analysis_save_path, filename, imo,)
            
        except Exception:
            logger.exception("Feature correlation heatmap failed - continuing")
            
        # ==================================================================
        # BRANCH A: Physics-based imputation
        # ==================================================================
        INTERP_SAFE = [ SignalNames.STW_LONG,
                        SignalNames.DRAFT_MEAN,
                        SignalNames.REL_WIND_SPEED,
                        SignalNames.REL_WIND_DIR,
                        SignalNames.HS_TOTAL,
                        SignalNames.MEAN_WAVE_PERIOD,
                        SignalNames.WAVE_DIRECTION,
                    ]
        NEVER_INTERP = [SignalNames.SHAFT_POWER,SignalNames.SHAFT_SPEED,]
        
        if perform_imputation:
            logger.info("Imputation branch: physics-based (ISO 19030 compliant)")
            df_impute = df.copy()

            # ------------------------------------------------------------
            # Step 0: cruising-state detection (explicit, auditable)
            # ------------------------------------------------------------
            try:
                df_impute["is_cruising"] = detect_cruising_state(df_impute,
                                                                speed_col=SignalNames.STW_LONG,
                                                                heading_col=str(),                     # not available
                                                                draft_col=SignalNames.DRAFT_MEAN,
                                                            )
                logger.info(
                    f"Cruising-state detection complete: "
                    f"{df_impute['is_cruising'].sum():,} / {len(df_impute):,} rows flagged"
                )
            except Exception:
                logger.exception("Cruising-state detection failed")
                df_impute["is_cruising"] = False

            # Track imputation provenance (critical for PhD defense)
            df_impute["_was_imputed"] = False

            # ------------------------------------------------------------
            # Step 1: short-gap fill (<= 3 × 10 min = 30 min)
            # ONLY on cruising + safe variables
            # ------------------------------------------------------------
            safe_cols = []
            try:
                max_short_gap = 3
                safe_cols = [c for c in INTERP_SAFE if c in df_impute.columns]
                mask = df_impute["is_cruising"]

                before_na = df_impute.loc[mask, safe_cols].isna()

                df_impute.loc[mask, safe_cols] = (
                    df_impute.loc[mask, safe_cols]
                    .ffill(limit=max_short_gap)
                    .bfill(limit=max_short_gap)
                )

                after_na = df_impute.loc[mask, safe_cols].isna()
                df_impute.loc[mask, "_was_imputed"] |= (before_na & ~after_na).any(axis=1)

                logger.info(f"Short-gap fill applied (±{max_short_gap} steps, cruising-only)")
            except Exception:
                logger.exception("Short-gap fill failed")

            # ------------------------------------------------------------
            # Step 2: time-aware interpolation (<= 6 × 10 min = 60 min)
            # cruising-only, physics-safe variables only
            # ------------------------------------------------------------
            try:
                if isinstance(df_impute.index, pd.DatetimeIndex):
                    max_interp_gap = 6
                    mask = df_impute["is_cruising"]
                    logger.info(f"Cruising rows: {df_impute['is_cruising'].mean()*100:.2f}%")
                    
                    before_na = df_impute.loc[mask, safe_cols].isna()
                    
                    df_impute.loc[mask, safe_cols] = (
                        df_impute.loc[mask, safe_cols]
                        .interpolate(
                            method="time",
                            limit=max_interp_gap,
                            limit_direction="both"
                        )
                    )

                    after_na = df_impute.loc[mask, safe_cols].isna()
                    df_impute.loc[mask, "_was_imputed"] |= (before_na & ~after_na).any(axis=1)
                    
                    logger.info(
                        f"Time interpolation applied "
                        f"(limit={max_interp_gap} steps, cruising-only)"
                    )
            except Exception:
                logger.exception("Time interpolation failed")

            # ------------------------------------------------------------
            # Step 3: physics-based imputation (model-consistent)
            # ------------------------------------------------------------
            try:
                df_impute_before = df_impute.copy()
                df_impute = data_imputation(df_impute)

                # Mark newly imputed rows
                newly_filled = (
                    df_impute_before.isna() &
                    ~df_impute.isna()
                ).any(axis=1)
                df_impute["_was_imputed"] |= newly_filled

                logger.info("Physics-based imputation applied")
            except Exception:
                logger.exception(
                    "data_imputation failed — proceeding with earlier stages only"
                )

            # ------------------------------------------------------------
            # Restore static vessel geometry (safety net)
            # ------------------------------------------------------------
            for static_col in (
                SignalNames.IMO,
                SignalNames.LPP,
                SignalNames.BEAM,
                SignalNames.PROPELLER_DIAM,
            ):
                if static_col in df.columns and static_col not in df_impute.columns:
                    df_impute[static_col] = df[static_col]

            # ------------------------------------------------------------
            # Column selection (OceanPulse scope)
            # ------------------------------------------------------------
            cols_to_keep = [
                c for c in ocean_pulse_signals
                if c in df_impute.columns
            ] + ["_was_imputed"]

            df_impute = df_impute[cols_to_keep]

            # ------------------------------------------------------------
            # Handle remaining NaNs explicitly (no silent corruption)
            # ------------------------------------------------------------
            nan_count = df_impute.isna().sum().sum()
            if nan_count > 0:
                logger.warning(
                    f"Imputation left {nan_count:,} NaNs — "
                    f"dropping rows with incomplete core signals"
                )
                df_impute = df_impute.dropna()

            logger.info(
                f"Imputation complete — retained {len(df_impute):,} rows | "
                f"Imputed rows: {df_impute['_was_imputed'].sum():,} "
                f"({df_impute['_was_imputed'].mean() * 100:.1f}%)"
            )

            # ------------------------------------------------------------
            # Continue with your existing filtering & plotting logic
            # ------------------------------------------------------------
            df_impute = df_impute.drop(columns="_was_imputed")
            logger.info(f"After dropna(): {len(df_impute):,} rows remaining")
            
            # ----------------------------------------------------------------
            # Apply the SAME 4 filter stages on the imputed data for the plot
            # ----------------------------------------------------------------
            df_original = df.copy()          # ← TRUE raw original (before imputation)
            df_imputed  = df_impute.copy()   # ← post-imputation frame (starting point for filters)
            original_count = len(df_original)
            
            # Speed filter
            df_after_speed = df_imputed.copy()   # ← filters run on imputed data
            if speed_filter_threshold is not None and SignalNames.STW_LONG in df_after_speed.columns:
                df_after_speed = df_after_speed[df_after_speed[SignalNames.STW_LONG] >= speed_filter_threshold]
                n_dropped = len(df_imputed) - len(df_after_speed)
                logger.info(f"Speed filter (STW_LONG >= {speed_filter_threshold} m/s): "
                            f"dropped {n_dropped:,} rows -> {len(df_after_speed):,} remaining"
                        )
            else:
                logger.warning("Speed filter: skipped")
                
            # Power filter
            df_after_power = df_after_speed.copy()
            if power_filter_threshold is not None and SignalNames.SHAFT_POWER in df_after_power.columns:
                df_after_power = df_after_power[
                    df_after_power[SignalNames.SHAFT_POWER] >= power_filter_threshold
                ]
                n_dropped = len(df_after_speed) - len(df_after_power)
                logger.info(
                    f"  Power filter (SHAFT_POWER >= {power_filter_threshold} W): "
                    f"dropped {n_dropped:,} rows -> {len(df_after_power):,} remaining"
                )
            else:
                logger.info("  Power filter: skipped")
                
            # CFD conditions
            df_after_cfd = df_after_power.copy()
            if cfd_conditions is not None:
                df_after_cfd = apply_cfd_conditions(df_after_cfd, cfd_conditions)
                n_dropped = len(df_after_power) - len(df_after_cfd)
                logger.info(
                    f"  CFD filter: dropped {n_dropped:,} rows -> "
                    f"{len(df_after_cfd):,} remaining"
                )
            else:
                logger.info("  CFD filter: skipped (no conditions provided)")

            # Final dropna
            df_final   = df_after_cfd.dropna()
            n_dropped  = len(df_after_cfd) - len(df_final)
            total_drop = original_count - len(df_final)
            logger.info(f"dropna: dropped {n_dropped:,} rows -> {len(df_final):,} remaining\n"
                        f"--- Filter summary (post-imputation) ---\n"
                        f"Original: {original_count:,} | Final: {len(df_final):,} | "
                        f"Total dropped: {total_drop:,} ({total_drop / original_count * 100:.1f}%) | "
                        f"Retention: {len(df_final) / original_count * 100:.1f}%"
                    )

            logger.info("Start plotting the data loss analysis")
            plot_data_loss_analysis(
                df_original    = df_original,       # raw original for the plot baseline
                df_after_speed = df_after_speed,
                df_after_power = df_after_power,
                df_after_cfd   = df_after_cfd,
                df_final       = df_final,
                speed_threshold = speed_filter_threshold,
                power_threshold = power_filter_threshold if SignalNames.SHAFT_POWER in df.columns else None,
                cfd_conditions  = cfd_conditions,
                save_path       = analysis_save_path,
                imo             = str(imo),
            )
            
            df_impute = df_final
        # ==================================================================
        # BRANCH B: Hard filters + dropna
        # ==================================================================
        else:
            logger.info("Imputation branch: filter-only (no imputation)")
            
            df_original = df.copy()
            original_count = len(df_original)
            
            # Speed filter
            df_after_speed = df_original.copy()
            if speed_filter_threshold is not None and SignalNames.STW_LONG in df_after_speed.columns:
                df_after_speed = df_after_speed[
                    df_after_speed[SignalNames.STW_LONG] >= speed_filter_threshold
                ]
                n_dropped = original_count - len(df_after_speed)
                logger.info(f"Speed filter (STW >= {speed_filter_threshold} m/s): "
                            f"dropped {n_dropped:,} rows -> {len(df_after_speed):,} remaining"
                        )
            else:
                logger.warning("Speed filter: skipped")
                
            # Power filter
            df_after_power = df_after_speed.copy()
            if power_filter_threshold is not None and SignalNames.SHAFT_POWER in df_after_power.columns:
                df_after_power = df_after_power[
                    df_after_power[SignalNames.SHAFT_POWER] >= power_filter_threshold
                ]
                n_dropped = len(df_after_speed) - len(df_after_power)
                logger.info(
                    f"  Power filter (SHAFT_POWER >= {power_filter_threshold} W): "
                    f"dropped {n_dropped:,} rows -> {len(df_after_power):,} remaining"
                )
            else:
                logger.info("  Power filter: skipped")
                
            # CFD conditions
            df_after_cfd = df_after_power.copy()
            if cfd_conditions is not None:
                df_after_cfd = apply_cfd_conditions(df_after_cfd, cfd_conditions)
                n_dropped = len(df_after_power) - len(df_after_cfd)
                logger.info(
                    f"  CFD filter: dropped {n_dropped:,} rows -> "
                    f"{len(df_after_cfd):,} remaining"
                )
            else:
                logger.info("  CFD filter: skipped (no conditions provided)")
                
            # Final dropna
            df_final   = df_after_cfd.dropna()
            n_dropped  = len(df_after_cfd) - len(df_final)
            total_drop = original_count - len(df_final)
            logger.info(f"dropna: dropped {n_dropped:,} rows -> {len(df_final):,} remaining\n"
                        f"--- Filter summary ---\n"
                        f"Original: {original_count:,} | Final: {len(df_final):,} | "
                        f"Total dropped: {total_drop:,} ({total_drop / original_count * 100:.1f}%) | "
                        f"Retention: {len(df_final) / original_count * 100:.1f}%"
                    )
            logger.info("Start plotting the data loss analysis")
            plot_data_loss_analysis(df_original=df_original,
                                    df_after_speed=df_after_speed,
                                    df_after_power=df_after_power,
                                    df_after_cfd=df_after_cfd,
                                    df_final=df_final,
                                    speed_threshold=speed_filter_threshold,
                                    power_threshold=power_filter_threshold if SignalNames.SHAFT_POWER in df.columns else None,
                                    cfd_conditions=cfd_conditions,
                                    save_path=analysis_save_path,
                                    imo=str(imo),
                                )
            
        # Select which filtered stage to persist
        valid_stages: Dict[str, pd.DataFrame] = {"original": df_original,
                                                "speed":    df_after_speed,
                                                "power":    df_after_power,
                                                "cfd":      df_after_cfd,
                                                "final":    df_final,
                                            }
        stage = return_df.lower() if return_df else "final"
        if stage not in valid_stages:
            logger.warning( f"return_df='{stage}' is not a valid option "
                            f"({list(valid_stages)}); defaulting to 'final'"
                        )
            stage = "final"
            
        df_impute = valid_stages[stage]
        nan_count = df_impute.isna().sum().sum()
        
        # Always deliver a NaN-free frame to downstream consumers.
        if nan_count > 0:
            logger.warning( f"Stage '{stage}' contains {nan_count:,} NaN values - "
                            f"applying dropna() before persisting"
                        )
            df_impute = df_impute.dropna()
            
        logger.info(f"Selected stage '{stage}': {len(df_impute):,} rows, "
                    f"{df_impute.isna().sum().sum()} NaN values"
                )
            
        # ==================================================================
        # SAVE OUTPUTS
        # ==================================================================
        try:
            current_imo = imo
            if current_imo == "unknownIMO" and SignalNames.IMO in df_impute.columns:
                fallback = df_impute[SignalNames.IMO].dropna()
                if not fallback.empty:
                    current_imo = str(fallback.iloc[0])
            if current_imo == "unknownIMO":
                current_imo = filename.split(".")[0]
                
            # Quality report
            description_path = os.path.join(analysis_save_path, f"description_{current_imo}.txt")
            save_data_description(data=data_filtered, df=df_impute, output_path=description_path,)
            
            # Feather file (reset DatetimeIndex -> plain integer index for Feather)
            out_df = df_impute.copy()
            if isinstance(out_df.index, pd.DatetimeIndex):
                out_df = out_df.reset_index()
            feather_path = os.path.join(preprocessed_save_path, f"preprocessed_{current_imo}.feather")
            out_df.reset_index(drop=True).to_feather(feather_path)
            logger.info(f"Feather saved -> {feather_path}")
            
            # Distribution plots - filtered data
            create_distribution_plots(high_freq_df=df_impute,
                                    plots_per_row=3,
                                    save_path=analysis_save_path,
                                    filename=f"numeric_distributions_{current_imo}",
                                )
            
            # Distribution plots - raw data (for visual comparison)
            create_distribution_plots(high_freq_df=data_filtered,
                                    plots_per_row=3,
                                    save_path=analysis_save_path,
                                    filename=f"numeric_distributions_ORIGINAL_{current_imo}",
                                )
            
            processed_files += 1
            logger.info(f"Successfully processed: {filename}")
            
        except Exception:
            logger.exception(f"Failed to save outputs for {filename}")
            continue
        
    if processed_files > 0:
        logger.info(f"hf_vars_extractor complete - "
                    f"processed {processed_files} file(s) from {high_frequency_folder}"
                )
        return True
    
    logger.warning("No files were successfully processed.")
    return False

# ============================================================
#                 ENTRY POINT
# ============================================================
def hf_vars_main(data_root: Path,
                outputs_root: Path,
                cfd_conditions: Optional[Dict[str, float]],
            ) -> None:
    """Iterate over all fleet sub-directories and run ``hf_vars_extractor`` for each.
    
    Args:
        data_root:      Root folder containing one sub-folder per fleet.
        outputs_root:   Root folder for all outputs.
        cfd_conditions: CFD filtering conditions forwarded to the extractor.
    """
    logger.info("=" * 70)
    logger.info("Starting high-frequency variable extraction")
    logger.info(f"  data root    : {data_root}")
    logger.info(f"  outputs root : {outputs_root}")
    logger.info("=" * 70)
    
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        return
    
    fleet_dirs: Dict[str, Path] = {
        "23K_HUDONG": data_root / "23000_CSSC_HUDONG_SCS",
        "23K_JN":     data_root / "23000_CSSC_JN_SCH",
        # "23K_DF":     data_root / "23000_DF_HZ",      This one folder has no real usable data in it.
    }
    
    analysis_save_path = outputs_root / "analysis"
    analysis_save_path.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    for fleet_name, fleet_dir in fleet_dirs.items():
        logger.info(f"\n{'='*70}\nFleet: {fleet_name}\n{'='*70}")
        
        if not fleet_dir.exists():
            logger.warning(f"Fleet directory not found: {fleet_dir} - skipping")
            continue
        
        fleet_analysis_path    = analysis_save_path / fleet_name
        fleet_preprocessed_path = outputs_root / fleet_name
        fleet_analysis_path.mkdir(parents=True, exist_ok=True)
        fleet_preprocessed_path.mkdir(parents=True, exist_ok=True)
        
        success = hf_vars_extractor(high_frequency_folder=str(fleet_dir),
                                    analysis_save_path=str(fleet_analysis_path),
                                    preprocessed_save_path=str(fleet_preprocessed_path),
                                    perform_imputation=True,      # Make it by default as True
                                    speed_filter_threshold=1.0,
                                    power_filter_threshold=500.0,
                                    return_df="power",
                                    cfd_conditions=cfd_conditions,
                                )
        
        if success:
            total_processed += 1
            
    logger.info("\n" + "=" * 70)
    logger.info(f"Extraction complete - "
                f"{total_processed}/{len(fleet_dirs)} fleets processed successfully"
            )
    logger.info(f"  Analysis    : {analysis_save_path}")
    logger.info(f"  Preprocessed: {outputs_root}")
    logger.info("=" * 70)