"""
Wind Resistance Calculator Module

This module calculates wind resistance forces on a ship based on CFD (Computational Fluid Dynamics) 
data and high-frequency sensor measurements. It performs the following operations:

1. Loads and interpolates wind drag coefficients (Cx) from CFD simulations
2. Processes relative wind angles and converts them to the appropriate range [0, 180°]
3. Calculates wind resistance forces at different heights (50m and 100m) using:
   R_wind = -0.5 * ρ_air * V_rel² * A_front * Cx

Engineering Assumptions:
------------------------
- The reference wind area (A_front / A_ref) is assumed constant.
- The CFD-derived drag coefficients (Cx) were computed using a fixed vessel draft and 
    corresponding fixed projected above-water geometry.
- High-frequency draft variations (e.g., due to heave in waves) are assumed not to 
    significantly affect the projected above-water wind area.
- Therefore, dynamic changes in draft are not applied to A_ref during force computation.
- Consistency is maintained by using the same reference area definition that was 
    employed during CFD coefficient generation.

Rationale:
Wind resistance primarily depends on the exposed above-water projected area. 
For operational sea states, short-term draft oscillations do not materially 
change this area. Maintaining a constant A_ref ensures physical consistency 
between CFD-derived coefficients and real-time force reconstruction, which is 
particularly important for downstream shaft power prediction models.

Author: Khalil Chouikri
Date: February 2026
"""
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Tuple, Optional
from scipy.interpolate import CubicSpline

# ----------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------
# Initial basic config - will be updated with file handler after setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
#                Ship and Environmental Constants
# ----------------------------------------------------------------
SHIP_CONSTANTS = {
    'FRONTAL_SURFACE_AREA': 3395,  # m² - Ship's frontal cross-sectional area
    'BEAM_WIDTH': 63.9,            # m - Ship's beam (width)
    'AIR_DENSITY': 1.21,           # kg/m³ - Standard air density at sea level
    'WATER_DENSITY': 1026,         # kg/m³ - Seawater density
    'GRAVITY': 9.81,               # m/s² - Gravitational acceleration
    'KNOTS_TO_MS': 0.514444        # Conversion factor: knots to meters per second
}

# ----------------------------------------------------------------
#                Cx data uploaded from CFD data
# ----------------------------------------------------------------
def load_cx_data(cfd_cx_file: Path) -> Tuple[CubicSpline, CubicSpline]:
    """
    Load wind drag coefficient (Cx) data from CFD simulations.
    
    The CSV file should contain:
    - HEADING: Wind angles in radians [0, 2π]
    - CX_50: Drag coefficients at 50m height
    - CX_100: Drag coefficients at 100m height
    
    Args:
        cfd_cx_file: Path to the CSV file containing Cx data
        
    Returns:
        Tuple of (Cx_50_interpolator, Cx_100_interpolator) as CubicSpline objects
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        KeyError: If required columns are missing
        ValueError: If data format is invalid
    """
    if not cfd_cx_file.exists():
        raise FileNotFoundError(f"CFD data file not found: {cfd_cx_file}")
    
    try:
        df = pd.read_csv(cfd_cx_file)
        
        # Validate required columns
        required_cols = ['HEADING', 'CX_50', 'CX_100']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        # Extract data
        angles = df['HEADING'].to_numpy()
        cx_50  = df['CX_50'].to_numpy()
        cx_100 = df['CX_100'].to_numpy()
        
        # Validate angle range
        if angles.min() < 0 or angles.max() > 2 * np.pi:
            raise ValueError("HEADING values must be in range [0, 2π] radians")
        
        # Create interpolators
        cx_50_interpolator = CubicSpline(angles, cx_50)
        cx_100_interpolator = CubicSpline(angles, cx_100)
        
        logger.info(f"Successfully loaded Cx data from {cfd_cx_file}")
        logger.info(f"  - Data points: {len(angles)}")
        logger.info(f"  - Angle range: [{angles.min():.3f}, {angles.max():.3f}] rad")
        
        return cx_50_interpolator, cx_100_interpolator
    
    except Exception as e:
        logger.error(f"Failed to load Cx data from {cfd_cx_file}: {e}")
        raise
    
# ----------------------------------------------------------------
#                Cx data interpolation from CFD data
# ----------------------------------------------------------------
def interpolate_cx_coefficients(high_frequency_url: Path,
                                wind_angle_col: str,
                                cx_50_interpolator: CubicSpline,
                                cx_100_interpolator: CubicSpline
                                ) -> None:
    """
    Interpolate Cx coefficients for measured wind angles and save to file.
    
    Reads high-frequency sensor data, validates wind angles, interpolates 
    corresponding Cx values, and updates the file with new columns.
    
    Args:
        high_frequency_url: Path to feather file with sensor data
        wind_angle_col: Name of the column containing wind angles (MUST be in radians)
        cx_50_interpolator: Interpolator for Cx at 50m height
        cx_100_interpolator: Interpolator for Cx at 100m height
    
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If wind angle column is missing or contains invalid data
    """
    if not high_frequency_url.exists():
        raise FileNotFoundError(f"Data file not found: {high_frequency_url}")
    
    logger.info(f"Loading data from {high_frequency_url}")
    df = pd.read_feather(high_frequency_url)
    
    # Validate wind angle column
    if wind_angle_col not in df.columns:
        raise ValueError(f"Column '{wind_angle_col}' not found. Available: {df.columns.tolist()}")
    
    if df[wind_angle_col].isnull().any():
        null_count = df[wind_angle_col].isnull().sum()
        raise ValueError(f"Column '{wind_angle_col}' contains {null_count} NaN values")
    
    # Check if angles are likely in degrees and convert if necessary
    angles = df[wind_angle_col].to_numpy()
    max_angle = angles.max()
    min_angle = angles.min()
    
    # If max angle > 2π (≈6.28), likely in degrees
    if max_angle > 2 * np.pi:
        logger.warning(f"Wind angles appear to be in DEGREES (max={max_angle:.2f}).")
        logger.warning(f"Converting to RADIANS for interpolation...")
        
        df[wind_angle_col] = np.radians(df[wind_angle_col])

        angles = df[wind_angle_col].to_numpy()
        logger.info(f"Converted to radians: range [{angles.min():.4f}, {angles.max():.4f}]")
    else:
        logger.info(f"Angles appear to be in radians: range [{min_angle:.4f}, {max_angle:.4f}]")
    
    # Normalize angle range to [0, 2π]
    if angles.min() < 0 or angles.max() > 2 * np.pi:
        logger.warning(f"Normalizing angles to [0, 2π] range...")
        df[wind_angle_col] = df[wind_angle_col] % (2 * np.pi)
        angles = df[wind_angle_col].to_numpy()
    
    # Interpolate Cx coefficients
    logger.info(f"Interpolating Cx coefficients for {len(df)} data points")
    df['CX_50'] = cx_50_interpolator(angles)
    df['CX_100'] = cx_100_interpolator(angles)
    
    # Save updated data
    df.to_feather(high_frequency_url)
    logger.info(f"Cx coefficients saved to {high_frequency_url}")
    logger.info(f"- CX_50 range: [{df['CX_50'].min():.4f}, {df['CX_50'].max():.4f}]")
    logger.info(f"- CX_100 range: [{df['CX_100'].min():.4f}, {df['CX_100'].max():.4f}]")

# ----------------------------------------------------------------
#                Wind resistance calculation
# ----------------------------------------------------------------
def calculate_wind_resistance(cx: float, v_rel: float) -> float:
    """
    Calculate wind resistance force using aerodynamic drag equation.
    
    Formula: R_wind = -0.5 * ρ_air * A_front * Cx * V_rel²
    
    The negative sign indicates that wind resistance opposes the ship's motion.
    
    Args:
        cx: Wind drag coefficient (dimensionless)
        v_rel: Relative wind speed in m/s
        
    Returns:
        Wind resistance force in Newtons (N)
        
    Note:
        Negative values indicate resistance opposing forward motion.
    """
    return (-0.5 
            * SHIP_CONSTANTS['AIR_DENSITY'] 
            * SHIP_CONSTANTS['FRONTAL_SURFACE_AREA'] 
            * cx 
            * (v_rel ** 2)
    )
    
# ----------------------------------------------------------------
#                Wind resistance computing and saving
# ----------------------------------------------------------------
def compute_wind_resistance(high_frequency_url: Path,
                            wind_speed_col: str,
                            wind_angle_col: str
                        ) -> None:
    """
    Compute wind resistance forces at 50m and 100m heights.
    
    Reads data with previously interpolated Cx values, calculates resistance
    forces for each measurement, and saves results back to file.
    
    Args:
        high_frequency_url: Path to feather file with sensor data
        wind_speed_col: Name of column with relative wind speed (m/s)
        wind_angle_col: Name of column with relative wind angle (radians)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        ValueError: If required columns are missing or contain invalid data
    """
    if not high_frequency_url.exists():
        raise FileNotFoundError(f"Data file not found: {high_frequency_url}")
    
    logger.info(f"Computing wind resistance from {high_frequency_url}")
    df = pd.read_feather(high_frequency_url)
    
    # Validate required columns
    required_cols = [wind_speed_col, wind_angle_col, 'CX_50', 'CX_100']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values
    for col in required_cols:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            raise ValueError(f"Column '{col}' contains {null_count} NaN values")
    
        # Calculate wind resistance at both heights
    logger.info(f"Calculating wind resistance for {len(df)} data points")
    df['WIND_RESISTANCE_50M']  = df.apply(lambda row: calculate_wind_resistance(row['CX_50'], row[wind_speed_col]),axis=1)
    df['WIND_RESISTANCE_100M'] = df.apply(lambda row: calculate_wind_resistance(row['CX_100'], row[wind_speed_col]), axis=1)
    
    # Save results
    df.to_feather(high_frequency_url)
    logger.info(f"Wind resistance data saved to {high_frequency_url}")
    logger.info(f"- Resistance 50m range: [{df['WIND_RESISTANCE_50M'].min():.2f}, "
                f"{df['WIND_RESISTANCE_50M'].max():.2f}] N")
    logger.info(f"- Resistance 100m range: [{df['WIND_RESISTANCE_100M'].min():.2f}, "
                f"{df['WIND_RESISTANCE_100M'].max():.2f}] N")

# ------------------------------------------------------------
def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
# ============================================================
#                    Main execution function
# ============================================================
def wind_resistance_main(cfd_cx_file, high_frequency_file,  config_path: Path):
    """
    Main execution function demonstrating the complete wind resistance workflow.
    """
    # Load config
    config = load_config(config_path)
    columns = config["wind_resistance"]["columns"]
    wind_angle_col = columns["wind_angle"]
    wind_speed_col = columns["wind_speed"]
    
    try:
        # Step 1: Load CFD data and create interpolators
        logger.info("=" * 70)
        logger.info("STEP 1: Loading CFD drag coefficient data")
        logger.info("=" * 70)
        cx_50_interp, cx_100_interp = load_cx_data(cfd_cx_file)
        
        # Step 2: Interpolate Cx for measured wind angles
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Interpolating Cx coefficients")
        logger.info("=" * 70)
        interpolate_cx_coefficients(high_frequency_file,
                                    wind_angle_col,
                                    cx_50_interp,
                                    cx_100_interp
                                )
        
        # Step 3: Calculate wind resistance forces
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Computing wind resistance forces")
        logger.info("=" * 70)
        compute_wind_resistance(high_frequency_file,
                                wind_speed_col,
                                wind_angle_col
                            )
        
        logger.info("\n" + "=" * 70)
        logger.info("Wind resistance calculation completed successfully!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Wind resistance calculation failed: {e}")
        raise
