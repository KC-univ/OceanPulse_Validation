"""
Calm Water Resistance Calculator Module

This module calculates calm water resistance forces on a ship's hull as it moves through water.
It uses CFD (Computational Fluid Dynamics) simulation data and RBF (Radial Basis Function) 
interpolation to estimate resistance for any combination of ship speed and draft.

Key Features:
1. Loads and processes CFD hydrodynamic simulation data
2. Creates 2D RBF interpolators for speed-draft combinations
3. Calculates calm water resistance power for high-frequency sensor data
4. Generates comprehensive visualization plots

Note: The HYDRO_DB 'FORCE' column is negative, so resistance = -1 * FORCE

Author: Optimized version
Date: December 2025
"""

import os 
import yaml
import logging 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


from pathlib import Path 
from typing import Optional, Tuple 
from mpl_toolkits.mplot3d import Axes3D 
from scipy.interpolate import RBFInterpolator 

# Import shared constants and utilities
from src.pbm_model.constants import SHIP_CONSTANTS, AVAILABLE_SOURCES, determine_epsilon
# ----------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------
# Initial basic config - will be updated with file handler after setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# --- Calm Water Resistance Functions
# ----------------------------------------------------------------

def load_cfd_hydro_data(hydro_db_path: Path,
                        source: str,
                        kernel: str = 'thin_plate_spline',
                        epsilon: Optional[float] = None
                    ) -> Tuple[float, RBFInterpolator]:
    """
    Load CFD hydrodynamic data and create an RBF interpolator.
    
    The CSV file should contain:
    - SOURCE: Data source identifier (e.g., 'CFD_WITH_ROUGHNESS')
    - DRAUGHT: Ship draft values in meters
    - SPEED: Ship speed values in m/s
    - FORCE: Force values (negative resistance in Newtons)
    
    Args:
        hydro_db_path: Path to the CSV file containing CFD data
        source: Source type to filter data (must be in AVAILABLE_SOURCES)
        kernel: RBF kernel type - options include:
                'thin_plate_spline' (recommended for smooth data)
                'cubic' (C2 continuous)
                'multiquadric' (default scipy)
                'linear', 'gaussian', 'inverse_multiquadric'
        
    Returns:
        RBFInterpolator configured for calm water resistance prediction
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If source is invalid or data is malformed
        KeyError: If required columns are missing
    """
    if not hydro_db_path.exists():
        raise FileNotFoundError(f"File not found: {hydro_db_path}")
    
    if source not in AVAILABLE_SOURCES:
        raise ValueError(f"Invalid source: {source}. Must be one of {AVAILABLE_SOURCES}")
    try:
        # Load and filter data
        logger.info(f"Loading CFD data from {hydro_db_path}")
        data = pd.read_csv(hydro_db_path)
        
        if "SOURCE" not in data.columns or "DRAUGHT" not in data.columns or "SPEED" not in data.columns:
            raise KeyError("Column SOURCE, DRAUGHT or SPEED not found in data")
        
        df = data[data['SOURCE'] == source].copy()
        
        if len(df) == 0:
            raise ValueError(f"No data found for source: {source}")
        
        logger.info(f"Filtered {len(df)} records for source: {source}")
        
        # Validate required columns
        required_cols = ['DRAUGHT', 'SPEED', 'FORCE']
        missing_cols  = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise KeyError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Extract and validate data
        draft = df['DRAUGHT'].to_numpy()
        speed = df['SPEED'].to_numpy()
        force = df['FORCE'].to_numpy()
        
        # Check for NaN values
        if np.isnan(draft).any() or np.isnan(speed).any() or np.isnan(force).any():
            raise ValueError("Data contains NaN values (invalid or missing in columns 'DRAUGHT', 'SPEED', 'FORCE')")
        
        # Convert force to resistance (force is negative in the database)
        resistance = -1 * np.asarray(force)
        
        # Remove zero-speed entries (they don't represent moving resistance)
        non_zero_mask  = speed != 0
        draft_filtered = draft[non_zero_mask]
        speed_filtered = speed[non_zero_mask]
        resistance_filtered = resistance[non_zero_mask]
        
        # ------------------------------ Adding epsilon parameter --------------------------
        epsilon = epsilon if epsilon is not None else determine_epsilon(speed, draft)
        # Ensure epsilon is a plain Python float (not numpy.float64) so YAML serializes it cleanly
        epsilon = float(epsilon)
        
        if len(speed_filtered) == 0 or len(draft_filtered) == 0:
            raise ValueError("No valid non zero speed or draft values found")
            
        logger.info(f"Removed {np.sum(~non_zero_mask)} zero-speed entries")
        logger.info(f"Creating RBF interpolator with kernel: {kernel}")
        logger.info(f"- Speed range: [{speed_filtered.min():.2f}, {speed_filtered.max():.2f}] m/s")
        logger.info(f"- Draft range: [{draft_filtered.min():.2f}, {draft_filtered.max():.2f}] m")
        logger.info(f"- Resistance range: [{resistance_filtered.min():.2f}, {resistance_filtered.max():.2f}] N")
        
        # Create 2D RBF interpolator (speed, draft) -> resistance
        interpolator_input = np.column_stack((speed_filtered, draft_filtered))
        pcw_interpolator = RBFInterpolator(interpolator_input, resistance_filtered, 
                                            kernel=kernel, epsilon=epsilon,
                                            smoothing=1e-3  # small regularization, helps avoid singular systems
                                            )
        
        logger.info("RBF interpolator created successfully")
        
        return epsilon, pcw_interpolator
        
    except Exception as e:
        logger.error(f"Error loading CFD data: {e}")
        raise e
    
# ------------------------------------------------------------
#           Calm water resistance function
# ------------------------------------------------------------
def calculate_calm_water_resistance(high_frequency_url: Path,
                                    speed_col: str,
                                    draft_col: str,
                                    pcw_interpolator: RBFInterpolator,
                                    fallback_interpolator: Optional[RBFInterpolator] = None,
                                    epsilon: float = 1.0
                                ) ->  pd.DataFrame:
    """
    Calculate calm water resistance power for high-frequency sensor data.
    Interpolates resistance values for each speed-draft combination in the dataset
    and adds a 'CALM_WATER_RESISTANCE' column with the results in Watts.
    
    If the primary interpolator produces negative values (physically impossible),
    the fallback interpolator is used for those specific points. If no fallback
    is provided, negative values are clipped to zero.
    
    Args:
        high_frequency_url: Path to feather file with sensor data
        speed_col: Name of column containing ship speed (m/s)
        draft_col: Name of column containing ship draft (m)
        pcw_interpolator: Primary RBF interpolator for resistance prediction
        fallback_interpolator: Optional fallback interpolator for points where
                                primary produces negative values
        epsilon: Epsilon value used in interpolator (for column naming)
        
    Raises:
        FileNotFoundError: If data file doesn't exist
        KeyError: If required columns are missing
        ValueError: If data contains invalid values (negative speed/draft, NaN)
    """
    resistance_col = 'CALM_WATER_RESISTANCE'
    
    if not high_frequency_url.exists():
        raise FileNotFoundError(f"Data file not found: {high_frequency_url}")
    
    logger.info(f"Loading data from {high_frequency_url}")
    
    # Load data
    df = pd.read_feather(high_frequency_url)
    logger.info(f"Data loaded: {len(df)} records")
    
    # Validate required columns
    required_cols = [speed_col, draft_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values
    for col in required_cols:
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            raise ValueError(f"Column '{col}' contains {null_count} NaN values")
        
    # Validate and clean data
    if (df[speed_col] < 0).any():
        neg_count = (df[speed_col] < 0).sum()
        logger.warning(f"Found {neg_count} negative speed values, taking absolute value")
        df[speed_col] = df[speed_col].abs()
        
    if (df[draft_col] < 0).any():
        neg_count = (df[draft_col] < 0).sum()
        logger.warning(f"Found {neg_count} negative draft values, taking absolute value")
        df[draft_col] = df[draft_col].abs()
    
    # Prepare input for interpolation
    logger.info("Interpolating calm water resistance for all data points")
    interpolation_input = df[[speed_col, draft_col]].to_numpy()
    
    # ------------------------------------------------------------------------
    # Calculate resistance power using primary interpolator
    # ------------------------------------------------------------------------
    logger.info("Using primary interpolator for initial predictions")
    raw_power = pcw_interpolator(interpolation_input)
    
    # Check for negative values (physically impossible for resistance/power)
    negative_mask = raw_power < 0
    negative_count = np.sum(negative_mask)
    
    if negative_count > 0:
        logger.warning(f"Primary interpolator produced {negative_count} negative predictions "
                       f"({100 * negative_count / len(raw_power):.2f}% of data)")
        # Use fallback interpolator for negative values if available
        if fallback_interpolator is not None:
            logger.info(f"Using fallback interpolator for {negative_count} negative predictions")
            fallback_power = fallback_interpolator(interpolation_input[negative_mask])
            
            # Check if fallback also produces negatives
            fallback_negative_mask = fallback_power < 0
            fallback_negative_count = np.sum(fallback_negative_mask)
            
            if fallback_negative_count > 0:
                logger.warning( f"Fallback interpolator also produced {fallback_negative_count} "
                                f"negative values - clipping to zero")
                fallback_power = np.maximum(fallback_power, 0)
            
            # Replace negative values with fallback predictions
            raw_power[negative_mask] = fallback_power
            
            logger.info(f"Successfully replaced {negative_count} values using fallback")
        else:
            # No fallback available - clip to zero
            logger.warning( f"No fallback interpolator provided - clipping {negative_count} "
                            f"negative values to zero")
            raw_power = np.maximum(raw_power, 0)
    else:
        logger.info("No negative predictions - all values physically valid")
    
    # Store results with epsilon in column name for tracking
    # Also store as standard column name for convenience
    #column_name = f'{resistance_col}_{epsilon:.4f}'
    df[resistance_col] = raw_power
    
    # Log statistics
    logger.info(f"Calm water resistance calculated successfully")
    logger.info(f"- Speed range: [{df[speed_col].min():.2f}, {df[speed_col].max():.2f}] m/s")
    logger.info(f"- Draft range: [{df[draft_col].min():.2f}, {df[draft_col].max():.2f}] m")
    logger.info(f"- Power range: [{df[resistance_col].min():.2f}, {df[resistance_col].max():.2f}] W")
    
    # Additional statistics on the corrected values
    if negative_count > 0:
        logger.info(f"- Values corrected: {negative_count} ({100 * negative_count / len(df):.2f}%)")
        logger.info(f"- Corrected value range: [{raw_power[negative_mask].min():.2f}, "
                    f"{raw_power[negative_mask].max():.2f}] W")
    
    # Save updated data
    df.to_feather(high_frequency_url)
    logger.info(f"Updated data saved to {high_frequency_url}")
    
    return df
    
# -----------------------------------------------------------------
#           The main calm water resistance
# -----------------------------------------------------------------
def cw_resistance_main( hydro_db_path: Path,
                        high_frequency_file: Path,
                        plots_dir: Path,
                        config_path: Path) -> None:
    """
    Main execution function for the calm water resistance workflow.
    
    Args:
        hydro_db_path:       Path to the CFD hydrodynamic CSV database.
        high_frequency_file: Path to the high-frequency sensor feather file.
        plots_dir:           Directory where plots will be saved.
        config_path:         Path to the YAML configuration file.
    """
    
    # ----------------------------------------------------------------
    # Load configuration
    # ----------------------------------------------------------------
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    source    = config["source"]
    speed_col = config["columns"]["speed"]
    draft_col = config["columns"]["draft"]
    kernel    = config["interpolator"]["kernel"]
    epsilon   = config["interpolator"].get("epsilon")   # None if null in yaml
    smoothing = config["interpolator"].get("smoothing", 1e-3)
    
    logger.info(f"Configuration loaded from {config_path}")
    logger.info(f"  source   : {source}")
    logger.info(f"  speed_col: {speed_col}")
    logger.info(f"  draft_col: {draft_col}")
    logger.info(f"  kernel   : {kernel}")
    logger.info(f"  epsilon  : {'auto' if epsilon is None else epsilon}")
    logger.info(f"  smoothing: {smoothing}")
    
    try:
        # ----------------------------------------------------------------
        # Step 1: Load CFD data and create interpolator
        # ----------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STEP 1: Loading CFD hydrodynamic data")
        logger.info("=" * 70)
        
        epsilon, pcw_interpolator = load_cfd_hydro_data(hydro_db_path,
                                                        source=source,
                                                        kernel=kernel,
                                                        epsilon=epsilon,
                                                    )
        
        # Write computed epsilon back to config so it can be reused next run
        # Cast to plain float so PyYAML writes it as a clean scalar, not a numpy object
        config["interpolator"]["epsilon"] = float(epsilon)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Computed epsilon ({epsilon:.6f}) saved back to {config_path}")
        
        # ----------------------------------------------------------------
        # Step 2: Calculate calm water resistance
        # ----------------------------------------------------------------
        logger.info("=" * 70)
        logger.info("STEP 2: Calculating calm water resistance")
        logger.info("=" * 70)
        
        calculate_calm_water_resistance(high_frequency_file,
                                        speed_col,
                                        draft_col,
                                        pcw_interpolator,
                                        epsilon=epsilon,
                                    )
        logger.info("=" * 70)
        logger.info("Calm water resistance calculation completed successfully!")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Calm water resistance calculation failed: {e}")
        raise