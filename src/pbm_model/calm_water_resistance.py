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
        # Guaranteed to be float here since determine_epsilon always returns float
        assert isinstance(epsilon, float), "epsilon must be a float at this point"
        
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

# ============================================================
#                     ENTRY POINT
# ============================================================
def cw_resistance_main():
    pass
