"""
Shared Constants for Physics-Based Model

This module contains constants and shared utilities used across multiple PBM modules
to avoid circular imports and maintain a single source of truth.
"""

import logging
import numpy as np 

# ----------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Ship and Environmental Constants
# ----------------------------------------------------------------
SHIP_CONSTANTS = {
    'FRONTAL_SURFACE_AREA': 3395,  # m² - Ship's frontal cross-sectional area
    'BEAM_WIDTH': 63.9,            # m - Ship's beam (width)
    'AIR_DENSITY': 1.21,           # kg/m³ - Standard air density at sea level
    'WATER_DENSITY': 1026,         # kg/m³ - Seawater density
    'GRAVITY': 9.81,               # m/s² - Gravitational acceleration
    'KNOTS_TO_MS': 0.514444        # Conversion factor: knots to meters per second
}

# Available data sources in the HYDRO_DB
AVAILABLE_SOURCES = ['CFD_WITH_ROUGHNESS',
                    'CFD_WITHOUT_ROUGHNESS',
                    'MT_WITHOUT_WIND',
                    'MT_WITH_WIND'
                ]

def determine_epsilon(speed_filtered: np.ndarray, draft_filtered: np.ndarray):
    """
    Determine optimal epsilon parameter for RBF interpolation.
    
    This function calculates an appropriate epsilon value based on the
    characteristic scales and spacing of the input data.
    
    Args:
        speed_filtered: Array of filtered speed values (m/s)
        draft_filtered: Array of filtered draft values (m)
        
    Returns:
        float: Calculated epsilon value for RBF kernel
        
    Note:
        Uses 0.1 * sqrt(speed_range * draft_range) as the epsilon formula.
    """
    # Calculate characteristic scales
    speed_range = speed_filtered.max() - speed_filtered.min()
    draft_range = draft_filtered.max() - draft_filtered.min()

    # Typical spacing between data points
    speed_spacing = np.median(np.diff(np.sort(np.unique(speed_filtered))))
    draft_spacing = np.median(np.diff(np.sort(np.unique(draft_filtered))))

    # Epsilon as geometric mean of spacings (good starting point)
    # epsilon = np.sqrt(speed_spacing * draft_spacing)

    # Or use a fraction of the data range (this is the preferred method)
    epsilon = 0.1 * np.sqrt(speed_range * draft_range)
    logger.info(f"Using epsilon={epsilon:.4f} for multiquadric kernel")
    
    return epsilon