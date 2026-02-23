import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Union, List

# ============================================================
#                 LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
#                 MAIN EXECUTION FUNCTION
# ============================================================
def cfd_data_extractors(cfd_data_paths: Dict[str, str]) -> tuple[Dict[str, Union[Dict, List]], Dict[str, float]]:
    """
    Test all CFD data extractors using provided database paths.
    
    Parameters
    ----------
    cfd_data_paths : Dict[str, str]
        Dictionary containing paths to CFD databases with the following keys:
        - 'aero'
        - 'hydro_db'
        - 'r_hydro'
        - 'propeller'
        - 'wave'
    """
    logger.info("Starting extractor tests...")
    logger.info("=" * 70)
    
    required_keys = {"aero",
                    "hydro_db",
                    "r_hydro",
                    "propeller",
                    "wave",
                    }
    
    missing = required_keys - set(cfd_data_paths)
    if missing:
        raise KeyError(f"Missing CFD database paths: {missing}")
    
    cfd_cx_path             = cfd_data_paths["aero"]
    cfd_hydro_db_path       = cfd_data_paths["hydro_db"]
    cfd_cw_resistances_path = cfd_data_paths["r_hydro"]
    cfd_propeller_path      = cfd_data_paths["propeller"]
    cfd_wave_path           = cfd_data_paths["wave"]
    spawave_coeffs_path     = cfd_data_paths["wave"]
    
    logger.info("All CFD paths resolved successfully.")
    
    results = {}
    cfd_conditions = {}
    # Test aero_vars_extractor
    try:
        result = aero_vars_extractor(cfd_cx_path)
        results['aero'] = result
        logger.info("\n[AERO VARIABLES]")
        logger.info(f"Heading: {result['heading']}")
        logger.info(f"Cx: {result['Cx']}")
        logger.info(f"Surface Ref: {result['surface_ref']}")
        logger.info(f"Beam Ref: {result['beam_ref']}")
        logger.info(f"Draft Ref: {result['draft_ref']}")
        
    except Exception as e:
        logger.error(f"aero_vars_extractor failed: {e}")
    
    # Test hydro_vars_extractor
    try:
        result = hydro_vars_extractor(cfd_hydro_db_path)
        results['hydro'] = result
        logger.info("\n[HYDRO VARIABLES]")
        logger.info(f"Type: {result['type']}")
        logger.info(f"Inputs: {result['inputs']}")
        logger.info(f"Outputs: {result['outputs']}")
    except Exception as e:
        logger.error(f"hydro_vars_extractor failed: {e}")
    # ************************************************************************************
    try:                    # This is needed for hf_vars_extractor in dataset_hf.py
        cfd_conditions = get_cfd_conditions(cfd_hydro_db_path)
        logger.info("[CFD CONDITIONS]")
        logger.info(f"  Conditions: {cfd_conditions}")
    except Exception as e:
        logger.error(f"get_cfd_conditions failed: {e}")
    # *******************************************************************************
    # Test cw_resistances_extractor
    try:
        result = cw_resistances_extractor(cfd_cw_resistances_path)
        results['cw'] = result
        logger.info("\n[CW RESISTANCES]")
        logger.info(f"  Inputs: {result['inputs']}")
        logger.info(f"  Outputs: {result['outputs']}")
    except Exception as e:
        logger.error(f"cw_resistances_extractor failed: {e}")
    
    # Test efficiency_vars_extractor
    try:
        result = efficiency_vars_extractor(cfd_propeller_path)
        results['efficiency'] = result
        logger.info("\n[PROPELLER EFFICIENCY]")
        logger.info(f"  Columns: {result}")
    except Exception as e:
        logger.error(f"efficiency_vars_extractor failed: {e}")
    
    # Test wave_vars_extractor
    try:
        result = wave_vars_extractor(cfd_wave_path)
        results['wave'] = result
        logger.info("\n[WAVE VARIABLES]")
        logger.info(f"Inputs: {result['inputs']}")
        logger.info(f"Outputs: {result['outputs']}")
    except Exception as e:
        logger.error(f"wave_vars_extractor failed: {e}")
    
    # Test spawave_coeffs_extractor
    try:
        result = spawave_coeffs_extractor(spawave_coeffs_path)
        results['spawave'] = result
        logger.info("\n[SPAWAVE COEFFICIENTS]")
        logger.info(f"Columns: {result}")
    except Exception as e:
        logger.error(f"spawave_coeffs_extractor failed: {e}")

    logger.info("\n" + "=" * 70)
    logger.info("Test execution completed!")
    return results, cfd_conditions
# ============================================================
#                 HELPER FUNCTIONS
# ============================================================
def _load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load CSV file with error handling."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
        raise


def _extract_columns(data: pd.DataFrame, 
                    input_cols: Union[int, slice], 
                    output_cols: Union[int, slice]) -> Dict[str, List[str]]:
    """Generic column extractor."""
    inputs = data.columns[input_cols] if isinstance(input_cols, slice) else data.columns[[input_cols]]
    outputs = data.columns[output_cols]
    
    return {
        'inputs': list(inputs),
        'outputs': list(outputs)
    }
# ============================================================
#                 EXTRACTION FUNCTIONS
# ============================================================
def aero_vars_extractor(cfd_cx_path: Union[str, Path]) -> Dict:
    """Extract aerodynamic variables from CFD Cx data."""
    data = _load_csv(cfd_cx_path)
    cols = data.columns
    
    return {'heading'    : cols[0],
            'Cx'         : cols[1:3],
            'surface_ref': cols[3],
            'beam_ref'   : cols[4],
            'draft_ref'  : cols[-1],
            }

def hydro_vars_extractor(cfd_hydro_db_path: Union[str, Path]) -> Dict[str, Union[str, List[str]]]:
    """Extract hydrodynamic variables from CFD hydro database."""
    data = _load_csv(cfd_hydro_db_path)
    data = data[data['SOURCE'] == 'CFD_WITH_ROUGHNESS']
    cols = data.columns
    
    return {
        'type': cols[0],
        'inputs': list(cols[1:3]),
        'outputs': list(cols[3:])
    }
# ************************************************************************************
def get_cfd_conditions(cfd_hydro_db_path: Union[str, Path]) -> Dict[str, float]:
    """
    Extract the range of CFD operating conditions from a hydrodynamic database.

    This function loads a CFD hydrodynamic database stored as a CSV file and
    computes the minimum and maximum values of ship speed and draft available
    in the dataset. These limits can be used for validation, interpolation,
    or consistency checks when coupling CFD-based models with higher-level
    performance or machine-learning models.

    Parameters
    ----------
    cfd_hydro_db_path : Union[str, pathlib.Path]
        Path to the CFD hydrodynamic database CSV file.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the CFD condition bounds:
        
        - ``min_speed`` : Minimum ship speed in the database
        - ``max_speed`` : Maximum ship speed in the database
        - ``min_draft`` : Minimum ship draft in the database
        - ``max_draft`` : Maximum ship draft in the database

    Raises
    ------
    KeyError
        If required columns ('SPEED', 'DRAUGHT') are missing from the dataset.
    ValueError
        If the dataset is empty or contains non-numeric values.
    """
    data = _load_csv(cfd_hydro_db_path)

    return {"min_speed": float(data["SPEED"].min()),
            "max_speed": float(data["SPEED"].max()),
            "min_draft": float(data["DRAUGHT"].min()),
            "max_draft": float(data["DRAUGHT"].max()),
        }

# ***************************************************************************************

def cw_resistances_extractor(cfd_cw_resistances_path: Union[str, Path]) -> Dict[str, List[str]]:
    """Extract calm water resistance variables."""
    data = _load_csv(cfd_cw_resistances_path)
    return _extract_columns(data, slice(0, 2), slice(2, None))


def efficiency_vars_extractor(cfd_propeller_path: Union[str, Path]) -> List[str]:
    """Extract efficiency variables from propeller data."""
    data = _load_csv(cfd_propeller_path)
    return list(data.columns)


def wave_vars_extractor(cfd_wave_path: Union[str, Path]) -> Dict[str, List[str]]:
    """Extract wave variables from CFD wave data."""
    data = _load_csv(cfd_wave_path)
    return _extract_columns(data, slice(0, 4), slice(4, None))


def spawave_coeffs_extractor(spawave_coeffs_path: Union[str, Path]) -> List[str]:
    """Extract seaway wave coefficients."""
    data = _load_csv(spawave_coeffs_path)
    return list(data.columns)
# ============================================================