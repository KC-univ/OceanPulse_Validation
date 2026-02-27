"""
Physics-Based Model (PBM) Pipeline Runner

This module orchestrates the complete physics-based modeling pipeline for ship
power prediction, from hydrodynamic calculations through final metrics evaluation.

The pipeline includes:
    1. Calm water resistance calculation
    2. Wind resistance modeling
    3. Propeller efficiency analysis (η0, ηR, ηD)
    4. Wave spectral analysis (JONSWAP)
    5. Wave-induced forces (QTF)
    6. Wave resistance computation using SPAWAVE_method or CFD
    7. Delivered power prediction
    8. Performance metrics evaluation

Author: Khalil Chouikri
Date: 2025
Version: 2.0
"""
import logging
import datetime
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional

# -------------------------------------------------------------
#              INTERNAL MODULE IMPORTS
# -------------------------------------------------------------
# efficiencies
from src.pbm_model.eta0_module import eta_0_main
from src.pbm_model.etar_module import eta_r_main
from src.pbm_model.etah_module import eta_h_main
from src.pbm_model.etad_module import eta_d_main, compute_overall_efficiency
# metrics
from src.pbm_model.pbm_metrics import metrics_main
# waves
from src.pbm_model.jonswap_module import jonswap_main
from src.pbm_model.qtf_module import qtf_spawave_main
# resistances
from src.pbm_model.wave_resistance import wave_resistance_main
from src.pbm_model.wind_resistance import wind_resistance_main
from src.pbm_model.calm_water_resistance import cw_resistance_main
# power
from src.pbm_model.delivered_power_module import delivered_power_main

# ============================================================
#                 LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
#                   Path config class
# ------------------------------------------------------------
class PathConfig:
    """
    Centralized path configuration for the PBM pipeline.
    All paths are injected at initialization time to allow
    environment-agnostic execution (local, cluster, CI, Docker).
    """
    
    def __init__(self,
                root_dir: Path,
                output_dir: Path,
                cfd_cx_file: Path,
                hydro_db_path: Path,
                high_freq_file: Path,
                propeller_db_path: Path,
                configs_dir: Path
            ):
        # Base directories
        self.ROOT_DIR = root_dir
        self.OUTPUT_DIR = output_dir
        
        # Input data files
        self.CFD_CX_FILE = cfd_cx_file
        self.HYDRO_DB_PATH = hydro_db_path
        self.HIGH_FREQ_FILE = high_freq_file
        self.PROPELLER_DB_PATH = propeller_db_path
        
        # config file
        self.cw_resistance_config = configs_dir / "cw_resistance_config.yaml"
        self.wind_resistance_config = configs_dir / "wind_resistance_config.yaml"
        
        # Output directories
        self.PLOTS_FOLDER = self.OUTPUT_DIR / "plots" / "physics_based_model"
        self.METRICS_FOLDER = self.OUTPUT_DIR / "metrics" / "pbm_metrics"
        
    def validate_inputs(self) -> bool:
        """
        Validate that all required input files exist.
        
        Returns
        -------
        bool
            True if all files exist, False otherwise.
        """
        required_files = [ self.CFD_CX_FILE,
                        self.HYDRO_DB_PATH,
                        self.PROPELLER_DB_PATH,
                        self.HIGH_FREQ_FILE,
                        self.cw_resistance_config,
                        self.wind_resistance_config,
                    ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            logger.error("Missing required input files:")
            for file in missing_files:
                logger.error(f"  - {file}")
            return False
        
        logger.info("All input files validated successfully")
        return True
    
    def create_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.PLOTS_FOLDER.mkdir(parents=True, exist_ok=True)
        self.METRICS_FOLDER.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directories prepared under {self.OUTPUT_DIR}")

# ============================================================
#              PIPELINE EXECUTION FUNCTIONS
# ============================================================
def run_resistance_modules(config: PathConfig) -> None:
    """
    Execute resistance calculation modules.
    
    Parameters
    ----------
    config : PathConfig
        Path configuration object.
    """
    logger.info("=" * 70)
    logger.info("PHASE 1: RESISTANCE CALCULATIONS")
    logger.info("=" * 70)
    
    # Calm water resistance
    logger.info("Computing calm water resistance...")
    cw_resistance_main(config.HYDRO_DB_PATH, config.HIGH_FREQ_FILE, config.PLOTS_FOLDER, config.cw_resistance_config)
    
    # Wind resistance
    logger.info("Computing wind resistance...")
    wind_resistance_main(config.CFD_CX_FILE, config.HIGH_FREQ_FILE, config.wind_resistance_config)
    
# ============================================================
#              MAIN PIPELINE EXECUTION
# ============================================================
def run_complete_pipeline(ROOT_DIR: Path, high_freq_file: Path, skip_phases: Optional[list] = None,
                        OUTPUT_DIR: Optional[Path] = None, status_callback=None,) -> None:
    """
    Execute the complete physics-based modeling pipeline.
    
    Parameters
    ----------
    skip_phases : list of int, optional
        List of phase numbers to skip (1-5). Useful for debugging
        or when certain phases have already been completed.
        Example: [1, 2] will skip resistance and efficiency modules.
    OUTPUT_DIR : Path, optional
        Root output directory (e.g. output/run_timestamp). 
        If None, defaults to ROOT_DIR/output.
    status_callback : callable, optional
        Function to call with status updates (str).
    
    Returns
    -------
    dict
        Dictionary containing results from each phase:
        - 'jonswap_df': JONSWAP spectrum DataFrame
        - 'qtf_df': QTF DataFrame
        - 'resistance_df': Wave resistance DataFrame
        - 'metrics_df': Performance metrics DataFrame
        - 'metrics': Dictionary of metric values
    
    Raises
    ------
    FileNotFoundError
        If required input files are missing.
    Exception
        If any phase fails during execution.
    """
    logger.info("\n" + "=" * 70)
    logger.info("PHYSICS-BASED MODEL PIPELINE - START")
    logger.info("=" * 70 + "\n")
    
    if status_callback:
        status_callback("Initializing pipeline...")
        
    if OUTPUT_DIR is None:
        OUTPUT_DIR = ROOT_DIR / "output"
    
    config = PathConfig(root_dir=ROOT_DIR,
                        output_dir=OUTPUT_DIR,
                        cfd_cx_file=ROOT_DIR / "data" / "CFD" / "AERO" / "CX.csv",
                        hydro_db_path=ROOT_DIR / "data" / "CFD" / "HYDRO" / "HYDRO_DB.csv",
                        propeller_db_path=ROOT_DIR / "data" / "CFD" / "HYDRO" / "PROPELLER.csv",
                        high_freq_file=high_freq_file,
                        configs_dir=ROOT_DIR / "src" / "pbm_model" / "configs" 
                        )
    
    # --------------------------------------------------
    # Validate & prepare
    # --------------------------------------------------
    # Validate inputs
    if not config.validate_inputs():
        raise FileNotFoundError("Required input files are missing")
    
    # Create output directories
    config.create_output_dirs()
    
    # Initialize skip list
    skip_phases = skip_phases or []
    
    # Results storage
    results = {}
    
    try:
        
        # Phase 1: Resistance calculations
        if 1 not in skip_phases:
            if status_callback:
                status_callback("Phase 1/5: Resistance Calculations")
            run_resistance_modules(config)
        else:
            logger.info("Skipping Phase 1: Resistance calculations\n")
    
    except Exception as e:
        logger.error(f"\n{'=' * 70}")
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error(f"{'=' * 70}")
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise e
    
# ============================================================
#              COMMAND LINE INTERFACE
# ============================================================
def pbm_main(ROOT_DIR: Path, high_freq_file: Path, OUTPUT_DIR: Optional[Path] = None,
            skip_phases: Optional[list] = None) -> None:
    """
    Programmatic entry point for the physics-based model pipeline.

    Parameters
    ----------
    ROOT_DIR : Path
        Project root directory (contains ``data/`` and ``src/`` sub-trees).
    high_freq_file : Path
        Full path to the pre-processed feather file to run through the pipeline.
    OUTPUT_DIR : Path, optional
        Root output directory.  Defaults to ``ROOT_DIR / "outputs"``.
    skip_phases : list of int, optional
        Phase numbers to skip (1=resistance, 2=efficiency, 3=waves, 4=power, 5=metrics).
    """
    if OUTPUT_DIR is None:
        OUTPUT_DIR = ROOT_DIR / "outputs"

    skip_list = skip_phases or []

    try:
        run_complete_pipeline(ROOT_DIR, high_freq_file, skip_phases=skip_list, OUTPUT_DIR=OUTPUT_DIR)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise