import os
import logging

from pathlib import Path

# custom imports
from src.runners import physics_based_runner
from src.datasets.dataset_hf import hf_vars_main
from src.datasets.dataset_cfd import cfd_data_extractors
# ----------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------
# Initial basic config - will be updated with file handler after setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
logger = logging.getLogger(__name__)

# ============================================================
#                     MAIN EXECUTION
# ============================================================
def variables_extraction(high_freq_dir: Path, cfd_data_dict: dict, outputs_root: Path) -> None:
    """
    Extract and process variables from both CFD and high-frequency datasets.
    
    This function orchestrates the extraction pipeline by:
    1. Testing and extracting variables from CFD (NetCDF/CDF) databases
    2. Processing high-frequency sensor data and extracting relevant variables
    3. Saving extracted variables to the specified output directory
    
    The extraction process is designed to handle multiple data sources and
    prepare them for downstream analysis, visualization, or ML model training.
    
    Args:
        high_freq_dir (Path): Directory containing high-frequency sensor data files.
                            Expected to contain time-series data from various sensors.
        cfd_data_dict (dict): Dictionary containing paths to CFD database files.
                            Keys are expected to be: 'aero', 'hydro_db', 'r_hydro', 'propeller', 'wave'.
        outputs_root (Path): Root directory where extracted variables and processed
                            data will be saved. Subdirectories may be created automatically.
    
    Returns:
        None: Results are written to disk in the outputs_root directory.
    
    Raises:
        FileNotFoundError: If input directories do not exist
        PermissionError: If output directory is not writable
        ValueError: If data files are malformed or incompatible
    
    Side Effects:
        - Writes extracted variable files to outputs_root
        - Logs progress and any extraction warnings/errors
        - May create subdirectories in outputs_root
    
    Example:
        >>> from pathlib import Path
        >>> hf_dir = Path("/data/sensors/high_freq")
        >>> cfd_dir = Path("/data/cfd/simulations")
        >>> output_dir = Path("/results/extracted_vars")
        >>> variables_extraction(hf_dir, cfd_dir, output_dir)
    """
    logger.info(f"Starting variable extraction from CFD data: {cfd_data_dict}")
    results, cfd_conditions = cfd_data_extractors(cfd_data_dict)
    
    logger.info(f"Starting variable extraction from HF data: {high_freq_dir}")
    hf_vars_main(high_freq_dir, outputs_root, cfd_conditions)
    
    logger.info(f"Variable extraction complete. Results saved to: {outputs_root}")
    
# ============================================================
#                     ENTRY POINT
# ============================================================
if __name__ == "__main__":
    """
    Main entry point for the OceanPulse data processing pipeline.
    
    This script orchestrates the complete workflow:
    1. Environment setup and configuration loading
    2. Path extraction for all data sources
    3. Variable extraction from CFD and high-frequency datasets
    
    The returned all_cfd_paths dictionary enables future expansion to process
    all CFD databases (aero, propeller, wave, etc.) in a loop or parallel tasks.
    
    Future Enhancement Example:
        # TODO: Process all CFD databases iteratively
        for db_name, db_path in all_cfd_paths.items():
            logger.info(f"Processing {db_name} database...")
            process_cfd_database(db_path, OUTPUT_BASE / db_name)
    """
    # ── PATH CONFIGURATION ────────────────────────────────────────────────────
    ROOT_DIR        = Path(__file__).resolve().parent
    OUTPUT_DIR      = ROOT_DIR / "outputs"
    DATA_DIR        = ROOT_DIR / "data"

    # Input data
    high_freq_dir   = DATA_DIR / "high_frequency"
    cfd_data_dict   = {
        "aero"     : DATA_DIR / "CFD" / "AERO"  / "CX.csv",
        "hydro_db" : DATA_DIR / "CFD" / "HYDRO" / "HYDRO_DB.csv",
        "r_hydro"  : DATA_DIR / "CFD" / "HYDRO" / "CFD_PcwPower.csv",
        "propeller": DATA_DIR / "CFD" / "HYDRO" / "PROPELLER.csv",
        "wave"     : DATA_DIR / "CFD" / "HYDRO" / "SPAWAVE_COEFFS.csv",
    }

    # Output sub-directories
    outputs_root    = OUTPUT_DIR / "preprocessed"
    plots_folder    = OUTPUT_DIR / "plots"
    metrics_folder  = OUTPUT_DIR / "metrics"
    pbm_plots       = plots_folder / "physics_based_model"
    # ─────────────────────────────────────────────────────────────────────────

    log_file = ROOT_DIR / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Starting OceanPulse Data Processing Pipeline")
    logger.info(f"Root Directory : {ROOT_DIR}")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("=" * 60)
    # Execute variable extraction pipeline
    try:
        variables_extraction(high_freq_dir, cfd_data_dict, outputs_root)
    except Exception as e:
        logger.error("Error during variables extraction:", exc_info=True)
    
    logger.info("=" * 60)
    # ══════════════════════════════════════════════════════════════════════════
    #                          Physics-Based Model
    # ══════════════════════════════════════════════════════════════════════════
    try:
        logger.info("Running physics-based runner...")
        stats = physics_based_runner(
            high_frequency_dir=outputs_root,
            root_dir=ROOT_DIR,
            output_dir=OUTPUT_DIR,
            plots_folder=pbm_plots,
            recursive=True,  # searches sub-directories (fleet structure)
        )
        logger.info(f"PBM run complete – {stats}")
    except Exception as e:
        logger.error("Error during physics_based_runner:", exc_info=True)