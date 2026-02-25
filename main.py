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
    # Define paths to CFD databases (these should be updated to actual paths)
    # Update logging to include a file handler in the new output directory
    ROOT_DIR = Path(__file__).resolve().parent 
    OUTPUT_DIR = Path(os.path.join(ROOT_DIR, "outputs"))
    
    high_freq_dir = Path(os.path.join(ROOT_DIR, "data", "high_frequency"))
    
    cfd_data_dict = {
        "aero":     os.path.join(ROOT_DIR, "data", "CFD", "AERO", "Cx.csv"),
        "hydro_db": os.path.join(ROOT_DIR, "data", "CFD", "HYDRO", "HYDRO_DB.csv"),
        "r_hydro":  os.path.join(ROOT_DIR, "data", "CFD", "HYDRO", "CFD_PcwPower.csv"),
        "propeller":os.path.join(ROOT_DIR, "data", "CFD", "HYDRO", "PROPELLER.csv"),
        "wave":     os.path.join(ROOT_DIR, "data", "CFD", "HYDRO", "SPAWAVE_COEFFS.csv"),
    }
    log_file = ROOT_DIR / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 60)
    logger.info(f"Starting OceanPulse Data Processing Pipeline")
    logger.info(f"Output Directory: {OUTPUT_DIR}")
    logger.info("=" * 60)
    

    models_dir    = os.path.join("models")
    
    outputs_root  = Path(os.path.join(OUTPUT_DIR , 'preprocessed'))
    plots_folder  = Path(os.path.join(OUTPUT_DIR , 'plots'))
    metrics_folder= Path(os.path.join(OUTPUT_DIR , 'metrics'))
    # Execute variable extraction pipeline
    try:
        variables_extraction(high_freq_dir, cfd_data_dict, outputs_root)
    except Exception as e:
        logger.error("Error during variables extraction:", exc_info=True)
    
    logger.info("=" * 60)
    # ==============================================================================
    #                                Physics based model 
    # ==============================================================================
    # Running physics-based model on all the ships.
    pbm_plots = Path(os.path.join(plots_folder, "physics_based_model"))
    try:
        logger.info("Running physics-based runner...")
        stats = physics_based_runner(high_frequency_dir=outputs_root,                
                                    root_dir=ROOT_DIR,                                  
                                    plots_folder=pbm_plots,                    
                                    recursive=True  # Default - searches subdirectories
                                    )
    except Exception as e:
        logger.error("Error during physics_based_runner:", exc_info=True)