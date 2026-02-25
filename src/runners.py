import os
import logging 
import pandas as pd

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

# custom imports:
from src.pbm_model_main import pbm_main
# ----------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------
# Initial basic config - will be updated with file handler after setup
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                )
logger = logging.getLogger(__name__)

# ============================================================
#                     ENTRY POINT
# ============================================================
# ----------------------------------------------------------------
# Configuration Dataclass
# ----------------------------------------------------------------
@dataclass
class RequiredColumns:
    """Schema definition for required data columns."""
    timestamp: str = 'TIMESTAMP'
    imo: str       = 'IMO'
    lbp: str       = 'LBP'
    beam: str      = 'BEAM'
    propeller_diam: str = 'PROPELLER_DIAM'  # Optional - will use default if missing
    draft_mean: str     = 'DRAFT_MEAN'
    speed: str          = 'SPEED_THROUGH_WATER_LONG'
    wave_period: str    = 'TOTAL_WAVE_MEAN_PERIOD'
    wave_height: str    = 'TOTAL_WAVE_SIGNIFICANT_HEIGHT'
    wave_dir: str       = 'TOTAL_WAVE_MEAN_DIR'
    wind_dir: str       = 'REL_WIND_DIR_10M'
    wind_speed: str     = 'REL_WIND_SPEED_10M'
    shaft_power: str    = 'SHAFT_POWER'
    
    def get_required_columns(self) -> List[str]:
        """Return list of strictly required column names (excludes optional columns)."""
        return [self.timestamp, self.imo, self.lbp, self.beam, 
                self.draft_mean, self.speed,
                self.wave_period, self.wave_height, self.wave_dir,
                self.wind_dir, self.wind_speed, self.shaft_power
            ]
    
    def get_all_columns(self) -> List[str]:
        """Return list of all column names including optional ones."""
        return [self.timestamp, self.imo, self.lbp, self.beam, 
                self.propeller_diam, self.draft_mean, self.speed,
                self.wave_period, self.wave_height, self.wave_dir,
                self.wind_dir, self.wind_speed, self.shaft_power
            ]

# ----------------------------------------------------------------
# Data Validation
# ----------------------------------------------------------------
def validate_directory(directory: Path) -> None:
    """
    Validate that directory exists and is accessible.
    
    Args:
        directory: Path to validate
        
    Raises:
        ValueError: If directory doesn't exist or isn't accessible
    """
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory}")
    
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    
    if not os.access(directory, os.R_OK):
        raise ValueError(f"Directory is not readable: {directory}")

# ------------------------------------------
#           Validate dataframe 
# ------------------------------------------
def validate_dataframe(df: pd.DataFrame, 
                        required_cols: RequiredColumns,
                        filename: str
                    ) -> Dict[str, bool]:
    """
    Validate dataframe contents and structure.
    
    Args:
        df: DataFrame to validate
        required_cols: Required column configuration
        filename: Name of file for logging
        
    Returns:
        Dictionary with validation results and warnings
    """
    results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check for missing columns (only strictly required ones)
    required = required_cols.get_required_columns()
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        results['is_valid'] = False
        results['errors'].append(f"Missing required columns: {missing}")
        return results
    
    # Check for optional columns and log warnings
    all_cols = required_cols.get_all_columns()
    optional_missing = [col for col in all_cols if col not in required and col not in df.columns]
    if optional_missing:
        logger.info(f"Optional columns missing in {filename}: {optional_missing}. Will use defaults.")
    
    # Check for empty dataframe
    if df.empty:
        results['is_valid'] = False
        results['errors'].append("DataFrame is empty")
        return results
    
    # Check for excessive null values
    null_threshold = 0.5  # 50% threshold
    for col in required:
        null_pct = df[col].isna().sum() / len(df)
        if null_pct > null_threshold:
            results['warnings'].append(f"Column '{col}' has {null_pct:.1%} null values")
    
    # Validate numeric columns
    numeric_cols = [required_cols.lbp, required_cols.beam, required_cols.propeller_diam,
                    required_cols.draft_mean, required_cols.speed, required_cols.wave_period,
                    required_cols.wave_height, required_cols.shaft_power
                    ]
    
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                results['warnings'].append( f"Column '{col}' is not numeric type")
            
            # Check for negative values where inappropriate
            if col in [required_cols.speed, required_cols.wave_height, required_cols.shaft_power]:
                if (df[col] < 0).any():
                    results['warnings'].append(f"Column '{col}' contains negative values")
    
    # Log validation results
    if results['warnings']:
        logger.warning( f"Validation warnings for {filename}: "
                        f"{'; '.join(results['warnings'])}")
    
    return results

# ----------------------------------------------------------------
# File Processing
# ----------------------------------------------------------------
def load_feather_file(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Safely load a feather file with error handling.
    
    Args:
        file_path: Path to feather file
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        logger.info(f"Loading FEATHER file: {file_path.name}")
        df = pd.read_feather(file_path)
        
        logger.info(f"Successfully loaded: {file_path.name} "
                    f"({len(df):,} rows, {len(df.columns)} columns)"
                )
        return df
        
    except Exception as e:
        logger.exception(f"Failed to load {file_path.name}: {e}")
        return None
# ----------------------------------------------------------------
# Getting feather files function
# ----------------------------------------------------------------

def get_feather_files(directory: Path, recursive: bool = True) -> List[Path]:
    """
    Get all feather files in directory and subdirectories.
    
    Args:
        directory: Directory to search
        recursive: If True, search subdirectories recursively
        
    Returns:
        List of feather file paths organized by fleet folder
        
    Raises:
        ValueError: If no feather files found
    """
    if recursive:
        # Recursively search all subdirectories for feather files
        feather_files = list(directory.rglob("*.feather"))
    else:
        # Only search immediate directory
        feather_files = list(directory.glob("*.feather"))
    
    if not feather_files:
        raise ValueError(f"No .feather files found in directory: {directory} "
                        f"(recursive={'enabled' if recursive else 'disabled'})"
                        )
    
    # Log found files organized by subdirectory
    files_by_folder = {}
    for file_path in feather_files:
        # Get the immediate parent folder name (fleet name)
        fleet_folder = file_path.parent.name
        if fleet_folder not in files_by_folder:
            files_by_folder[fleet_folder] = []
        files_by_folder[fleet_folder].append(file_path.name)
    
    logger.info(f"Found {len(feather_files)} feather file(s) across {len(files_by_folder)} folder(s)")
    for folder, files in files_by_folder.items():
        logger.info(f" |--> {folder}: {len(files)} file(s)")
    
    return feather_files

# ----------------------------------------------------------------
#   Ensuring physical constraints are validated
# -----------------------------------------------------------------
def _ensure_physical_constraints(df: pd.DataFrame, columns: list, filename: str) -> bool:
    """Helper to ensure specific columns contain only non-negative values."""
    modified = False
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            neg_mask = df[col] < 0
            if neg_mask.any():
                neg_count = neg_mask.sum()
                logger.warning(f"File {filename}: Found {neg_count} negative values in '{col}'. Applying abs().")
                df[col] = df[col].abs()
                modified = True
    return modified


# ----------------------------------------------------------------
# Main Runner Function
# ----------------------------------------------------------------
def physics_based_runner(high_frequency_dir: Path,
                        root_dir: Path,
                        plots_folder: Path,
                        required_cols: Optional[RequiredColumns] = None,
                        skip_on_error: bool = False,
                        validate_data: bool = True,
                        recursive: bool = True
                    ) -> Dict[str, int]:
    """
    Process all feather files in directory through physics-based model.
    
    Args:
        high_frequency_dir: Directory containing feather files (preprocessed folder)
        base_dir: Project root directory
        plots_folder: Directory where plots will be saved
        required_cols: Column configuration (uses defaults if None)
        skip_on_error: Continue processing if a file fails
        validate_data: Perform data validation checks
        recursive: Search subdirectories for feather files (True for fleet structure)
        
    Returns:
        Dictionary with processing statistics
    """
    required_cols = required_cols or RequiredColumns()
    
    stats = {'total_files': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0
            }
    
    # Validate directory
    try:
        validate_directory(high_frequency_dir)
    except ValueError as e:
        logger.error(f"Directory validation failed: {e}")
        raise
    
    # Get feather files
    try:
        feather_files = get_feather_files(high_frequency_dir, recursive=recursive)
        stats['total_files'] = len(feather_files)
    except ValueError as e:
        logger.error(str(e))
        raise
    
    # Process each file
    for file_path in feather_files:
        filename = file_path.name
        
        logger.info(f"{'='*60}")
        logger.info(f"Processing: {filename}")
        logger.info(f"{'='*60}")
        
        # Load file
        df = load_feather_file(file_path)
        if df is None:
            stats['failed'] += 1
            if not skip_on_error:
                raise RuntimeError(f"Failed to load file: {filename}")
            continue
        
        # Prevent negative values for essential physical features that must be >= 0
        cols_to_abs = [required_cols.speed, required_cols.wave_height, required_cols.shaft_power]
        modified_for_negatives = False
        for col in cols_to_abs:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if (df[col] < 0).any():
                    neg_count = (df[col] < 0).sum()
                    logger.warning(f"File {filename}: Found {neg_count} negative values in '{col}'. Taking absolute value.")
                    
                    df[col] = df[col].abs()
                    modified_for_negatives = True
        
        if modified_for_negatives:
            df.to_feather(file_path)
        
        # Validate data
        if validate_data:
            v_result = validate_dataframe(df, required_cols, filename)
            if not v_result.get('is_valid', False):
                # Safeguard against non-string errors or None types
                errs = v_result.get('errors')
                err_str = "; ".join(map(str, errs)) if isinstance(errs, list) else "Unknown error"
                
                logger.warning(f"Skipping {filename} | Errors: {err_str}")
                stats['skipped'] += 1
                continue
        
        # Run physics-based model
        try:
            
            # Debug logging
            logger.debug(f"DEBUG - base_dir: {root_dir}")
            logger.debug(f"DEBUG - high_frequency_dir: {high_frequency_dir}")
            logger.debug(f"DEBUG - file_path: {file_path}")
            logger.debug(f"DEBUG - File exists: {file_path.exists()}")
            
            logger.info(f"Running PBM for {filename}...")
            
            # Call pbm_main with the correct parameters
            pbm_main(ROOT_DIR=root_dir, filename=filename, OUTPUT_DIR=plots_folder)
            
            stats['processed'] += 1
            logger.info(f"Successfully processed: {filename}")
            
        except Exception as e:
            logger.exception(f"PBM processing failed for {filename}: {e}")
            stats['failed'] += 1
            if not skip_on_error:
                raise
    
    # Log summary
    logger.info(f"\n{'='*60}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total files: {stats['total_files']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"{'='*60}\n")
    
    return stats