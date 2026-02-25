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

def pbm_main(ROOT_DIR: Path, filename: str, OUTPUT_DIR: Path):
    pass