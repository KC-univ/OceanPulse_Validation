import os 
import logging 

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
def wave_resistance_main():
    pass
