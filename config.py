"""Central configuration for the ship-power ML project.

All tuneable parameters (data paths, ship geometry, training hyper-parameters)
are read from environment variables so experiments can be reproduced or varied
without touching source code.  Defaults match the Aframax test vessel.
"""

import os
from dotenv import load_dotenv

# Populate os.environ from a .env file if present; silently ignored otherwise.
load_dotenv()


# ---------------------------------------------------------------------------
# Typed environment-variable helpers
# ---------------------------------------------------------------------------

def _get_str(key, default=''):
    return os.getenv(key, default)


def _get_float(key, default='0.0'):
    return float(os.getenv(key, default))


def _get_int(key, default='0'):
    return int(os.getenv(key, default))


def _get_list(key, default='', sep=','):
    raw = os.getenv(key, default)
    return [item.strip() for item in raw.split(sep) if item.strip()]


class DataConfig:
    """Paths and quality-filter thresholds for the raw operational dataset."""

    FILE_PATH = _get_str('DATA_FILE_PATH', 'data/Aframax/P data_20200213-20200726_Democritos.csv')
    TARGET_COLUMN = _get_str('TARGET_COLUMN', 'Power')
    # Columns to drop before training (e.g. raw timestamps already extracted elsewhere).
    DROP_COLUMNS = _get_list('DROP_COLUMNS', 'TIME')
    TIME_COLUMN = _get_str('TIME_COLUMN', 'TIME')
    # Rows below these thresholds are harbour/manoeuvring noise, not ocean passages.
    MIN_POWER = _get_float('MIN_POWER', '1000')     # kW
    MIN_SPEED = _get_float('MIN_SPEED', '4')        # knots
    TEST_SIZE = _get_float('TEST_SIZE', '0.2')
    RANDOM_STATE = _get_int('RANDOM_STATE', '42')


class ColumnConfig:
    """Canonical column names for the features referenced by the physics layers."""

    SPEED = _get_str('SPEED_COLUMN', 'Speed-Through-Water')
    WAVE_HEIGHT = _get_str('WAVE_HEIGHT_COLUMN', 'SG_Significant_Wave_Height')
    DRAFT_FORE = _get_str('DRAFT_FORE_COLUMN', 'Draft_Fore')
    DRAFT_AFT = _get_str('DRAFT_AFT_COLUMN', 'Draft_Aft')
    HEADING = _get_str('HEADING_COLUMN', 'True_Heading')
    WAVE_DIRECTION = _get_str('WAVE_DIRECTION_COLUMN', 'SG_Mean_Wave_Direction')


class ShipConfig:
    """Hull and propulsive parameters for the resistance calculation (ITTC-78 method).

    Defaults are representative of the Aframax tanker used in the study.
    """

    # Seawater density; 1025 kg/m³ is the standard ITTC value for salt water.
    RHO = _get_float('WATER_DENSITY', '1025.0')          # kg/m³
    S = _get_float('WETTED_SURFACE_AREA', '9950.0')       # m²  — bare hull
    S_APP = _get_float('APPENDAGE_WETTED_SURFACE', '150.0')  # m²  — rudder + bilge keels
    A_T = _get_float('TRANSOM_AREA', '50.0')              # m²  — for transom drag term
    # ITTC-78 roughness/correlation allowance; 4.5×10⁻⁴ is the standard value.
    C_A = _get_float('CORRELATION_ALLOWANCE', '0.00045')
    # Hull form factor (1+k) — accounts for viscous pressure resistance above flat-plate C_F.
    K = _get_float('FORM_FACTOR', '0.15')
    # Base wave-resistance coefficient; multiplied by a trim-correction factor at runtime.
    STWAVE1 = _get_float('BASE_WAVE_RESISTANCE_COEFF', '0.001')
    # Linear trim sensitivity: STWAVE2 = 1 + ALPHA_TRIM * trim (m).
    ALPHA_TRIM = _get_float('TRIM_EFFECT_COEFF', '0.1')
    # Overall propulsive efficiency η_D = η_0 · η_R · η_H; used to convert thrust power to shaft power.
    ETA_D = _get_float('PROPULSIVE_EFFICIENCY', '0.93')
    L = _get_float('SHIP_LENGTH', '230.0')                # m — used in Froude number
    NU = _get_float('KINEMATIC_VISCOSITY', '1e-6')        # m²/s — seawater at ~15 °C
    G = _get_float('GRAVITY', '9.81')                     # m/s²
    L_T = _get_float('TRANSOM_LENGTH', '20.0')            # m — for transom immersion Froude number
    # Displacement mass (kg); used with added-mass coefficient for transient inertia term.
    MASS = _get_float('SHIP_DISPLACEMENT', '115000000.0') # kg
    # Added-mass fraction m' = ADDED_MASS_COEFF * MASS; longitudinal surge added mass ≈ 0.05–0.10.
    ADDED_MASS_COEFF = _get_float('ADDED_MASS_COEFF', '0.1')


class PropellerConfig:
    """Fixed-pitch propeller geometry for the Wageningen B-series KT/KQ polynomials."""

    D = _get_float('PROPELLER_DIAMETER', '6.95')          # m
    Z = _get_int('PROPELLER_BLADES', '5')
    P_D = _get_float('PROPELLER_PITCH_RATIO', '0.7718')   # pitch/diameter ratio
    AE_A0 = _get_float('PROPELLER_AREA_RATIO', '0.52')    # expanded area ratio


class SequenceConfig:
    """Controls how time-series windows are constructed for the LSTM / PI-NODE."""

    LENGTH = _get_int('SEQUENCE_LENGTH', '10')            # time steps per window
    # Sequences that span a gap wider than this are discarded; 3600 s = 1 h covers
    # typical sensor dropouts without allowing cross-voyage contamination.
    MAX_TIME_GAP = _get_float('MAX_TIME_GAP_SECONDS', '3600')  # seconds


class ModelConfig:
    """Architecture defaults for the pure-data LSTM baseline."""

    LSTM_HIDDEN_SIZE = _get_int('LSTM_HIDDEN_SIZE', '64')
    LSTM_NUM_LAYERS = _get_int('LSTM_NUM_LAYERS', '1')
    LSTM_DROPOUT = _get_float('LSTM_DROPOUT', '0.2')


class TrainingConfig:
    """Shared training hyper-parameters used across all model variants."""

    # Shorter epoch budget for cross-validation folds; full training uses EPOCHS_FINAL.
    EPOCHS_CV = _get_int('DEFAULT_EPOCHS_CV', '50')
    EPOCHS_FINAL = _get_int('DEFAULT_EPOCHS_FINAL', '200')
    OPTIMIZER = _get_str('DEFAULT_OPTIMIZER', 'Adam')
    LOSS_FUNCTION = _get_str('DEFAULT_LOSS_FUNCTION', 'MSE')
    WEIGHT_DECAY = _get_float('WEIGHT_DECAY', '1e-5')
    # Generous patience (150 epochs) is intentional: loss curves for physics-hybrid
    # models can plateau for many epochs before resuming descent.
    EARLY_STOPPING_PATIENCE = _get_int('EARLY_STOPPING_PATIENCE', '150')
    EARLY_STOPPING_MIN_DELTA = _get_float('EARLY_STOPPING_MIN_DELTA', '1e-4')
