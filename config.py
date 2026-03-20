import os
from dotenv import load_dotenv

load_dotenv()


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
    FILE_PATH = _get_str('DATA_FILE_PATH', 'data/Aframax/P data_20200213-20200726_Democritos.csv')
    TARGET_COLUMN = _get_str('TARGET_COLUMN', 'Power')
    DROP_COLUMNS = _get_list('DROP_COLUMNS', 'TIME')
    TIME_COLUMN = _get_str('TIME_COLUMN', 'TIME')
    MIN_POWER = _get_float('MIN_POWER', '1000')
    MIN_SPEED = _get_float('MIN_SPEED', '4')
    TEST_SIZE = _get_float('TEST_SIZE', '0.2')
    RANDOM_STATE = _get_int('RANDOM_STATE', '42')


class ColumnConfig:
    SPEED = _get_str('SPEED_COLUMN', 'Speed-Through-Water')
    WAVE_HEIGHT = _get_str('WAVE_HEIGHT_COLUMN', 'SG_Significant_Wave_Height')
    DRAFT_FORE = _get_str('DRAFT_FORE_COLUMN', 'Draft_Fore')
    DRAFT_AFT = _get_str('DRAFT_AFT_COLUMN', 'Draft_Aft')
    HEADING = _get_str('HEADING_COLUMN', 'True_Heading')
    WAVE_DIRECTION = _get_str('WAVE_DIRECTION_COLUMN', 'SG_Mean_Wave_Direction')


class ShipConfig:
    RHO = _get_float('WATER_DENSITY', '1025.0')
    S = _get_float('WETTED_SURFACE_AREA', '9950.0')
    S_APP = _get_float('APPENDAGE_WETTED_SURFACE', '150.0')
    A_T = _get_float('TRANSOM_AREA', '50.0')
    C_A = _get_float('CORRELATION_ALLOWANCE', '0.00045')
    K = _get_float('FORM_FACTOR', '0.15')
    STWAVE1 = _get_float('BASE_WAVE_RESISTANCE_COEFF', '0.001')
    ALPHA_TRIM = _get_float('TRIM_EFFECT_COEFF', '0.1')
    ETA_D = _get_float('PROPULSIVE_EFFICIENCY', '0.93')
    L = _get_float('SHIP_LENGTH', '230.0')
    NU = _get_float('KINEMATIC_VISCOSITY', '1e-6')
    G = _get_float('GRAVITY', '9.81')
    L_T = _get_float('TRANSOM_LENGTH', '20.0')
    MASS = _get_float('SHIP_DISPLACEMENT', '115000000.0')
    ADDED_MASS_COEFF = _get_float('ADDED_MASS_COEFF', '0.1')

class PropellerConfig:
    D = _get_float('PROPELLER_DIAMETER', '6.95')
    Z = _get_int('PROPELLER_BLADES', '5')
    P_D = _get_float('PROPELLER_PITCH_RATIO', '0.7718')
    AE_A0 = _get_float('PROPELLER_AREA_RATIO', '0.52')


class SequenceConfig:
    LENGTH = _get_int('SEQUENCE_LENGTH', '10')
    MAX_TIME_GAP = _get_float('MAX_TIME_GAP_SECONDS', '3600')


class ModelConfig:
    LSTM_HIDDEN_SIZE = _get_int('LSTM_HIDDEN_SIZE', '64')
    LSTM_NUM_LAYERS = _get_int('LSTM_NUM_LAYERS', '1')
    LSTM_DROPOUT = _get_float('LSTM_DROPOUT', '0.2')


class TrainingConfig:
    EPOCHS_CV = _get_int('DEFAULT_EPOCHS_CV', '50')
    EPOCHS_FINAL = _get_int('DEFAULT_EPOCHS_FINAL', '200')
    OPTIMIZER = _get_str('DEFAULT_OPTIMIZER', 'Adam')
    LOSS_FUNCTION = _get_str('DEFAULT_LOSS_FUNCTION', 'MSE')
    WEIGHT_DECAY = _get_float('WEIGHT_DECAY', '1e-5')
    EARLY_STOPPING_PATIENCE = _get_int('EARLY_STOPPING_PATIENCE', '150')
    EARLY_STOPPING_MIN_DELTA = _get_float('EARLY_STOPPING_MIN_DELTA', '1e-4')
