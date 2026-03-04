import matplotlib.pyplot as plt
import numpy as np

from config import ShipConfig, ColumnConfig
from read_data import DataProcessor


def calculate_shaft_power(V_knots, trim):
    """Calculate shaft power from speed and trim using ship resistance equations."""
    ship = ShipConfig

    V = V_knots * 0.51444
    V = np.clip(V, 1e-5, None)

    Re = np.clip(V * ship.L / ship.NU, 1e-5, None)
    C_f = 0.075 / (np.log10(Re) - 2) ** 2

    R_F = 0.5 * ship.RHO * V**2 * ship.S * C_f

    STWAVE2 = 1 + ship.ALPHA_TRIM * trim
    C_W = ship.STWAVE1 * STWAVE2
    R_W = 0.5 * ship.RHO * V**2 * ship.S * C_W

    R_APP = 0.5 * ship.RHO * V**2 * ship.S_APP * C_f

    F_nt = V / np.sqrt(ship.G * ship.L_T)
    R_TR = 0.5 * ship.RHO * V**2 * ship.A_T * (1 - F_nt)

    R_C = 0.5 * ship.RHO * V**2 * ship.S * ship.C_A

    R_T = R_F * (1 + ship.K) + R_W + R_APP + R_TR + R_C

    P_S = ((V * R_T) / ship.ETA_D) / 1000
    return P_S


if __name__ == "__main__":
    data_processor = DataProcessor()

    result = data_processor.load_and_prepare_data()
    if result is not None:
        (X_train, X_test, X_train_unscaled, X_test_unscaled,
         y_train, y_test, y_train_unscaled, y_test_unscaled) = result

        V_knots = X_train_unscaled[ColumnConfig.SPEED].values
        trim = (X_train_unscaled[ColumnConfig.DRAFT_FORE].values
                - X_train_unscaled[ColumnConfig.DRAFT_AFT].values)

        P_S = calculate_shaft_power(V_knots, trim)

        plt.figure(figsize=(12, 6))
        plt.plot(y_train_unscaled.values, label='Original Power (CSV)', color='blue')
        plt.plot(P_S, label='Calculated Shaft Power', color='green')
        plt.title('Original Power vs. Calculated Shaft Power')
        plt.xlabel('Index')
        plt.ylabel('Power (kW)')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Failed to load and prepare data.")
