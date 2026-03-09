import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from config import DataConfig, ColumnConfig, SequenceConfig


class DataProcessor:
    def __init__(
        self,
        file_path=None,
        target_column=None,
        drop_columns=None,
        test_size=None,
        random_state=None,
        fill_missing_with_median=True,
        exclude_missing_Hs=True,
    ):
        self.file_path = file_path or DataConfig.FILE_PATH
        self.target_column = target_column or DataConfig.TARGET_COLUMN
        self.drop_columns = drop_columns if drop_columns is not None else DataConfig.DROP_COLUMNS
        self.test_size = test_size if test_size is not None else DataConfig.TEST_SIZE
        self.random_state = random_state if random_state is not None else DataConfig.RANDOM_STATE
        self.fill_missing_with_median = fill_missing_with_median
        self.exclude_missing_Hs = exclude_missing_Hs
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.df = None

    def _read_input_file(self):
        file_path = Path(self.file_path)
        suffix = file_path.suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(self.file_path)
        return pd.read_csv(self.file_path)

    def _drop_all_nan_numeric_columns(self):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        all_nan_numeric = [col for col in numeric_cols if self.df[col].isna().all()]
        if all_nan_numeric:
            self.df.drop(columns=all_nan_numeric, inplace=True)
            print(f"Dropped all-NaN numeric columns: {all_nan_numeric}")

    def load_and_prepare_data(self):
        try:
            self.df = self._read_input_file()
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

        if self.drop_columns is not None:
            for col in self.drop_columns:
                if col in self.df.columns:
                    self.df.drop(columns=col, inplace=True)
                else:
                    print(f"Warning: Column '{col}' not found in the dataset.")

        if self.target_column not in self.df.columns:
            print(f"Error: Target column '{self.target_column}' not found in the dataset.")
            return None

        if self.exclude_missing_Hs:
            wave_col = ColumnConfig.WAVE_HEIGHT
            if wave_col in self.df.columns:
                initial_row_count = len(self.df)
                self.df = self.df.dropna(subset=[wave_col])
                rows_dropped = initial_row_count - len(self.df)
                print(f"Dropped {rows_dropped} rows due to missing '{wave_col}'")
            else:
                print(f"Warning: '{wave_col}' column not found in the dataset.")

        self.df = self.df[self.df[self.target_column] >= DataConfig.MIN_POWER]

        speed_col = ColumnConfig.SPEED
        if speed_col in self.df.columns:
            self.df = self.df[self.df[speed_col] >= DataConfig.MIN_SPEED]

        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self._drop_all_nan_numeric_columns()
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if self.fill_missing_with_median:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[[self.target_column]]

        X_unscaled = X.copy()
        y_unscaled = y.copy()

        X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(
            X_unscaled, y_unscaled,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False,
        )

        self.scaler_X.fit(X_train_unscaled)
        X_train_scaled = self.scaler_X.transform(X_train_unscaled)
        X_test_scaled = self.scaler_X.transform(X_test_unscaled)

        self.scaler_y.fit(y_train_unscaled)
        y_train_scaled = self.scaler_y.transform(y_train_unscaled)
        y_test_scaled = self.scaler_y.transform(y_test_unscaled)

        X_train = pd.DataFrame(X_train_scaled, columns=X_unscaled.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_unscaled.columns)

        y_train = pd.Series(y_train_scaled.flatten(), name=self.target_column)
        y_test = pd.Series(y_test_scaled.flatten(), name=self.target_column)

        return (
            X_train, X_test,
            X_train_unscaled, X_test_unscaled,
            y_train, y_test,
            y_train_unscaled, y_test_unscaled,
        )

    def print_dataset_shapes(self, X_train, X_test):
        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")

    def print_dataset_head(self, X_train, X_test):
        print("First few rows of training features (X_train):")
        print(X_train.head())
        print("\nFirst few rows of test features (X_test):")
        print(X_test.head())

    def list_column_names(self):
        if self.df is None:
            try:
                self.df = self._read_input_file()
                if self.drop_columns is not None:
                    self.df.drop(columns=self.drop_columns, inplace=True, errors='ignore')
            except FileNotFoundError:
                print(f"File not found: {self.file_path}")
                return None
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")
                return None

        columns = self.df.columns.tolist()
        print("Column names:")
        for col in columns:
            print(col)
        return columns

    def inverse_transform_y(self, y_scaled):
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    def load_and_prepare_temporal_data(self):
        """Load data keeping TIME for dt/acceleration computation.

        Returns the same 8-tuple as load_and_prepare_data, plus:
            dt_train, dt_test : arrays of time deltas (seconds) between consecutive rows
            accel_train, accel_test : arrays of dV/dt (m/s^2)
        """
        try:
            self.df = self._read_input_file()
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

        time_col = DataConfig.TIME_COLUMN
        if time_col not in self.df.columns:
            print(f"Error: Time column '{time_col}' not found. Cannot build temporal data.")
            return None

        self.df[time_col] = pd.to_datetime(self.df[time_col])
        self.df = self.df.sort_values(by=time_col).reset_index(drop=True)

        # Drop non-TIME columns that were requested
        if self.drop_columns is not None:
            for col in self.drop_columns:
                if col == time_col:
                    continue  # keep TIME for now
                if col in self.df.columns:
                    self.df.drop(columns=col, inplace=True)

        if self.target_column not in self.df.columns:
            print(f"Error: Target column '{self.target_column}' not found.")
            return None

        if self.exclude_missing_Hs:
            wave_col = ColumnConfig.WAVE_HEIGHT
            if wave_col in self.df.columns:
                initial = len(self.df)
                self.df = self.df.dropna(subset=[wave_col])
                print(f"Dropped {initial - len(self.df)} rows due to missing '{wave_col}'")

        self.df = self.df[self.df[self.target_column] >= DataConfig.MIN_POWER]

        speed_col = ColumnConfig.SPEED
        if speed_col in self.df.columns:
            self.df = self.df[self.df[speed_col] >= DataConfig.MIN_SPEED]

        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        self._drop_all_nan_numeric_columns()
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if self.fill_missing_with_median:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        else:
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())

        # Compute dt (seconds) and acceleration dV/dt (m/s^2)
        knots_to_ms = 0.51444
        time_series = self.df[time_col]
        dt_seconds = time_series.diff().dt.total_seconds().values
        dt_seconds[0] = dt_seconds[1] if len(dt_seconds) > 1 else 1.0

        V_ms = self.df[speed_col].values * knots_to_ms
        dV = np.diff(V_ms, prepend=V_ms[0])
        dt_safe = np.where(dt_seconds > 0, dt_seconds, 1.0)
        acceleration = dV / dt_safe

        self.df['dt'] = dt_seconds
        self.df['acceleration'] = acceleration

        # Now drop the TIME column (we've extracted what we need)
        self.df.drop(columns=[time_col], inplace=True)

        X = self.df.drop(self.target_column, axis=1)
        y = self.df[[self.target_column]]

        X_unscaled = X.copy()
        y_unscaled = y.copy()

        X_train_unscaled, X_test_unscaled, y_train_unscaled, y_test_unscaled = train_test_split(
            X_unscaled, y_unscaled,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=False,
        )

        self.scaler_X.fit(X_train_unscaled)
        X_train_scaled = self.scaler_X.transform(X_train_unscaled)
        X_test_scaled = self.scaler_X.transform(X_test_unscaled)

        self.scaler_y.fit(y_train_unscaled)
        y_train_scaled = self.scaler_y.transform(y_train_unscaled)
        y_test_scaled = self.scaler_y.transform(y_test_unscaled)

        X_train = pd.DataFrame(X_train_scaled, columns=X_unscaled.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X_unscaled.columns)

        y_train = pd.Series(y_train_scaled.flatten(), name=self.target_column)
        y_test = pd.Series(y_test_scaled.flatten(), name=self.target_column)

        return (
            X_train, X_test,
            X_train_unscaled, X_test_unscaled,
            y_train, y_test,
            y_train_unscaled, y_test_unscaled,
        )


def create_sequences(X_scaled, X_unscaled, y_scaled, seq_length=None, max_gap=None):
    """Build overlapping sequences for LSTM input.

    Skips sequences that span a time gap larger than max_gap seconds.

    Args:
        X_scaled: DataFrame of scaled features (includes 'dt' and 'acceleration')
        X_unscaled: DataFrame of unscaled features (includes 'dt' and 'acceleration')
        y_scaled: Series of scaled target
        seq_length: number of time steps per sequence
        max_gap: maximum allowed dt (seconds) within a sequence

    Returns:
        X_seq: np.array (N_seq, seq_length, n_features)
        X_unscaled_seq: np.array (N_seq, seq_length, n_features)
        y_seq: np.array (N_seq,) - target at the last step of each sequence
    """
    if seq_length is None:
        seq_length = SequenceConfig.LENGTH
    if max_gap is None:
        max_gap = SequenceConfig.MAX_TIME_GAP

    X_vals = X_scaled.values
    X_uns_vals = X_unscaled.values
    y_vals = y_scaled.values

    dt_col_idx = list(X_unscaled.columns).index('dt')

    X_seq, X_uns_seq, y_seq = [], [], []

    for i in range(len(X_vals) - seq_length + 1):
        window_dt = X_uns_vals[i:i + seq_length, dt_col_idx]

        if np.any(window_dt > max_gap):
            continue

        X_seq.append(X_vals[i:i + seq_length])
        X_uns_seq.append(X_uns_vals[i:i + seq_length])
        y_seq.append(y_vals[i + seq_length - 1])

    return np.array(X_seq), np.array(X_uns_seq), np.array(y_seq)


if __name__ == "__main__":
    data_processor = DataProcessor()

    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, *_ = result
        data_processor.print_dataset_shapes(X_train, X_test)

    print("\n--- Temporal data test ---")
    data_processor_temporal = DataProcessor()
    result_t = data_processor_temporal.load_and_prepare_temporal_data()
    if result_t is not None:
        X_train_t, X_test_t, X_train_uns, X_test_uns, \
            y_train_t, y_test_t, _, _ = result_t
        print(f"Temporal X_train shape: {X_train_t.shape}")
        print(f"Columns: {list(X_train_t.columns)}")

        X_seq, X_uns_seq, y_seq = create_sequences(X_train_t, X_train_uns, y_train_t)
        print(f"Sequences: {X_seq.shape} -> targets: {y_seq.shape}")
