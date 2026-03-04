import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import DataConfig, ColumnConfig


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

    def load_and_prepare_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
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
                self.df = pd.read_csv(self.file_path)
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


if __name__ == "__main__":
    data_processor = DataProcessor()

    result = data_processor.load_and_prepare_data()
    if result is not None:
        X_train, X_test, *_ = result
        data_processor.print_dataset_shapes(X_train, X_test)
