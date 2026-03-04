# Predicting Ship Propulsion Power using Physics-Guided Neural Networks (PGNN) and Physics-Informed Neural Networks (PINN)

## Introduction

This project explores the use of Physics-Guided Neural Networks (PGNNs) and Physics-Informed Neural Networks (PINNs) to predict ship propulsion power. By incorporating physical laws into the neural network training process, we aim to improve the model's predictive capabilities and assess whether PGNNs/PINNs offer advantages over purely data-driven models.

## Repository Structure

```
├── .env                # All configurable parameters (ship constants, paths, training defaults)
├── config.py           # Loads .env into typed config classes
├── base_model.py       # Shared NN architecture and training infrastructure
├── read_data.py        # Data loading, preprocessing, and scaling
├── main_DATA.py        # Data-driven neural network model
├── main_PGNN.py        # Physics-Guided Neural Network (ship resistance loss)
├── main_PINN.py        # Physics-Informed Neural Network (PDE residuals + boundary conditions)
├── power_charts.py     # Physics validation: calculated vs. actual power plots
├── requirements.txt    # Python dependencies
└── data/               # Ship operational data (not tracked in git)
```

## Features

- **Data Preprocessing**: Handles missing values and scales features using StandardScaler.
- **Data-Driven Model**: A multi-layer neural network trained solely on data to predict propulsion power.
- **Physics-Guided Neural Network**: Enhances the data-driven model by adding a physics-based loss term derived from ship resistance equations.
- **Physics-Informed Neural Network**: Enhances the data-driven model by adding a PDE loss term derived from ship resistance partial derivative equations.
- **Hyperparameter Tuning**: Uses grid search and k-fold cross-validation to find optimal learning rates and batch sizes.
- **Model Evaluation**: Provides training and validation loss during training and evaluates the final model on a test set.
- **Centralized Configuration**: All ship constants, data paths, column names, and training defaults are configured via a `.env` file.

## Requirements

Python 3.12.2. It is advised to run under a virtual environment created with `requirements.txt` to ensure trouble-free execution.

## Installation

1. Clone the repository:

        git clone https://github.com/kiriakosal2017-dot/thesis_PINN.git

2. Set up a virtual environment (optional but recommended):

        python -m venv venv
        source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install dependencies:

        pip install -r requirements.txt

## Configuration

All configurable parameters are in the `.env` file at the project root. Edit it to match your setup:

- **Data paths**: `DATA_FILE_PATH`, `TARGET_COLUMN`, `DROP_COLUMNS`
- **Data filtering**: `MIN_POWER`, `MIN_SPEED`
- **Column names**: `SPEED_COLUMN`, `WAVE_HEIGHT_COLUMN`, `DRAFT_FORE_COLUMN`, etc.
- **Ship constants**: `WATER_DENSITY`, `WETTED_SURFACE_AREA`, `SHIP_LENGTH`, etc.
- **Training defaults**: `DEFAULT_EPOCHS_CV`, `DEFAULT_EPOCHS_FINAL`, `DEFAULT_OPTIMIZER`, etc.

## Data Preparation

Ensure that you have the required CSV data files in the `data/` directory. The data path is configured in `.env` via the `DATA_FILE_PATH` variable.

## Usage

### Data Preprocessing

        python read_data.py

This will load the dataset, handle missing values (fill with median), filter by power/speed thresholds, scale features, and split into training/testing sets.

### Physics Validation

        python power_charts.py

Plots calculated shaft power (from resistance equations) against actual power from the dataset for visual comparison.

### Running the Data-Driven Model

        python main_DATA.py

Performs hyperparameter tuning (learning rate and batch size) using k-fold cross-validation, trains the final model with the best hyperparameters, and evaluates on the test set (RMSE).

### Running the Physics-Guided Neural Network (PGNN)

        python main_PGNN.py

Incorporates ship resistance equations (frictional, wave-making, appendage, transom, correlation allowance, and added wave resistance) into the loss function. Tunes `alpha`, `beta`, and `k_wave` alongside learning rate and batch size.

### Running the Physics-Informed Neural Network (PINN)

        python main_PINN.py

Incorporates PDE residuals using automatic differentiation and boundary conditions (P=0 when V=0) into the loss. Tunes `alpha`, `beta`, and `gamma` alongside learning rate and batch size.

## Formulation of a PDE for the Problem (PINN)

Assuming we can model the resistance R as a function of speed V and other variables, we consider a PDE:

![image](https://github.com/user-attachments/assets/c72f77d4-7d20-4d31-920c-0dc76dcc7fec)

Where:
- P is the Power (the target variable of the data model).
- V is the Speed-Through-Water.
- a and b are constants derived from physical considerations.

## Physics-Based Loss Function (PGNN)

The PGNN incorporates a physics-based loss term calculated using ship resistance equations:

- Frictional Resistance

![image](https://github.com/user-attachments/assets/7ff9b3d1-fb57-4cf5-a885-3f4148322d84)

It is calculated using the ITTC-1957 formula, the frictional resistance coefficient accounts for the friction between the ship's hull and the water.

- Wave-Making Resistance

  ![image](https://github.com/user-attachments/assets/5dbf4b12-984d-4e44-8a36-ad1290fcdb90)

Wave-making resistance is caused by the energy lost in generating waves as the ship moves through the water. It is influenced by the ship's speed and trim.

- Appendage Resistance

![image](https://github.com/user-attachments/assets/99d428c4-6b61-4697-9a93-ebfde8f9ca95)

Appendage resistance accounts for the additional resistance from appendages such as rudders, shafts, and propellers.

- Transom Stern Resistance

![image](https://github.com/user-attachments/assets/9e77f536-ec0c-488f-a6fb-bb5ce40776f0)

Transom stern resistance is significant for ships with a flat stern (transom) and is calculated using the transom Froude number.

- Correlation Allowance Resistance

![image](https://github.com/user-attachments/assets/21007a57-e06d-407b-8fb2-d0b841bcad94)

This is an empirical correction factor to account for additional resistance not captured by other components.

- Added Resistance due to waves

![image](https://github.com/user-attachments/assets/c84ba982-d45f-4eed-af88-1195b8d277b1)

Where:

![image](https://github.com/user-attachments/assets/7ff1d96c-18eb-426d-8bfe-3e72edc4a105)

and:

![image](https://github.com/user-attachments/assets/dd9a3f4d-2fb0-4d84-875e-ce887a87d553)

![image](https://github.com/user-attachments/assets/13c194a4-48f0-4dc9-b7f7-1bd48662a21c)

Finally the physics-based loss is computed as the squared difference between the predicted power and the power calculated using the total resistance and propulsive efficiency:

![Screenshot_2](https://github.com/user-attachments/assets/88e88f14-73b3-4232-92ed-693ce98a8c87)

## Hyperparameter Tuning

All three models perform hyperparameter tuning over a predefined grid using k-fold cross-validation (default is 5 folds). Training defaults (epochs, optimizer, loss function) are configured in `.env`.

## Results

- **Training Loss**: Displayed during each epoch for all models.
- **Validation Loss**: Reported during hyperparameter tuning for each parameter combination.
- **Test Loss**: Final evaluation metric on the test set.

Compare the test losses of all three models to assess the impact of incorporating physics into the model.

## Acknowledgments

- Special thanks to Christoforos Rekatsinas (Ph.D.) for his guidance and support.

## Contact

For any questions or inquiries, please contact:

- Alexiou Kiriakos
- Email: kiriakosal2004@yahoo.gr
