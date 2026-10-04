# PI-NODE: physics-informed shaft-power prediction that transfers between ships

Code for the paper *Physics-Informed Neural ODE with Trainable Wageningen B-Series Propeller
Model for Operational Ship Shaft-Power Prediction and Zero-Shot Cross-Vessel Transfer*.

Most data-driven shaft-power models map sensor readings straight to kilowatts, so they learn one
ship. PI-NODE never outputs power. A small network reads the last ten minutes of operational
records and predicts one quantity, an effective wake fraction `w`. A closed-form propeller layer
turns it into power with the torque characteristic of the vessel's own propeller:

    J = V (1 - w) / (n D)
    P = 2 pi rho n^3 D^5 K_Q(J)

`K_Q(J)` is the Wageningen B-series cubic for the vessel's pitch ratio, blade-area ratio and blade
number (Oosterveld and van Oossanen, 1975), plus a small learned offset. Moving the model to
another ship means replacing four numbers from the propeller drawing; nothing is retrained.

## Results in brief

Trained on one 82,000 DWT bulk carrier, tested on a strictly later period and on four other
vessels (five seeds; full tables in the paper).

| Model | Source vessel, test RMSE (kW) | Zero-shot MAPE on four unseen vessels (%) | With 1 % of target data (%) |
|---|---:|---:|---:|
| **PI-NODE** | **276 ± 14** | **8.0 / 6.6 / 5.8 / 6.5** | **4.9** |
| DATA (MLP) | 296 ± 5 | 6.7 / 19.9 / 42.6 / 82.0 | 13.7 |
| HYBRID (MLP + soft hull physics) | 412 ± 14 | 6.8 / 19.5 / 37.6 / 73.8 | 11.4 |
| PI-KAN (KAN + soft hull physics) | 468 ± 11 | 14.7 / 25.4 / 32.4 / 65.2 | not run |
| DATA-GRU (recurrent black box) | 262 ± 17 | 5.9 / 28.6 / 39.3 / 66.6 | 12.0 |
| B-series torque identity, constant wake, no network | 547 | 9.4 / 5.9 / 6.0 / 3.7 | 5.6 |

Things worth knowing before reusing the model:

- The gain on other ships comes from the propeller layer. The torque identity with a single
  constant wake and no network transfers about as well; the network adds accuracy on the source
  vessel and helps when very little target data are available.
- The ODE-RNN encoder is not better than a plain GRU or than no memory at all. The ablations are
  in the paper.
- On the source vessel a recurrent black box and a ridge regression on physical monomials are as
  accurate as PI-NODE. They fail on other ships.
- With the trainable `K_Q` offset the predicted wake is a model parameter, not a measurement of
  the physical wake fraction.

## Layout

    pinode/
        bseries.py     Wageningen B-series polynomials (39 K_T and 47 K_Q terms)
        data.py        vessel loading, shared feature schema, quality rules, splits
        models.py      PI-NODE (propeller layer, encoders) and the neural baselines
        kan_layer.py   B-spline KAN used by the PI-KAN baseline
        training.py    one training and evaluation protocol for every neural model
        baselines.py   closed-form and classical baselines
    experiments/
        train.py         train one model on the source vessel, score it everywhere
        summarise.py     tables: source accuracy, steady/transient, zero-shot, wake statistics
        fewshot.py       adaptation with 1-25 % of a target vessel's data
        blocked_cv.py    five-fold blocked cross-validation in time
        uncertainty.py   deep-ensemble intervals, temperature scaling, split conformal
        multivessel.py   leave-one-vessel-out training
        search.py        validation-based configuration search
        ode_search.py    training length, history length and size of the temporal encoder
        irregular.py     sparse, irregular windows (exploratory)
        figures.py       every figure of the paper and supplement
    tests/               unit tests
    run_all_v2.sh        reproduce the tables of the paper
    prepare_ship_data.py merge raw logger exports into one file per vessel

## Setup

    pip install -r requirements.txt
    python -m pytest tests -q

## Data

The vessel data used in the paper are confidential and are not part of this repository. To run
the code on your own vessels:

1. Bring each vessel's record to the reference column names (`prepare_ship_data.py` does this for
   two common logger formats). The columns used are `TIME`, `Speed-Through-Water`,
   `Speed-Over-Ground`, `Propeller-Shaft-RPM`, `Shaft Power_TRQM`, `Fore draft_AMS`,
   `Aft draft_AMS`, `Middle draft(P)_AMS`, `Middle draft(S)_AMS`, `Rel-Wind-Speed` and
   `Wind_angle_BRG_WIND`.
2. For each vessel copy `.env.example` to `.env.<name>` and fill in the data path and the
   propeller particulars.
3. Copy `ship_registry.example.py` to `ship_registry.py` and list the source vessel and the
   transfer vessels there.

Rows with shaft power below 1,000 kW, speed below 4 kn or shaft speed below 10 rpm are dropped.
Draft readings outside 3-20 m are treated as missing. Missing inputs are filled with medians of
the source training data.

## Reproducing the paper

    ./run_all_v2.sh                  # all models, five seeds, about two hours on a laptop CPU
    python -m experiments.figures    # figures into results_v2/figures

Single steps:

    python -m experiments.train --baselines
    python -m experiments.train --tag pinode --seed 0
    python -m experiments.summarise
    python -m experiments.fewshot --seed 0
    python -m experiments.blocked_cv --fold 0
    python -m experiments.uncertainty
    python -m experiments.multivessel --holdout C --tag pinode --seed 0

Model tags are listed in `experiments/train.py`: `pinode` is the proposed model, the other
`pinode_*` tags change one component at a time, and `data`, `data_gru`, `hybrid` and `pikan` are
the neural baselines.

## Protocol

- The last 20 % of the source record is the test period and is never used for tuning.
- Validation is every fifth day-long block of the first 80 %. One-minute records are strongly
  autocorrelated and a voyage lasts weeks, so a validation period taken from the end of the
  training record would be a single voyage.
- An epoch draws one window in ten, because consecutive windows share nine of their ten records.
- The learning rate is chosen on validation for every model and seed; training stops after 40
  epochs without improvement.
- Every model sees the same eleven inputs. Position is excluded.

## Licence

MIT, see `LICENSE`.
