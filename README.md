# Ship Shaft-Power Prediction with a Physics-Informed Neural ODE (PI-NODE)

Predicting a ship's **shaft power** from operational time-series (speed, drafts, trim, shaft RPM,
weather) matters for voyage optimisation, fuel and emissions management, and condition monitoring.
This repository implements and evaluates **PI-NODE**, a grey-box model. Instead of leaning on the
noisy empirical hull-resistance formulas that most physics-informed approaches use, it constrains
the network with the rigid laws of **propeller hydrodynamics**.

The main result of the project is straightforward. Pushing empirical *hull*-resistance physics into
a model trained on real sea states does not help; it ends up no better than a plain data-driven
network. Moving the physics to the *propeller* is a different story: it improves accuracy on the
source vessel and lets the model transfer to ships it has never seen, without any retraining.

---

## 1. The core idea: why the propeller?

Most physics-informed power models inject the **entire ship resistance** (frictional, wave-making,
air) into the loss. The empirical hull formulas behind that resistance carry roughly 10 to 15 %
error in real sea states, so a network forced to match them inherits the same error. That is why
the soft-hull-physics baseline (HYBRID) ends up no better than the pure data-driven model.

PI-NODE puts the physical constraint on the **propeller** instead. The propeller obeys much
stricter kinematic and dynamic laws (the Wageningen B-series), and the network only has to act as a
**virtual sensor** for the three propulsive factors that nobody measures directly:

- **wake fraction** `w`
- **thrust-deduction factor** `t`
- **relative rotative efficiency** `η_R`

Those three feed analytically exact propeller equations that compute the power, so the physics, not
the network, has the last word.

---

## 2. Architecture

The model splits the problem into two physical regimes, so that slow-varying hull behaviour and
fast-varying weather disturbances do not get tangled together.

**Branch A: calm-water dynamics (Neural ODE).**
The inputs here are the calm-water channels (speed-through-water, fore/aft draft, trim, shaft RPM,
and the slow navigational signals); weather is deliberately left out. An encoder feeds a Neural ODE
(`torchdiffeq`) whose latent state is integrated over the operational window and then decoded,
which captures the inertia and transients a static MLP cannot. The branch outputs the calm-water
factors `[w, t, η_R]`, bounded by a sigmoid.

**Branch B: sea-state residual (weather MLP).**
The inputs are the relative/true wind channels. A small feed-forward network outputs instantaneous
residual corrections `[Δw, Δt, Δη_R]`.

**Aggregation and analytical physics layer.**

```
w_total   = w + Δw       (clamped to physical bounds)
t_total   = t + Δt
η_R_total = η_R + Δη_R

Va = V_ship · (1 − w_total)          # advance speed
J  = Va / (n · D)                    # advance coefficient
K_T(J) = b0 + b1·J + b2·J² + b3·J³   # trainable, initialised at B-series values
K_Q(J) = c0 + c1·J + c2·J² + c3·J³   # trainable, initialised at B-series values
Q = K_Q · ρ · n² · D⁵                # torque
P = 2π · n · Q                       # shaft power
```

The `K_T` / `K_Q` coefficients are **trainable**, which lets the model absorb long-term propeller
degradation (biofouling). Physics-informed penalties keep them realistic:

```
L = SmoothL1(P_pred, P_true)
  + λ_range     · keep K_T, K_Q positive and within bounds
  + λ_curvature · penalise the 2nd derivative (no unphysical oscillation)
  + λ_prior     · keep coefficients near the B-series initialisation
  + λ_η0        · keep open-water efficiency η0 = J·K_T / (2π·K_Q) in [0.40, 0.75]
```

Training uses `ReduceLROnPlateau`, gradient clipping (max-norm 5.0), and a 1.5× learning-rate
boost on the polynomial coefficients. For **zero-shot transfer** to a new vessel you only swap the
propeller constants `D`, `P/D`, `Z`; the network weights stay frozen and the analytical layer
adapts to the new geometry.

---

## 3. Models in this repository

| Model | File | Description |
|---|---|---|
| **PI-NODE** | `main_PI_NODE_Propeller.py` | The grey box: Neural-ODE encoder + trainable propeller physics. |
| **DATA** | `main_DATA.py` | Pure data-driven MLP (black-box reference). |
| **HYBRID** | `main_HYBRID.py` | MLP with soft ITTC-78 hull-resistance penalties (the soft-hull-physics baseline). |
| **PI-KAN** | `main_PI_KAN.py`, `kan_layer.py` | A Kolmogorov-Arnold-Network backbone with the same soft physics as HYBRID, used as a strong head-to-head competitor. |

DATA and HYBRID share the MLP defined in `base_model.py`. PI-KAN subclasses HYBRID and swaps only
the backbone, so the comparison isolates the architecture.

---

## 4. Repository structure

```
.
├── config.py                       # Typed configuration classes, loaded from the active .env
├── read_data.py                    # DataProcessor: loading, filtering, leakage-safe scaling, sequencing
├── base_model.py                   # Shared MLP backbone + common training infrastructure
├── pinode_common.py                # Shared PI-NODE helpers (sequence loading, loaders, metrics)
│
│  # Models
├── main_DATA.py                    # Data-driven MLP baseline
├── main_HYBRID.py                  # Soft-physics (ITTC-78) MLP baseline
├── main_PI_NODE_Propeller.py       # PI-NODE: Neural ODE + trainable propeller physics
├── kan_layer.py                    # Self-contained B-spline Kolmogorov-Arnold Network
├── main_PI_KAN.py                  # PI-KAN: HYBRID physics with a KAN backbone
│
│  # Data preparation and training
├── prepare_ship_data.py            # Merge raw per-vessel logs into the common schema
├── sweep_pinode_regularization.py  # Grid search over the physics-regularisation weights
├── train_final_pinode.py           # Train the final PI-NODE, save weights + fitted scaler
├── train_multiseed.py              # Train PI-NODE under several seeds (CI + deep-ensemble members)
├── train_multiseed_pikan.py        # Multi-seed training for the PI-KAN baseline
├── finalize_frozen.py              # Finalise an ablation variant from its best checkpoint
│
│  # Evaluation
├── evaluate_baselines.py           # DATA vs HYBRID on the source vessel
├── evaluate_pikan.py               # Single-run PI-KAN train + evaluation
├── evaluate_transient.py           # Steady vs transient regime comparison
├── evaluate_transfer.py            # Zero-shot transfer to an unseen vessel
├── evaluate_fewshot.py             # Few-shot fine-tuning across data budgets
├── ablation_study.py               # Component ablations of the PI-NODE
├── evaluate_uncertainty.py         # MC-Dropout and deep-ensemble predictive uncertainty
├── calibrate_uncertainty.py        # Post-hoc recalibration (temperature scaling + split conformal)
├── make_figures.py                 # Generate the publication figures (F1 to F10)
│
├── tests/                          # Standalone unit/regression tests (no pytest required)
├── .env.example                    # Template config; copy to .env and fill in per-vessel values
├── requirements.txt                # Python dependencies
├── results/                        # Checkpoints, CSV outputs, and figures (generated; not tracked)
└── PhD/                            # Raw vessel logs and technical drawings
```

Training runs save the model checkpoints (`best_model_*.pt`) and the fitted scalers
(`data_processor_*.pkl`) to the repository root, and the multi-seed run writes the per-seed
checkpoints `results/best_model_PI_NODE_seed{0..4}.pt` to `results/`.

---

## 5. Installation

Python 3.12. Install the dependencies into an isolated environment:

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The main dependencies are PyTorch, `torchdiffeq` (the Neural-ODE solver), scikit-learn, pandas,
matplotlib, and `python-dotenv`. Training picks the device automatically, trying CUDA first, then
MPS, then CPU. PI-KAN is the one exception: it runs on CPU, because the Apple-MPS backend
miscomputes the second-order automatic differentiation through the B-spline basis.

---

## 6. Configuration

All tunable parameters live in `.env` files read by `config.py`. Start from the template, create
one file per vessel, then switch the active vessel by copying its file over `.env`:

```bash
cp .env.example .env.source    # then edit it with the vessel's paths and constants
cp .env.source .env            # make the source vessel active

cp .env.target .env            # switch to another vessel
cp .env.source .env            # restore the source vessel
```

Environment files are kept out of version control (they hold local paths and per-vessel constants);
only `.env.example` is tracked.

The parameters fall into a few groups: **data** (`DATA_FILE_PATH`, `TARGET_COLUMN`, `DROP_COLUMNS`,
`MIN_POWER`, `MIN_SPEED`), **column names**, **ship constants** (density, length, displacement,
propulsive efficiency, and so on), **propeller constants** (diameter, blade number, pitch ratio,
area ratio), **sequencing** (`SEQUENCE_LENGTH`, default 10 timesteps; `MAX_TIME_GAP_SECONDS`), and
**training** (epochs, optimiser, early-stopping patience).

**Leakage prevention.** Columns that are derived directly from the target (shaft torque, shaft
thrust, propeller-shaft-power, main-engine RPM) are excluded through `DROP_COLUMNS`. PI-NODE puts
back only `Propeller-Shaft-RPM`, which it needs as a physical input to the propeller law; that
channel is available to every model.

---

## 7. Dataset

The source (training) vessel is an 82,000 DWT bulk carrier, with roughly 120,000 usable records
spanning several months of in-service operation. The target is shaft power (kW) from the onboard
torque meter.

Preprocessing happens in `read_data.py`. Port and anchor rows are filtered out (power below 1000 kW
or speed below 4 knots). The data is split 80/20 **chronologically**, with no shuffling. Missing
values are filled with column medians **computed on the training split only**, and features and
target are standardised with a `StandardScaler` that is also fitted on the training split only. For
PI-NODE, overlapping windows of 10 consecutive timesteps are built by `create_sequences`. The
derived `dt` and `dV/dt` (acceleration) are used for sequence gap-detection and for the
transient-regime split, but they are **not** fed to the models as inputs.

---

## 8. Reproducing the pipeline

Run from the project root with the source vessel active (`cp .env.source .env`). The steps are
ordered, since the later ones depend on the checkpoints produced earlier.

**Baselines (DATA and HYBRID).**
```bash
python -u evaluate_baselines.py
```

**PI-NODE: tune and train.**
```bash
python -u sweep_pinode_regularization.py    # search the regularisation weights (auto-resumes via results/ CSV)
python -u train_final_pinode.py             # train the selected configuration, save weights + scaler
```

**Robustness across operating regimes.**
```bash
python -u evaluate_transient.py             # steady vs transient
```

**Zero-shot transfer to an unseen vessel.**
```bash
cp .env.target .env
python -u evaluate_transfer.py
cp .env.source .env
```
Only the propeller constants change; the weights stay frozen.

**Few-shot adaptation.**
```bash
cp .env.target .env
python -u evaluate_fewshot.py
cp .env.source .env
```

**Ablation, multi-seed, and uncertainty.**
```bash
python -u ablation_study.py                 # component ablations
python -u train_multiseed.py --seeds 0 1 2 3 4   # confidence interval + ensemble members
python -u evaluate_uncertainty.py --ensemble     # MC-Dropout and deep-ensemble UQ
python -u calibrate_uncertainty.py          # temperature scaling + split conformal recalibration
```

**Head-to-head competitor (PI-KAN).**
```bash
python -u train_multiseed_pikan.py --seeds 0 1 2 3 4
```

**Figures.**
```bash
python -u make_figures.py                   # writes results/figures/F1 to F10
```

**Adding a new vessel.**
```bash
python prepare_ship_data.py <vessel>        # merge raw logs into the common schema
```

---

## 9. Results

All figures are written to `results/figures/` (`F1` to `F10`).

**Source domain (source vessel).**

| Model | Test RMSE (kW) | vs DATA | Description |
|---|---:|---|---|
| **PI-NODE** | **312.52** | **−43.9 %** | Grey box: Neural ODE + propeller physics |
| PI-KAN | 471.04 ± 72.80 | −15.5 % | KAN backbone + soft hull-physics, 5-seed |
| DATA | 557.52 | baseline | Black-box MLP |
| HYBRID | 583.88 | +4.7 % | Soft-physics MLP (ITTC-78 penalties) |

The PI-NODE multi-seed estimate is **286.42 ± 9.89 kW** over five seeds (coefficient of variation
about 3.5 %); the 312.52 figure is the single canonical run. DATA and HYBRID are single runs under
the same protocol. The HYBRID number is the key negative finding: soft hull-physics buys nothing
over plain data, which is exactly what motivates moving the physics to the propeller.

**Head-to-head vs PI-KAN.** Trained on the source vessel under an identical protocol (same split,
features, physics, and five seeds), PI-KAN reaches 471.04 ± 72.80 kW. It is a serious competitor,
since it beats both DATA and HYBRID, but PI-NODE still wins by **39 % in RMSE** and is about
**seven times more stable across seeds**. The Camp-A "B-series-ML" surrogates from the literature
predict open-water `K_T`/`K_Q` rather than operational power, so they belong here as a contextual
contrast rather than a trained baseline.

**Zero-shot transfer (trained on the source vessel, no retraining), MAPE.**

| Target vessel | PI-NODE | DATA | HYBRID |
|---|---:|---:|---:|
| Sister vessel (82K) | **3.75 %** | 9.47 % | 23.46 % |
| Smaller bulk carrier (64K) | **4.87 %** | 39.72 % | 35.05 % |
| Different class | **27.72 %** | 41.55 % | 41.07 % |
| Different vessel (75.2K) | **32.19 %** | 88.63 % | 77.69 % |

PI-NODE has the lowest error on every unseen vessel. The two highest numbers come from the vessels
that are furthest out of distribution; PI-NODE stays first there but is not immune.

**Summary across studies.**

| Study | Key finding |
|---|---|
| Source domain | PI-NODE 286.42 ± 9.89 kW RMSE (about 3.6 % MAPE), 43.9 % better than DATA and better than HYBRID. |
| Transient | PI-NODE holds about 3.5 to 3.9 % MAPE across steady and transient regimes; the baselines degrade most in transients. |
| Zero-shot | Lowest MAPE on every unseen vessel; range 3.75 to 32 %. |
| Few-shot | PI-NODE reaches about 3 to 4 % MAPE from as little as 1 % of target data; the data-driven model needs far more. |
| Ablation | Freezing the learnable propeller polynomials triples the error (312 to 941 kW), so the trainable B-series is the decisive component. |
| Multi-seed | 286.42 ± 9.89 kW over five seeds, a robust result rather than a single fortunate run. |
| Uncertainty | The deep ensemble is the best estimator (278.4 kW). Raw intervals are over-confident (51.4 % coverage at the 95 % level), but split conformal restores about 90/95 % coverage on a held-out half and stays well-behaved on MC-Dropout, where Gaussian temperature scaling breaks down. |

The pattern across all of these is the same: the physics-informed architecture pays off most on
unseen vessels and in the low-data regime, which is where the baselines struggle.

---

## 10. Tests

The `tests/` directory holds standalone scripts (no test framework required). Run any of them
directly, for example:

```bash
python tests/test_kan_layer.py
python tests/test_calibration.py
```

Each prints a pass line on success and raises on failure.

---

## 11. Acknowledgments

Special thanks to Christoforos Rekatsinas (Ph.D.) for his guidance and support.

## 12. Contact

- Alexiou Kiriakos
- Email: kiriakosal2004@yahoo.gr
