# Ship Shaft Power Prediction with a Physics-Informed Neural ODE (PI-NODE)

## Introduction

This project predicts a vessel's **shaft power** from real-world operational time-series data (speed, drafts, trim, RPM, weather). Its core contribution is a **grey-box** architecture — the **Physics-Informed Neural ODE (PI-NODE)** — that constrains a neural network with the rigid laws of **propeller hydrodynamics** instead of the noisy empirical hull-resistance formulas (ITTC-78) commonly used in the literature.

The PI-NODE is benchmarked against two established baselines:

- **DATA** — a pure data-driven MLP (black box).
- **HYBRID** — a soft-physics PINN that penalizes deviations from ITTC-78 resistance formulas.

The central finding is that forcing empirical *hull*-resistance physics onto a model in real sea states **degrades** accuracy, whereas constraining the model with accurate *propeller* physics acts as a powerful inductive bias that improves accuracy **and** enables zero-shot transfer to unseen vessels.

> **Full theory and experimental record** live in `docs/`:
> - `docs/PI_NODE_THEORY_AND_ARCHITECTURE.md` — architecture, equations, training infrastructure.
> - `docs/EXPERIMENTAL_PLAN.md` — dataset, methodology, and all results (Phases 1–5).
> - `docs/PHYSICS_EVALUATION_AND_LLM_PROMPT.md` — positioning vs. recent literature.

## Why the Propeller? (The Core Idea)

Most physics-informed approaches inject the **entire ship resistance** (frictional, wave-making, air) into the loss function. But these empirical hull formulas carry 10–15% error in real sea states, so forcing the network to satisfy them makes it inherit their errors (this is exactly why the HYBRID baseline performs *worse* than the pure DATA model).

The PI-NODE instead shifts the physical constraints to the **propeller**, whose operation obeys much stricter kinematic/dynamic laws (Wageningen B-Series). The neural network becomes a **"virtual sensor"** that predicts the three unmeasurable *propulsive factors*:

- **Wake fraction** `w`
- **Thrust deduction factor** `t`
- **Relative rotative efficiency** `η_R`

These are then routed through analytically exact propeller equations to compute power.

## Architecture (Dual-Branch)

The model splits the problem into two physical regimes to prevent parameter entanglement between slow-varying hull phenomena and fast-varying weather disturbances.

**Branch A — Calm-Water Dynamics (Neural ODE)**
- Inputs: Speed-Through-Water, Fore/Aft draft, Trim, RPM (weather excluded).
- An `Encoder → Neural ODE (torchdiffeq) → Decoder` integrates the latent state over time, capturing inertia/transients that static MLPs cannot.
- Outputs the calm-water factors `[w_calm, t_calm, η_R_calm]` (bounded via sigmoid).

**Branch B — Sea-State Residual (Weather MLP)**
- Inputs: relative/true wind speed and direction.
- A small feed-forward MLP outputs instantaneous residual corrections `[Δw, Δt, Δη_R]`.

**Aggregation + Physics Layer (analytical, non-negotiable)**
```
w_total  = w_calm + Δw      (clamped to physical bounds)
t_total  = t_calm + Δt
η_R_total = η_R_calm + Δη_R

Va = V_ship · (1 − w_total)          # advance speed
J  = Va / (n · D)                    # advance coefficient
K_T(J) = b0 + b1·J + b2·J² + b3·J³   # trainable (B-Series init)
K_Q(J) = c0 + c1·J + c2·J² + c3·J³   # trainable (B-Series init)
Q = K_Q · ρ · n² · D⁵                # torque
P = 2π · n · Q                       # shaft power
```

The `K_T` / `K_Q` polynomial coefficients are **trainable** so the model can account for long-term propeller degradation (biofouling), while physics-informed penalties keep them realistic.

**Physics-Informed Loss**
```
L = SmoothL1(P_pred, P_true)
  + λ_range     · (K_T, K_Q stay positive / within bounds)
  + λ_curvature · (penalize 2nd derivative → no unrealistic oscillation)
  + λ_prior     · (keep coefficients near B-Series init)
  + λ_η0        · (enforce η0 = J·K_T / (2π·K_Q) ∈ [0.40, 0.75])
```

Training uses `ReduceLROnPlateau`, gradient clipping (max-norm 5.0), and a 1.5× LR boost on the polynomial coefficients. See `docs/PI_NODE_THEORY_AND_ARCHITECTURE.md` for details.

## Repository Structure

```
.
├── config.py                      # Loads .env into typed config classes
├── read_data.py                   # DataProcessor: loading, filtering, scaling, sequencing
├── base_model.py                  # Shared MLP architecture (DATA / HYBRID)
│
├── main_DATA.py                   # Baseline 1: pure data-driven MLP
├── main_HYBRID.py                 # Baseline 2: soft-physics PINN (ITTC-78 penalties)
├── main_PI_NODE_Propeller.py      # The PI-NODE model (Neural ODE + propeller physics)
│
├── prepare_ship_data.py           # Merge raw per-ship CSVs into DANAE's schema
├── sweep_pinode_regularization.py # Phase 2: regularization-weight sweep
├── train_final_pinode.py          # Phase 2.2: train + save the final PI-NODE
│
├── evaluate_baselines.py          # Phase 1: DATA vs HYBRID on DANAE
├── evaluate_transient.py          # Phase 3: steady-state vs transient regimes
├── evaluate_transfer.py           # Phase 4: zero-shot transfer to unseen vessels
├── evaluate_fewshot.py            # Phase 5: few-shot fine-tuning on a new vessel
├── quick_compare.py               # Quick DATA-vs-PI-NODE sanity check
│
├── .env.<ship>                    # Per-vessel constants (danae, kastor, thalia, ...)
├── results/                       # Sweep CSV + checkpoints
├── docs/                          # Theory, experimental plan, literature notes
└── PhD/                           # Raw vessel data (Excel/CSV) and technical drawings
```

Saved artifacts: `best_model_DATA_danae.pt`, `best_model_HYBRID_danae.pt`, `best_model_PI_NODE_danae.pt`, plus the fitted scalers `data_processor_danae.pkl` (tabular) and `data_processor_danae_temporal.pkl` (sequenced).

## Requirements

Python 3.12.2. Install dependencies into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies include PyTorch, `torchdiffeq` (Neural ODE solver), scikit-learn, pandas, and `python-dotenv`. Training auto-selects CUDA → MPS → CPU.

## Configuration

All tunable parameters live in `.env` files loaded by `config.py`. Each vessel has its own file (`.env.danae`, `.env.kastor`, ...) carrying that ship's constants. Key groups:

- **Data**: `DATA_FILE_PATH`, `TARGET_COLUMN`, `DROP_COLUMNS`, `MIN_POWER`, `MIN_SPEED`.
- **Columns**: `SPEED_COLUMN`, `DRAFT_FORE_COLUMN`, `DRAFT_AFT_COLUMN`, etc.
- **Ship**: `WATER_DENSITY`, `SHIP_LENGTH`, `SHIP_DISPLACEMENT`, `PROPULSIVE_EFFICIENCY`, ...
- **Propeller** (`PropellerConfig`): `PROPELLER_DIAMETER`, `PROPELLER_BLADES`, `PROPELLER_PITCH_RATIO`, `PROPELLER_AREA_RATIO`.
- **Sequencing**: `SEQUENCE_LENGTH` (default 10 timesteps for the ODE), `MAX_TIME_GAP_SECONDS`.
- **Training**: `DEFAULT_EPOCHS_FINAL`, `DEFAULT_OPTIMIZER`, `EARLY_STOPPING_PATIENCE`, etc.

> **Data leakage prevention:** columns directly derived from the target (Shaft Torque, Shaft Thrust, Propeller-Shaft-Power, ME RPM) are excluded via `DROP_COLUMNS`. The PI-NODE temporarily re-includes `Propeller-Shaft-RPM`, which it needs as a physical input.

To switch the active vessel, copy its env file over the active one:

```bash
cp .env.kastor .env
```

## Dataset

The source (training) domain is **M/V DANAE**, an 82,000 DWT bulk carrier (Laros Shipping), ~120,000 usable records from April–September 2022. The target is **Shaft Power_TRQM** (kW) from the onboard torque meter.

Preprocessing (`read_data.py`): filter port/anchor rows (power < 1000 kW or speed < 4 kn), fill NaNs with column medians, standardize features and target with `StandardScaler` (fitted on train only), and split 80/20 **chronologically** (no shuffling). For the PI-NODE, overlapping windows of 10 consecutive timesteps are created via `create_sequences`.

## Usage

### 1. Train the baselines (Phase 1)
```bash
python -u evaluate_baselines.py
```

### 2. Tune and train the PI-NODE (Phase 2)
```bash
python -u sweep_pinode_regularization.py   # regularization sweep (auto-resumes via results/ CSV)
python -u train_final_pinode.py            # train winning config, save weights + scaler
```
You can also run the PI-NODE module directly for a quick standalone run:
```bash
python -u main_PI_NODE_Propeller.py
```

### 3. Transient analysis (Phase 3)
```bash
python -u evaluate_transient.py
```

### 4. Zero-shot transfer to an unseen vessel (Phase 4)
```bash
cp .env.kastor .env        # swap in the target ship's constants
python -u evaluate_transfer.py
cp .env.danae .env         # restore source vessel
```
For PI-NODE, transfer needs **only** swapping the propeller constants in `.env` — the network weights stay frozen and the analytical equations adapt to the target propeller geometry.

### 5. Few-shot fine-tuning (Phase 5)
```bash
cp .env.kastor .env
python -u evaluate_fewshot.py
cp .env.danae .env
```

### Adding a new vessel
```bash
python prepare_ship_data.py kastor   # merge raw CSVs into DANAE's schema
```

## Results Summary

**Source domain (DANAE):**

| Model   | Test RMSE (kW) | vs DATA  | Description |
|---------|---------------:|----------|-------------|
| **PI-NODE** | **275.95** | **−35.0%** | Grey-box: Neural ODE + hard propeller physics |
| DATA    | 424.80         | baseline | Black-box MLP |
| HYBRID  | 703.69         | +65.6%   | Soft-physics MLP + ITTC-78 penalties |

**Zero-shot transfer (trained on DANAE, no retraining), MAPE:**

| Target ship | PI-NODE | DATA | HYBRID |
|-------------|--------:|-----:|-------:|
| KASTOR (sister, 82K) | **4.39%** | 7.47% | 26.25% |
| MENELAOS (64K)       | **11.76%** | 40.55% | 30.39% |
| THALIA (different class) | **18.33%** | 44.49% | 33.53% |
| THISSEAS (75.2K)     | **18.03%** | 83.40% | 82.68% |

| Phase | Key finding |
|-------|-------------|
| **1–2 Source domain** | PI-NODE: 275.95 kW RMSE (~3.1% MAPE) — 35% better than DATA, 61% better than HYBRID. |
| **3 Transient** | PI-NODE holds ~3% MAPE across all regimes; HYBRID degrades most in transients (ITTC-78 assumes steady state). |
| **4 Zero-shot** | PI-NODE achieves 4–18% MAPE on unseen vessels; its advantage **grows** with ship dissimilarity (1.7× → 4.6×). |
| **5 Few-shot** | PI-NODE reaches commercial accuracy (~5%) with ~2 days of data; DATA needs ~3 weeks. Sister ships need zero data. |

**Defining result:** physics-informed architectures provide disproportionately larger benefits precisely when needed most — on unseen vessels.

## Acknowledgments

Special thanks to Christoforos Rekatsinas (Ph.D.) for his guidance and support.

## Contact

- Alexiou Kiriakos
- Email: kiriakosal2004@yahoo.gr
