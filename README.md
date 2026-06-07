# Ship Shaft-Power Prediction with a Physics-Informed Neural ODE (PI-NODE)

Predicting a vessel's **shaft power** from real operational time-series (speed, drafts, trim,
shaft RPM, weather) is central to voyage optimisation, fuel and emissions management, and
condition monitoring. This repository implements and evaluates **PI-NODE**, a grey-box model that
constrains a neural network with the rigid laws of **propeller hydrodynamics** rather than the
noisy empirical hull-resistance formulas that most physics-informed approaches rely on.

The central finding of the project: forcing empirical *hull*-resistance physics onto a model in
real sea states does **not** help — it performs no better than a plain data-driven network —
whereas constraining the model with accurate *propeller* physics acts as a strong inductive bias
that improves source-domain accuracy **and** enables zero-shot transfer to vessels the model has
never seen.

Companion documents in `docs/`:

- `docs/PI_NODE_THEORY_AND_ARCHITECTURE.md` — architecture, equations, training infrastructure.
- `docs/EXPERIMENTAL_PLAN.md` — dataset, methodology, and the full results tables.
- `docs/PAPER_FINDINGS.md` — the Results-and-Discussion narrative.
- `docs/RELATED_WORK.md` — positioning against the recent literature.
- `docs/EXPERIMENT_RUNBOOK.md` — the end-to-end run record and resume notes.
- `docs/paper/manuscript.md` — the assembled journal-paper draft.

---

## 1. The core idea — why the propeller?

Most physics-informed power models inject the **entire ship resistance** (frictional, wave-making,
air) into the loss. Those empirical hull formulas carry roughly 10–15 % error in real sea states,
so a network forced to satisfy them inherits that error. This is exactly why the soft-hull-physics
baseline (HYBRID) ends up no better than the pure data-driven model.

PI-NODE instead places the physical constraint on the **propeller**, whose operation obeys much
stricter kinematic and dynamic laws (the Wageningen B-series). The network is reduced to a
**virtual sensor** that infers the three unmeasurable *propulsive factors*:

- **wake fraction** `w`
- **thrust-deduction factor** `t`
- **relative rotative efficiency** `η_R`

These are routed through analytically exact propeller equations to compute power, so the hard
physics — not the network — has the final say.

---

## 2. Architecture

The model splits the problem into two physical regimes so that slow-varying hull behaviour and
fast-varying weather disturbances do not entangle.

**Branch A — calm-water dynamics (Neural ODE).**
Inputs are the calm-water channels (speed-through-water, fore/aft draft, trim, shaft RPM, and the
slow navigational signals); weather is deliberately excluded. An `Encoder → Neural ODE
(torchdiffeq) → Decoder` integrates a latent state over the operational window, capturing the
inertia and transients a static MLP cannot. It outputs the calm-water factors
`[w, t, η_R]`, bounded by a sigmoid.

**Branch B — sea-state residual (weather MLP).**
Inputs are the relative/true wind channels. A small feed-forward network outputs instantaneous
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
degradation (biofouling), while physics-informed penalties keep them realistic:

```
L = SmoothL1(P_pred, P_true)
  + λ_range     · keep K_T, K_Q positive and within bounds
  + λ_curvature · penalise the 2nd derivative (no unphysical oscillation)
  + λ_prior     · keep coefficients near the B-series initialisation
  + λ_η0        · keep open-water efficiency η0 = J·K_T / (2π·K_Q) in [0.40, 0.75]
```

Training uses `ReduceLROnPlateau`, gradient clipping (max-norm 5.0), and a 1.5× learning-rate
boost on the polynomial coefficients. **Zero-shot transfer** to a new vessel requires only swapping
the propeller constants `D`, `P/D`, `Z`; the network weights stay frozen and the analytical layer
adapts to the new geometry.

---

## 3. Models in this repository

| Model | File | Description |
|---|---|---|
| **PI-NODE** | `main_PI_NODE_Propeller.py` | The grey box: Neural-ODE encoder + trainable propeller physics. |
| **DATA** | `main_DATA.py` | Pure data-driven MLP (black-box reference). |
| **HYBRID** | `main_HYBRID.py` | MLP with soft ITTC-78 hull-resistance penalties (the soft-hull-physics baseline). |
| **PI-KAN** | `main_PI_KAN.py`, `kan_layer.py` | A Kolmogorov–Arnold-Network backbone with the same soft physics as HYBRID — a strong competitor benchmarked head-to-head. |

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
├── kan_layer.py                    # Self-contained B-spline Kolmogorov–Arnold Network
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
├── make_figures.py                 # Generate the publication figures (F1–F10)
│
├── tests/                          # Standalone unit/regression tests (no pytest required)
├── .env.<vessel>                   # Per-vessel constants (danae, kastor, menelaos, thalia, thisseas, ...)
├── requirements.txt                # Python dependencies
├── results/                        # Checkpoints, CSV outputs, and figures (generated; not tracked)
├── docs/                           # Theory, experimental plan, findings, related work, paper draft
└── PhD/                            # Raw vessel logs and technical drawings
```

Saved artefacts (written to the repository root and `results/`): `best_model_DATA_danae.pt`,
`best_model_HYBRID_danae.pt`, `best_model_PI_NODE_danae.pt`, the per-seed checkpoints
`results/best_model_PI_NODE_seed{0..4}.pt`, and the fitted scalers `data_processor_danae.pkl`
(tabular) and `data_processor_danae_temporal.pkl` (sequenced).

---

## 5. Installation

Python 3.12. Install the dependencies into an isolated environment:

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: PyTorch, `torchdiffeq` (the Neural-ODE solver), scikit-learn, pandas, matplotlib,
and `python-dotenv`. Training auto-selects the device in the order CUDA → MPS → CPU. The PI-KAN
model is the one exception: it runs on CPU because the Apple-MPS backend miscomputes the
second-order automatic differentiation through the B-spline basis.

---

## 6. Configuration

All tunable parameters live in `.env` files read by `config.py`. Each vessel has its own file
carrying that ship's constants; switch the active vessel by copying its file over `.env`:

```bash
cp .env.kastor .env       # work on KASTOR
cp .env.danae .env        # restore the source vessel
```

The main configuration groups are: **data** (`DATA_FILE_PATH`, `TARGET_COLUMN`, `DROP_COLUMNS`,
`MIN_POWER`, `MIN_SPEED`), **column names**, **ship constants** (density, length, displacement,
propulsive efficiency, …), **propeller constants** (diameter, blade number, pitch ratio, area
ratio), **sequencing** (`SEQUENCE_LENGTH`, default 10 timesteps; `MAX_TIME_GAP_SECONDS`), and
**training** (epochs, optimiser, early-stopping patience).

**Leakage prevention.** Columns directly derived from the target (shaft torque, shaft thrust,
propeller-shaft-power, main-engine RPM) are excluded through `DROP_COLUMNS`. PI-NODE re-includes
only `Propeller-Shaft-RPM`, which it needs as a physical input to the propeller law; this channel
is available to every model.

---

## 7. Dataset

The source (training) vessel is **M/V DANAE**, an 82,000 DWT bulk carrier, with roughly 120,000
usable records spanning several months of in-service operation. The target variable is shaft power
(kW) from the onboard torque meter.

Preprocessing (`read_data.py`): port and anchor rows are filtered out (power below 1000 kW or speed
below 4 knots); the data is split 80/20 **chronologically** with no shuffling; missing values are
filled with column medians **computed on the training split only**; and features and target are
standardised with a `StandardScaler` fitted on the training split only. For PI-NODE, overlapping
windows of 10 consecutive timesteps are built by `create_sequences`. The derived `dt` and `dV/dt`
(acceleration) drive sequence gap-detection and the transient-regime split but are **not** used as
model inputs.

---

## 8. Reproducing the pipeline

Run from the project root with the source vessel active (`cp .env.danae .env`). The steps are
ordered; later ones depend on the checkpoints produced earlier.

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
cp .env.kastor .env
python -u evaluate_transfer.py
cp .env.danae .env
```
Only the propeller constants change; the weights stay frozen.

**Few-shot adaptation.**
```bash
cp .env.kastor .env
python -u evaluate_fewshot.py
cp .env.danae .env
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
python -u make_figures.py                   # writes results/figures/F1–F10
```

**Adding a new vessel.**
```bash
python prepare_ship_data.py kastor          # merge raw logs into the common schema
```

---

## 9. Results

All figures are written to `results/figures/` (`F1`–`F10`).

**Source domain (DANAE).**

| Model | Test RMSE (kW) | vs DATA | Description |
|---|---:|---|---|
| **PI-NODE** | **312.52** | **−43.9 %** | Grey box: Neural ODE + propeller physics |
| PI-KAN | 471.04 ± 72.80 | −15.5 % | KAN backbone + soft hull-physics, 5-seed |
| DATA | 557.52 | baseline | Black-box MLP |
| HYBRID | 583.88 | +4.7 % | Soft-physics MLP (ITTC-78 penalties) |

The PI-NODE multi-seed estimate is **286.42 ± 9.89 kW** over five seeds (coefficient of variation
≈ 3.5 %); the 312.52 figure is the single canonical run. DATA and HYBRID are single runs under the
same protocol. The HYBRID result is the key negative finding — soft hull-physics buys nothing over
plain data, which is what motivates moving the physics to the propeller.

**Head-to-head vs PI-KAN.** Trained on DANAE under an identical protocol (same split, features,
physics, and five seeds), PI-KAN reaches 471.04 ± 72.80 kW. It is a strong competitor — it beats
both DATA and HYBRID — yet PI-NODE still wins by **39 % in RMSE** and is about **seven times more
stable across seeds**. The Camp-A "B-series-ML" surrogates from the literature predict open-water
`K_T`/`K_Q` rather than operational power, so they are a contextual contrast rather than a trained
baseline.

**Zero-shot transfer (trained on DANAE, no retraining), MAPE.**

| Target vessel | PI-NODE | DATA | HYBRID |
|---|---:|---:|---:|
| KASTOR (sister, 82K) | **3.75 %** | 9.47 % | 23.46 % |
| MENELAOS (64K) | **4.87 %** | 39.72 % | 35.05 % |
| THALIA (different class) | **27.72 %** | 41.55 % | 41.07 % |
| THISSEAS (75.2K) | **32.19 %** | 88.63 % | 77.69 % |

PI-NODE has the lowest error on every unseen vessel. The high THALIA/THISSEAS errors are reported
openly: these vessels are further out of distribution, and PI-NODE remains first but is not immune.

**Summary across studies.**

| Study | Key finding |
|---|---|
| Source domain | PI-NODE 286.42 ± 9.89 kW RMSE (≈ 3.6 % MAPE) — 43.9 % better than DATA, and better than HYBRID. |
| Transient | PI-NODE holds ≈ 3.5–3.9 % MAPE across steady and transient regimes; the baselines degrade most in transients. |
| Zero-shot | Lowest MAPE on every unseen vessel; range 3.75–32 %. |
| Few-shot | PI-NODE reaches ≈ 3–4 % MAPE from as little as 1 % of target data; the data-driven model needs far more. |
| Ablation | Freezing the learnable propeller polynomials triples the error (312 → 941 kW) — the trainable B-series is the decisive component. |
| Multi-seed | 286.42 ± 9.89 kW over five seeds — a robust result, not a single fortunate run. |
| Uncertainty | The deep ensemble is the best estimator (278.4 kW). Raw intervals are over-confident (51.4 % coverage at the 95 % level); split conformal restores ≈ 90/95 % coverage on a held-out half and stays well-behaved on MC-Dropout, where Gaussian temperature scaling breaks down. |

**Takeaway.** The physics-informed architecture delivers its largest gains precisely where they
matter most — on unseen vessels and in the low-data regime.

---

## 10. Tests

The `tests/` directory holds standalone scripts (no test framework required); run any of them
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
