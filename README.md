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

## 2. How it works, end to end

The model rests on one division of labour: the network never predicts power directly. Power can be
computed exactly from propeller theory, as long as we know a handful of factors that nobody can
measure on a working ship. So the neural network estimates only those hidden factors, and the
standard propeller equations turn them into power. The network acts as a virtual sensor, and the
physics does the arithmetic. Everything after the network is plain differentiable arithmetic, which
is what lets the error signal flow all the way back during training.

The rest of this section follows the data through that pipeline: what goes in, what the network puts
out, how those outputs become a power number, and what the training loop pushes on.

### 2.1 The physics we lean on

When a propeller turns, the water it bites into is not arriving at the ship's speed. The hull drags
a layer of water along with it, so the propeller sees a slower advance speed `Va`. The fraction of
speed lost this way is the wake fraction `w`:

```
Va = V_ship · (1 − w)
```

The propeller produces thrust, but the suction it creates over the stern adds to the hull's
resistance, so the thrust the engine has to deliver is larger than the net force pushing the ship.
That gap is the thrust-deduction factor `t`. A propeller working in the disturbed flow behind a hull
is also slightly less (or more) efficient than the same propeller in open water, and that correction
is the relative rotative efficiency `η_R`. These three (`w`, `t`, `η_R`) are the classic propulsive
factors. They are real, physically meaningful quantities, and none of them is logged by any onboard
instrument, which is exactly why a network is useful here.

Once we have the inflow speed, the propeller's behaviour is described non-dimensionally by the
advance coefficient `J` and two coefficient curves: thrust `K_T(J)` and torque `K_Q(J)`. For a given
propeller geometry these curves are tabulated by the Wageningen B-series, which comes from decades of
systematic open-water tank tests. That makes them about as close to ground truth as marine
hydrodynamics gets, and it is why we anchor the model to the propeller rather than to the much
shakier empirical hull-resistance formulas (see Section 1).

### 2.2 What goes into the network (inputs)

The model reads overlapping windows of 10 consecutive timesteps of operational data, and it splits
the channels into two groups on purpose, so that slow hull behaviour and fast weather disturbances
do not get tangled together:

- Calm-water channels: everything that describes the ship's own steady operating point, namely
  speed-through-water, fore and aft draft, trim, shaft RPM, and the slow navigational signals. These
  feed Branch A.
- Weather channels: anything whose name mentions wind, wave, or swell. These feed Branch B.

Two derived signals, `dt` (time gap) and `dV/dt` (acceleration), are computed but never fed to the
model. They exist only to detect gaps between sequences and to label the steady-versus-transient
regimes for analysis.

Separately from the windowed scaled inputs that the network sees, the pipeline also carries the
unscaled shaft RPM and ship speed of the final timestep. Those two raw values are not network inputs.
They go straight into the propeller equations later, where they need their true units (rev/s and m/s)
rather than standardised z-scores.

### 2.3 What the network outputs

Neither branch outputs power. Between them they produce the three propulsive factors, split into a
calm-water baseline and a small weather correction.

Branch A handles the calm-water dynamics with a Neural ODE. An encoder turns the calm-water features
into an initial latent state `z₀`. A Neural ODE (`torchdiffeq`) then integrates that state along a
learned vector field `dz/dt = f_θ(z; context)` over a normalised time interval, and a decoder reads
the final state into three numbers. The ODE is what gives the model memory of the inertia and
transients that a static MLP cannot represent. A sigmoid squashes the three outputs into physically
sensible intervals:

```
w_calm   ∈ [0.05, 0.45]      # wake fraction
t_calm   ∈ [0.05, 0.30]      # thrust deduction
η_R_calm ∈ [0.95, 1.10]      # relative rotative efficiency
```

Branch B is a small feed-forward network for the sea-state residual. It looks at the weather channels
of the most recent timestep and outputs three additive corrections `[Δw, Δt, Δη_R]`. These start near
zero (they are scaled down at initialisation) so that early in training the calm-water branch sets the
baseline and weather only nudges it:

```
w_total   = clamp(w_calm   + Δw)
t_total   = clamp(t_calm   + Δt)
η_R_total = clamp(η_R_calm + Δη_R)
```

So the network's whole job is to produce these three combined factors per timestep. That is the line
between what is learned and what is known.

### 2.4 From factors to power (the analytical layer)

Now the physics takes over. Using the network's `w_total`, the measured shaft speed `n` (rev/s) and
ship speed `V_ship` (m/s), and the fixed propeller geometry, the power follows from a short chain of
exact relations, with no learned function anywhere in it except the coefficient curves:

```
Va  = V_ship · (1 − w_total)             # advance speed (Section 2.1)
J   = Va / (n · D)                       # advance coefficient, D = propeller diameter

K_T(J) = b0 + b1·J + b2·J² + b3·J³       # thrust  coefficient (cubic, trainable)
K_Q(J) = c0 + c1·J + c2·J² + c3·J³       # torque  coefficient (cubic, trainable)

Q   = K_Q · ρ · n² · D⁵                  # delivered shaft torque  [N·m],  ρ = water density
P   = 2π · n · Q                         # delivered shaft power    [W]  →  kW
```

That last line is the model's prediction of shaft power. One honest implementation note is worth
making here. Power flows through the torque path (`w → J → K_Q → Q → P`), so the factor that actually
drives the result is the wake fraction `w`, together with the two coefficient curves. The other two
factors, `t` and `η_R`, are predicted by the network and bounded as above, but in the current code
they are not wired into the power equation. Even the open-water efficiency used by the regulariser
(Section 2.5) is derived from `K_T`/`K_Q`, not from `η_R`. They are carried as part of the
propulsive-factor vector for completeness and possible future use, while the gradient that trains the
network arrives through `w`.

The `K_T` and `K_Q` coefficients are trainable. They are initialised at the Wageningen B-series
values for this propeller, but left as learnable parameters. Real propellers foul and wear over
months at sea, and a frozen B-series curve cannot track that drift. Letting the four coefficients of
each cubic move lets the model absorb that long-term degradation, while the regularisation below keeps
them from wandering into physically impossible shapes. The ablation in Section 9 is blunt about how
much this matters: freezing these polynomials roughly triples the error.

### 2.5 The loss, and what backpropagation actually moves

Predicted power is converted into the same standardised space as the logged target, and the primary
loss is a SmoothL1 (Huber) error against the real torque-meter reading. Huber is used rather than
plain MSE because manoeuvring and port approaches throw occasional power outliers that an L2 loss
would overweight. On top of that sit four soft physics penalties that encode prior knowledge about
the propeller's operating envelope:

```
L = SmoothL1(P_pred, P_true)                               # fit the measured power
  + λ_range     · (push K_T, K_Q back inside admissible bounds)
  + λ_curvature · (penalise the polynomials' 2nd derivative, no unphysical oscillation)
  + λ_prior     · (soft pull of the 8 coefficients toward their B-series values)
  + λ_η0        · (keep open-water efficiency η0 = J·K_T / (2π·K_Q) in [0.40, 0.75])
```

This is what makes the model physics-informed rather than a network with a physics step bolted on
afterwards. The whole chain in Section 2.4 is differentiable, so gradients from the power error flow
backwards straight through the propeller equations. When `P_pred` is too high, backprop does not
adjust some opaque output head. It propagates `∂L/∂P` through `P = 2πnQ`, through `Q = K_Q·ρ·n²·D⁵`,
through the cubic `K_Q(J)`, and through `J = Va/(nD)` and `Va = V_ship(1−w)`, so it reaches two places
at once:

1. the eight polynomial coefficients `b0..b3, c0..c3`, which nudge the propeller curves; and
2. the network weights (encoder, ODE vector field, decoder, weather MLP), because `w_total` sits
   inside `Va`, so the gradient reaches the parameters that produced `w`.

So the loss trains the network and the physics together, and every gradient path has a physical
meaning. The optimiser is Adam, with a mild 1.5x learning-rate boost on the eight polynomial
coefficients (they live in a tiny, well-constrained space and can take larger steps than the network),
gradient clipping at max-norm 5.0 to guard against ODE-solver blow-ups, and `ReduceLROnPlateau` on the
validation loss.

This is also why the model transfers. The geometry of a new ship enters only through the constants
`D`, `P/D`, `Z` in the analytical layer, so zero-shot transfer to an unseen vessel is just a matter of
swapping those constants. The network weights stay frozen, and the physics re-derives power for the
new propeller. That is what the transfer results in Section 9 are showing.

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

Special thanks to Christoforos Rekatsinas (Ph.D.) for his guidance and to Pariotis Efthimios / Leligkou Eleni Aikaterini for their support.

## 12. Contact

- Alexiou Kiriakos
- Email: kiriakosal2004@yahoo.gr
