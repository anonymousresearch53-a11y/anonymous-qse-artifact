# Quantum Optimizer Orchestration

This repository studies whether a learned decision policy can select an
effective optimization strategy for each quantum circuit. The available
strategies are no optimization, one optimizer, or an ordered two-optimizer
chain.

The project evaluates four quantum-circuit optimizers:

- Qiskit (`qiskit_standard`)
- TKET (`tket`)
- WISQ Rules (`wisq_rules`)
- WISQ-BQSKit (`wisq_bqskit`)

For every retained original circuit, the experimental data contains the four
single-optimizer outcomes and the sixteen ordered two-optimizer outcomes. The
selection problem is therefore evaluated using measured optimizer results,
rather than assuming that one optimizer or one fixed chain is best for every
circuit.

## Research questions

The repository supports experiments on questions such as:

1. Can a circuit-aware policy improve over a fixed optimizer or fixed chain?
2. Do circuit features help a policy choose among optimizers and chains?
3. Does sequential selection provide an advantage over selecting one optimizer?
4. How do reinforcement-learning, supervised, and LLM-based selectors compare?
5. How sensitive are the conclusions to reward definitions, hyperparameters,
   and random seeds?

## Experimental design

The canonical dataset is built from the combined single- and second-optimization
databases and the corresponding QASM artifacts. Feature extraction recomputes
circuit metrics and structural features from QASM, including:

- number of qubits and circuit depth;
- total gates and two-qubit gates;
- two-qubit depth and gate density;
- gate-type fractions and ratio features; and
- interaction-graph features.

Circuits are split by `original_circuit` for training, not by individual result row. Thus,
all outcomes and chains for a circuit remain in the same train, validation, or
test split. Circuits with zero initial two-qubit gates are excluded from the
canonical analysis because percentage reduction in that metric is undefined.

The current analysis uses four optimizer actions and a stop action. For a
two-step episode:

1. the policy can stop or select the first optimizer;
2. after the first optimizer, it can stop or select a second optimizer; and
3. the resulting circuit is evaluated against the original circuit.

## Methods

### Data
All original circuits used in this study are stored in `data/original_circuits`. Artifact circuits
created from optimization of the initial circuits are not included in the repo because there are thousands of large files.
All optimization run information is stored in `dbs/combined_master.csv` which is the dataset that is used for the machine learning
workflows. 

Optimization results can be recomputed using `scripts/run_single_step_grid_search.py`.

### Brute-Force Optimizer Analysis

All original circuits in the current dataset were optimized to chains of length 2 with all 4 optimizers.
All single optimization runs are stored in `dbs/combined_fixed/data_single_opt/combined_single_opt.db`.
All two optimizer runs are stored in `dbs/combined_fixed/data_second_opt.db`.
Analysis of performance of the different optimizers and optimizer chains can be found in `data/analyze_data.ipynb`


### Reinforcement learning

The current RL implementation is `two_opt_chain_rl_v4`. It uses a Gymnasium
environment and configurable Stable-Baselines3 algorithms, including PPO, A2C, and DQN.
The environment exposes normalized circuit observations and returns rewards for
the selected optimization action.

The configured RL utility is a reward-defined objective combining some or all
of:

- two-qubit-gate reduction;
- circuit-depth reduction;
- a runtime penalty; and
- a terminal correction based on the final circuit relative to the original.

See [`machine_learning/RL/two_opt_chain_rl_v4/README.md`](machine_learning/RL/two_opt_chain_rl_v4/README.md)
for the environment, reward, feature-batch, training, and evaluation details.

### Supervised baselines

The supervised-learning modules provide learned selection baselines, including
multiclass chain selection and utility regression. These models use the same
prepared data and should be compared using the same circuit-level splits and
held-out evaluation protocol.
`machine_learning/multiclass` model was the best performing supervised learning baseline.

### LLM selector

`machine_learning/LLM/llm_v3/` contains an LLM-based selector for the same
decision space. It supports structured selection, candidate shortlisting,
train-set chain profiles, retrieval features, and configurable objective
weights.

See [`machine_learning/LLM/llm_v3/README.md`](machine_learning/LLM/llm_v3/README.md)
for prompt modes, experiment presets, and provider configuration.

## Evaluation protocol

Results should be interpreted using both aggregate and circuit-level measures:

- two-qubit-gate reduction;
- depth reduction;
- runtime;
- configured RL utility;
- per-circuit wins, losses, and ties against relevant baselines.

The held-out test split should be used only for final reporting. Features,
feature scales, reward weights, category vocabularies, and hyperparameters must
be selected using training data and, where appropriate, validation utility.
Test categories are handled explicitly rather than deriving category
information from the full dataset.

The RL audit suite checks these safeguards, including independent reward parity,
split disjointness, train-derived normalization, category handling, feature
behavior, seed uncertainty, and per-circuit comparisons.

## Reproducing the main workflow

Create the project environment from the repository’s environment specification:

```bash
conda env create -f environment.yml
conda activate circuits
```

From the repository root, rebuild the canonical dataset:

```bash
python machine_learning/feature_extraction/build_combined_master.py
```
Note: Optimization will need to be recomputed in order to rebuild the master files.
    CSV files are included in this repo with the values we computed. If you want to use our values, you can skip this step.

The main dataset outputs are:

- `dbs/combined_master.csv`;
- `machine_learning/feature_extraction/combined_master_features.csv`; and
- `machine_learning/feature_extraction/combined_master_build_summary.json`.

Train the default RL v4 configuration:

```bash
python machine_learning/RL/two_opt_chain_rl_v4/train.py
```

Run the workflow tests and audit report:

```bash
pytest -q machine_learning/RL/two_opt_chain_rl_v4/tests
python machine_learning/RL/two_opt_chain_rl_v4/tests/workflow_audit.py
```

Run a provider-free LLM smoke test:

```bash
python machine_learning/LLM/llm_v3/run_experiment.py \
  --provider mock \
  --split test \
  --limit 3
```

## Repository map

| Path | Role in the study |
| --- | --- |
| `data/original_circuits/` | Original QASM benchmark circuits. |
| `data/artifacts/single_opt/` | Single-optimizer output artifacts. |
| `data/artifacts/second_opt/` | Two-optimizer-chain output artifacts. |
| `dbs/combined_fixed/` | Combined source SQLite databases. |
| `dbs/combined_master.csv` | Canonical feature-complete dataset. |
| `machine_learning/feature_extraction/` | Dataset construction and feature extraction. |
| `machine_learning/RL/two_opt_chain_rl_v4/` | Current RL method and audits. |
| `machine_learning/supervised_learning/` | Supervised selection baselines. |
| `machine_learning/LLM/llm_v3/` | Current LLM selector. |
| `benchmarks/ai_transpile/` | Optimizer interfaces and benchmark execution. |
| `scripts/` | Data import, validation, evaluation, and analysis utilities. |
| `data/analyze_data.ipynb` | Main comparative analysis. |

Older implementations are retained under `machine_learning/RL/archive/` and
`machine_learning/LLM/archive/` for historical reference.

## Reproducibility and limitations

- Runtime results depend on the execution environment and optimizer versions.
- Reward-based rankings depend on the selected reward type and weights.
- A policy that improves RL utility is not necessarily best on every individual
  metric.
- Aggregate means can hide circuit-level failures or ties, so both summaries
  are reported.
- Large QASM and artifact files may be stored with Git LFS and must be present
  locally before rebuilding the dataset.
- The test split must remain untouched until the final evaluation.

## Detailed documentation

- [Feature extraction](machine_learning/feature_extraction/README.md)
- [RL v4](machine_learning/RL/two_opt_chain_rl_v4/README.md)
- [RL workflow audits](machine_learning/RL/two_opt_chain_rl_v4/tests/README.md)
- [LLM v3](machine_learning/LLM/llm_v3/README.md)
