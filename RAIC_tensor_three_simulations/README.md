# Three simulations for nonlinear low-rank tensor regression

This folder contains the three end-to-end MATLAB simulations used for the nonlinear tensor RAIC paper:

1. truncated tensor single-index model;
2. tensor ReLU regression;
3. one-bit tensor sensing.

The code is self-contained except for the **MATLAB Tensor Toolbox**. It follows the tensor notation, tangent projection, and T-HOSVD retraction used in the original repository.

## Requirements

- MATLAB R2022a or later;
- MATLAB Tensor Toolbox on the MATLAB path.

## Run

Open MATLAB in this folder (or add this folder to the path), then run

```matlab
run_quick_check
```

or equivalently

```matlab
results = run_three_simulations('quick');
```

This is a small smoke test. For the intended paper settings, run

```matlab
run_paper_experiments
```

or equivalently

```matlab
results = run_three_simulations('paper');
```

The paper profile is computationally substantial. It runs serially and uses double-precision Gaussian designs by default. The number of trials, dimensions, ranks, sample sizes, design precision, and iteration limits are all in `experiment_config.m`. Set `cfg.design_precision='single'` there if memory is limited. For a final figure, increasing `cfg.num_trials` from 20 to 30--50 is recommended after the settings have been checked.

## Output

Results are saved under

```text
results/quick/
results/paper/
```

The main files are

```text
three_simulation_results.mat
three_simulations.pdf
three_simulations.png
three_simulations.fig
```

The figure is a one-row, three-panel plot of final relative Frobenius error against sample size `m`.

## Exact procedures implemented

All ground-truth tensors have unit Frobenius norm.

### Truncated single-index model

The observations are

```matlab
y = max(min(A*x, lambda), -lambda);
```

with `lambda=1`. The algorithm uses the paper's unmodified squared-loss gradient

```matlab
(A' * (A*x_t - y))/m
```

and therefore estimates `mu*X`, where

```matlab
mu = erf(lambda/sqrt(2));
```

No `mu` is inserted into the loss, gradient, initialization, or iterations. Only the final iterate is divided by `mu` before its error relative to `X` is reported.

The three curves use the baseline, larger-rank, and larger-dimension settings in `experiment_config.m`.

### ReLU regression

The first curve uses the paper's ReLU subgradient

```matlab
(A' * ((relu(A*x_t)-y) .* (1+sign(A*x_t))))/m
```

and the first-moment direction scaled by

```matlab
sqrt(2*pi)*mean(y)
```

The second curve uses the exact, unmodified generic SIM squared-loss gradient. It estimates `(1/2)X`; only the final iterate is multiplied by `2` before its error is reported. Both curves use the baseline dimension and rank.

### One-bit tensor sensing

The three model-specific curves use

```matlab
sqrt(pi/2) * (A' * (sign(A*x_t)-y))/m
```

followed by unit-Frobenius normalization after each retraction. They correspond to the baseline, larger-rank, and larger-dimension settings.

The fourth curve uses the exact generic SIM squared-loss gradient at the baseline setting. It estimates `sqrt(2/pi)*X`; only its final iterate is divided by `sqrt(2/pi)` before its error is reported.

### Initialization

Every method starts from the current paper's first-moment initializer:

```matlab
H_r( (1/m) * sum_i y_i A_i )
```

where `H_r` is fixed-rank T-HOSVD. The raw initializer is retained for generic SIM. It is norm-rescaled only when required by the model-specific ReLU or one-bit procedure.
