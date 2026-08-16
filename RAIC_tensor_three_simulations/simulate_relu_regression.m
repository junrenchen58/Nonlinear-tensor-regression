function output = simulate_relu_regression(cfg)
%SIMULATE_RELU_REGRESSION Compare the ReLU and generic SIM gradients.
%
% Both curves use the baseline (n,r) setting and the same observations.
% The model-specific ReLU gradient targets X exactly. The generic squared-
% loss SIM gradient is not modified and targets mu*X with mu=1/2; its final
% output is divided by mu only when the reported error is computed.

fprintf('Simulation 2/3: ReLU regression\n');

setting = cfg.settings.baseline;
n = setting.n;
r = setting.r;
dims = [n, n, n];
ranks = [r, r, r];
d = prod(dims);
m_values = cfg.m_values;
num_m = numel(m_values);
mu = 1/2;

% Method 1: model-specific ReLU gradient.
% Method 2: exact generic SIM squared-loss gradient, then final 1/mu scaling.
errors = nan(cfg.num_trials, num_m, 2);
iterations = nan(cfg.num_trials, num_m, 2);

for trial = 1:cfg.num_trials
    rng(cfg.seed + 200000 + trial, 'twister');
    X = traic.generate_unit_tucker_tensor( ...
        n, r, cfg.max_condition, cfg.max_tensor_draws);
    A_all = traic.gaussian_design(max(m_values), d, ...
        cfg.design_precision, cfg.design_block_rows);
    z_all = traic.forward(A_all, X);
    y_all = max(z_all, 0);

    for m_idx = 1:num_m
        m = m_values(m_idx);
        A = A_all(1:m, :);
        y = y_all(1:m);

        % The unscaled first moment estimates (1/2)X.
        raw_state0 = traic.first_moment_initializer(A, y, dims, ranks);

        % ReLU initialization: use the first-moment direction and the norm
        % estimator sqrt(2*pi)*mean(y), exactly as in the paper.
        beta_hat = sqrt(2*pi) * mean(y);
        relu_state0 = traic.normalize_state(raw_state0, beta_hat);
        [relu_state, relu_info] = traic.run_rgd(A, y, relu_state0, ...
            'relu', cfg.max_iterations.relu, cfg.tolerance, ...
            cfg.min_iterations);
        errors(trial, m_idx, 1) = norm(relu_state.X - X) / norm(X);
        iterations(trial, m_idx, 1) = relu_info.iterations;

        % Generic SIM comparator: do not change its loss or gradient.
        [sim_state, sim_info] = traic.run_rgd(A, y, raw_state0, ...
            'sim', cfg.max_iterations.sim, cfg.tolerance, ...
            cfg.min_iterations);
        X_hat_sim = sim_state.X / mu;
        errors(trial, m_idx, 2) = norm(X_hat_sim - X) / norm(X);
        iterations(trial, m_idx, 2) = sim_info.iterations;
    end

    if cfg.verbose
        fprintf('  baseline           trial %3d/%3d completed\n', ...
            trial, cfg.num_trials);
    end
    clear A_all z_all y_all X
end

output = struct();
output.model = 'ReLU regression';
output.mu_sim = mu;
output.m_values = m_values;
output.errors = errors;
output.iterations = iterations;
output.labels = {'ReLU gradient', 'SIM gradient'};
output.setting = setting;
end
