function output = simulate_truncated_single_index(cfg)
%SIMULATE_TRUNCATED_SINGLE_INDEX End-to-end truncated SIM experiment.
%
% The algorithm uses exactly the squared-loss gradient from the paper,
%     m^{-1} sum_i (<A_i,U>-y_i) A_i,
% which estimates mu*X. Only after the final iteration is the output divided
% by mu before computing its error relative to X.

fprintf('Simulation 1/3: truncated single-index model\n');

lambda = cfg.lambda;
mu = erf(lambda / sqrt(2));
m_values = cfg.m_values;
num_m = numel(m_values);
num_settings = numel(cfg.setting_names);
errors = nan(cfg.num_trials, num_m, num_settings);
iterations = nan(cfg.num_trials, num_m, num_settings);

for setting_idx = 1:num_settings
    setting_name = cfg.setting_names{setting_idx};
    setting = cfg.settings.(setting_name);
    n = setting.n;
    r = setting.r;
    dims = [n, n, n];
    ranks = [r, r, r];
    d = prod(dims);

    for trial = 1:cfg.num_trials
        rng(cfg.seed + 100000 * setting_idx + trial, 'twister');
        X = traic.generate_unit_tucker_tensor( ...
            n, r, cfg.max_condition, cfg.max_tensor_draws);
        A_all = traic.gaussian_design(max(m_values), d, ...
            cfg.design_precision, cfg.design_block_rows);
        z_all = traic.forward(A_all, X);
        y_all = max(min(z_all, lambda), -lambda);

        for m_idx = 1:num_m
            m = m_values(m_idx);
            A = A_all(1:m, :);
            y = y_all(1:m);

            state0 = traic.first_moment_initializer(A, y, dims, ranks);
            [state_hat, info] = traic.run_rgd(A, y, state0, ...
                'sim', cfg.max_iterations.sim, cfg.tolerance, ...
                cfg.min_iterations);

            % The raw SIM procedure estimates mu*X. Rescale only here.
            X_hat = state_hat.X / mu;
            errors(trial, m_idx, setting_idx) = norm(X_hat - X) / norm(X);
            iterations(trial, m_idx, setting_idx) = info.iterations;
        end

        if cfg.verbose
            fprintf('  %-18s trial %3d/%3d completed\n', ...
                setting_name, trial, cfg.num_trials);
        end
        clear A_all z_all y_all X
    end
end

output = struct();
output.model = 'Truncated single-index model';
output.lambda = lambda;
output.mu = mu;
output.m_values = m_values;
output.errors = errors;
output.iterations = iterations;
output.labels = cellfun(@(name) cfg.settings.(name).label, ...
    cfg.setting_names, 'UniformOutput', false);
output.setting_names = cfg.setting_names;
end
