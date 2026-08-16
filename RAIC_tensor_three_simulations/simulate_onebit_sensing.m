function output = simulate_onebit_sensing(cfg)
%SIMULATE_ONEBIT_SENSING End-to-end one-bit tensor sensing experiment.
%
% Curves 1--3 use the model-specific one-bit gradient for the baseline,
% larger-rank, and larger-dimension settings. Curve 4 uses the exact generic
% SIM squared-loss gradient for the baseline setting. The latter estimates
% mu*X, mu=sqrt(2/pi), and is divided by mu only after the final iteration.

fprintf('Simulation 3/3: one-bit tensor sensing\n');

m_values = cfg.m_values;
num_m = numel(m_values);
num_settings = numel(cfg.setting_names);
mu = sqrt(2/pi);

% Three one-bit curves plus one baseline generic-SIM curve.
errors = nan(cfg.num_trials, num_m, num_settings + 1);
iterations = nan(cfg.num_trials, num_m, num_settings + 1);

for setting_idx = 1:num_settings
    setting_name = cfg.setting_names{setting_idx};
    setting = cfg.settings.(setting_name);
    n = setting.n;
    r = setting.r;
    dims = [n, n, n];
    ranks = [r, r, r];
    d = prod(dims);

    for trial = 1:cfg.num_trials
        rng(cfg.seed + 300000 + 100000 * setting_idx + trial, 'twister');
        X = traic.generate_unit_tucker_tensor( ...
            n, r, cfg.max_condition, cfg.max_tensor_draws);
        A_all = traic.gaussian_design(max(m_values), d, ...
            cfg.design_precision, cfg.design_block_rows);
        z_all = traic.forward(A_all, X);
        y_all = traic.binary_sign(z_all);

        for m_idx = 1:num_m
            m = m_values(m_idx);
            A = A_all(1:m, :);
            y = y_all(1:m);

            raw_state0 = traic.first_moment_initializer(A, y, dims, ranks);

            % Model-specific one-bit NRGD: unit-norm initialization and
            % unit-norm rescaling after every Riemannian update.
            onebit_state0 = traic.normalize_state(raw_state0, 1);
            [onebit_state, onebit_info] = traic.run_rgd(A, y, ...
                onebit_state0, 'onebit', cfg.max_iterations.onebit, ...
                cfg.tolerance, cfg.min_iterations);
            errors(trial, m_idx, setting_idx) = ...
                norm(onebit_state.X - X) / norm(X);
            iterations(trial, m_idx, setting_idx) = onebit_info.iterations;

            % The generic SIM comparison is included only for the baseline.
            if setting_idx == 1
                [sim_state, sim_info] = traic.run_rgd(A, y, raw_state0, ...
                    'sim', cfg.max_iterations.sim, cfg.tolerance, ...
                    cfg.min_iterations);
                X_hat_sim = sim_state.X / mu;
                errors(trial, m_idx, num_settings + 1) = ...
                    norm(X_hat_sim - X) / norm(X);
                iterations(trial, m_idx, num_settings + 1) = ...
                    sim_info.iterations;
            end
        end

        if cfg.verbose
            fprintf('  %-18s trial %3d/%3d completed\n', ...
                setting_name, trial, cfg.num_trials);
        end
        clear A_all z_all y_all X
    end
end

labels = cellfun(@(name) ...
    ['One-bit gradient: ' cfg.settings.(name).label], ...
    cfg.setting_names, 'UniformOutput', false);
labels{end+1} = ['SIM gradient: ' cfg.settings.baseline.label];

output = struct();
output.model = 'One-bit tensor sensing';
output.mu_sim = mu;
output.m_values = m_values;
output.errors = errors;
output.iterations = iterations;
output.labels = labels;
output.setting_names = [cfg.setting_names, {'baseline_sim'}];
end
