function cfg = experiment_config(profile)
%EXPERIMENT_CONFIG Configuration for the three tensor simulations.
%
% Edit this file to change dimensions, ranks, sample sizes, trial counts,
% or stopping criteria. All ground-truth tensors have unit Frobenius norm.

profile = lower(string(profile));

cfg = struct();
cfg.lambda = 1;                       % truncation threshold in the SIM
cfg.seed = 20260815;
cfg.max_condition = 5;                % reject unusually ill-conditioned cores
cfg.max_tensor_draws = 100;
cfg.design_precision = 'double';      % change to 'single' if memory is limited
cfg.design_block_rows = 256;
cfg.min_iterations = 3;
cfg.verbose = true;

switch profile
    case "quick"
        % Small configuration for checking installation and code paths.
        cfg.num_trials = 3;
        cfg.m_values = [200, 350, 600, 1000];
        cfg.max_iterations.sim = 8;
        cfg.max_iterations.relu = 12;
        cfg.max_iterations.onebit = 10;
        cfg.tolerance = 1e-7;

        cfg.settings.baseline = struct('n', 10, 'r', 2, ...
            'label', '$n=10,\ r=2$');
        cfg.settings.larger_rank = struct('n', 10, 'r', 3, ...
            'label', '$n=10,\ r=3$');
        cfg.settings.larger_dimension = struct('n', 12, 'r', 2, ...
            'label', '$n=12,\ r=2$');

    case "paper"
        % Intended settings for the final one-row, three-panel figure.
        % Increase num_trials to 30--50 for a final high-precision run.
        cfg.num_trials = 50;
        cfg.m_values = [1200, 1800, 2600, 3800, 5500, 8800];
        cfg.max_iterations.sim = 40;
        cfg.max_iterations.relu = 40;
        cfg.max_iterations.onebit = 40;
        cfg.tolerance = 1e-8;

        cfg.settings.baseline = struct('n', 20, 'r', 2, ...
            'label', '$n=20,\ r=2$');
        cfg.settings.larger_rank = struct('n', 28, 'r', 2, ...
            'label', '$n=28,\ r=2$');
        cfg.settings.larger_dimension = struct('n', 28, 'r', 4, ...
            'label', '$n=28,\ r=4$');

    otherwise
        error('Unknown profile "%s". Use "quick" or "paper".', char(profile));
end

cfg.setting_names = {'baseline', 'larger_rank', 'larger_dimension'};
end
