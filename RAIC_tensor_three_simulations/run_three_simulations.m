function results = run_three_simulations(profile)
%RUN_THREE_SIMULATIONS Run the three end-to-end tensor simulations.
%
%   RESULTS = RUN_THREE_SIMULATIONS('quick') runs a small smoke test.
%   RESULTS = RUN_THREE_SIMULATIONS('paper') runs the intended paper-scale
%   configuration. The settings can be changed in experiment_config.m.
%
%   Required package: MATLAB Tensor Toolbox.

if nargin < 1 || isempty(profile)
    profile = 'quick';
end

root_dir = fileparts(mfilename('fullpath'));
addpath(root_dir);

if exist('tensor', 'class') ~= 8 && exist('tensor', 'file') ~= 2
    error(['MATLAB Tensor Toolbox was not found. Add it to the MATLAB path, ' ...
        'then rerun this function.']);
end

cfg = experiment_config(profile);
cfg.output_dir = fullfile(root_dir, 'results', char(profile));
if ~exist(cfg.output_dir, 'dir')
    mkdir(cfg.output_dir);
end

fprintf('\n============================================================\n');
fprintf('Nonlinear low-rank tensor simulations: %s profile\n', upper(char(profile)));
fprintf('Trials: %d; sample sizes: %s\n', cfg.num_trials, mat2str(cfg.m_values));
fprintf('Output directory: %s\n', cfg.output_dir);
fprintf('============================================================\n\n');

results = struct();
results.config = cfg;
results.single_index = simulate_truncated_single_index(cfg);
save(fullfile(cfg.output_dir, 'checkpoint_single_index.mat'), ...
    'results', '-v7.3');

results.relu = simulate_relu_regression(cfg);
save(fullfile(cfg.output_dir, 'checkpoint_relu.mat'), ...
    'results', '-v7.3');

results.onebit = simulate_onebit_sensing(cfg);
save(fullfile(cfg.output_dir, 'three_simulation_results.mat'), ...
    'results', '-v7.3');

plot_three_simulations(results, cfg.output_dir);

fprintf('\nCompleted all simulations.\n');
fprintf('Raw results: %s\n', ...
    fullfile(cfg.output_dir, 'three_simulation_results.mat'));
fprintf('Figure:      %s\n\n', ...
    fullfile(cfg.output_dir, 'three_simulations.pdf'));
end
