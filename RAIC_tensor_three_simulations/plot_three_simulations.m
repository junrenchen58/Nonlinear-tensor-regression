function plot_three_simulations(results, output_dir)
%PLOT_THREE_SIMULATIONS Create the final one-row, three-panel figure.

fig = figure('Color', 'w', 'Position', [80, 100, 1500, 430]);
tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

marker_list = {'o', 's', '^', 'd', 'v'};
line_list = {'-', '--', '-.', ':', '-'};

% Panel (a): truncated single-index model.
nexttile;
med = squeeze(mean(results.single_index.errors, 1, 'omitnan'));
for j = 1:size(med, 2)
    loglog(results.single_index.m_values, max(med(:,j), 1e-12), ...
        'LineWidth', 1.8, 'Marker', marker_list{j}, ...
        'LineStyle', line_list{j}, 'MarkerSize', 6);
    hold on;
end
grid on; box on;
xlabel('Sample size $m$', 'Interpreter', 'latex');
ylabel('Relative Frobenius error', 'Interpreter', 'latex');
title('(a) Truncated single-index model', 'Interpreter', 'latex');
legend(results.single_index.labels, 'Interpreter', 'latex', ...
    'Location', 'southwest', 'FontSize', 9);
set(gca, 'FontSize', 11);

% Panel (b): ReLU regression.
nexttile;
med = squeeze(mean(results.relu.errors, 1, 'omitnan'));
for j = 1:size(med, 2)
    loglog(results.relu.m_values, max(med(:,j), 1e-12), ...
        'LineWidth', 1.8, 'Marker', marker_list{j}, ...
        'LineStyle', line_list{j}, 'MarkerSize', 6);
    hold on;
end
grid on; box on;
xlabel('Sample size $m$', 'Interpreter', 'latex');
ylabel('Relative Frobenius error', 'Interpreter', 'latex');
title('(b) ReLU regression', 'Interpreter', 'latex');
legend(results.relu.labels, 'Interpreter', 'latex', ...
    'Location', 'southwest', 'FontSize', 9);
set(gca, 'FontSize', 11);

% Panel (c): one-bit tensor sensing.
nexttile;
med = squeeze(mean(results.onebit.errors, 1, 'omitnan'));
for j = 1:size(med, 2)
    loglog(results.onebit.m_values, max(med(:,j), 1e-12), ...
        'LineWidth', 1.8, 'Marker', marker_list{j}, ...
        'LineStyle', line_list{j}, 'MarkerSize', 6);
    hold on;
end
grid on; box on;
xlabel('Sample size $m$', 'Interpreter', 'latex');
ylabel('Relative Frobenius error', 'Interpreter', 'latex');
title('(c) One-bit tensor sensing', 'Interpreter', 'latex');
legend(results.onebit.labels, 'Interpreter', 'latex', ...
    'Location', 'southwest', 'FontSize', 8);
set(gca, 'FontSize', 11);

exportgraphics(fig, fullfile(output_dir, 'three_simulations.pdf'), ...
    'ContentType', 'vector');
exportgraphics(fig, fullfile(output_dir, 'three_simulations.png'), ...
    'Resolution', 300);
savefig(fig, fullfile(output_dir, 'three_simulations.fig'));
end
