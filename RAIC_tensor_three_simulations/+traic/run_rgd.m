function [state, info] = run_rgd(A, y, state, method, ...
    max_iterations, tolerance, min_iterations)
%RUN_RGD Run RGD/NRGD with one of the paper's exact gradients.
%
% method = 'sim':
%   raw squared-loss gradient m^{-1} sum_i (<A_i,U>-y_i) A_i;
%   no normalization and no use of mu inside the algorithm.
%
% method = 'relu':
%   m^{-1} sum_i [ReLU(<A_i,U>)-y_i]
%                 [1+sign(<A_i,U>)] A_i;
%   no normalization.
%
% method = 'onebit':
%   sqrt(pi/2) m^{-1} sum_i [sign(<A_i,U>)-y_i] A_i;
%   the retracted iterate is normalized to unit Frobenius norm.

method = lower(string(method));
update_history = nan(max_iterations, 1);

for iter = 1:max_iterations
    prediction = traic.forward(A, state.X);

    switch method
        case "sim"
            weights = prediction - y;
            Z = traic.adjoint_average(A, weights, state.dims);
            normalize_after_update = false;

        case "relu"
            relu_prediction = max(prediction, 0);
            weights = (relu_prediction - y) .* (1 + sign(prediction));
            Z = traic.adjoint_average(A, weights, state.dims);
            normalize_after_update = false;

        case "onebit"
            weights = traic.binary_sign(prediction) - y;
            Z = sqrt(pi/2) * traic.adjoint_average(A, weights, state.dims);
            normalize_after_update = true;

        otherwise
            error('Unknown method "%s".', method);
    end

    riemannian_gradient = traic.tangent_projection(Z, state);
    candidate = state.X - riemannian_gradient;  % fixed step size one
    new_state = traic.tucker_retract(candidate, state.ranks);

    if normalize_after_update
        new_state = traic.normalize_state(new_state, 1);
    end

    update_history(iter) = norm(new_state.X - state.X) / ...
        max(norm(state.X), eps);
    state = new_state;

    if iter >= min_iterations && update_history(iter) <= tolerance
        break;
    end
end

info = struct();
info.iterations = iter;
info.relative_updates = update_history(1:iter);
end
