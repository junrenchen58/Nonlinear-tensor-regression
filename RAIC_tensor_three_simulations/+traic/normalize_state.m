function state = normalize_state(state, target_norm)
%NORMALIZE_STATE Radially rescale a Tucker state to target Frobenius norm.

current_norm = norm(state.X);
if ~isfinite(current_norm) || current_norm <= eps
    error('Cannot normalize a zero or nonfinite tensor iterate.');
end
scale = target_norm / current_norm;
state.X = state.X * scale;
state.core = state.core * scale;
end
