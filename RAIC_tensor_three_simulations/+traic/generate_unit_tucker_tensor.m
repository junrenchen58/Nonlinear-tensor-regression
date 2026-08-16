function X = generate_unit_tucker_tensor(n, r, max_condition, max_draws)
%GENERATE_UNIT_TUCKER_TENSOR Random, unit-Frobenius, low-Tucker-rank tensor.
%
% A random Gaussian core is redrawn if its mode-wise condition number is
% unusually large. This keeps comparisons across n and r from being driven
% by a rare nearly singular core.

for draw = 1:max_draws
    U = cell(1,3);
    for k = 1:3
        [Q, ~] = qr(randn(n, r), 0);
        U{k} = Q;
    end

    core = tensor(randn(r, r, r));
    X_candidate = ttm(core, U, [1, 2, 3]);
    X_candidate = X_candidate / norm(X_candidate);

    largest = 0;
    smallest = inf;
    for k = 1:3
        singular_values = svd(double(tenmat(X_candidate, k)), 'econ');
        singular_values = singular_values(1:min(r, numel(singular_values)));
        largest = max(largest, max(singular_values));
        smallest = min(smallest, min(singular_values));
    end

    if smallest > 0 && largest / smallest <= max_condition
        X = X_candidate;
        return;
    end
end

warning(['Could not draw a tensor with condition number at most %.2f ' ...
    'after %d attempts. Using the last draw.'], max_condition, max_draws);
X = X_candidate;
end
