function state = tucker_retract(T, ranks)
%TUCKER_RETRACT Fixed-rank T-HOSVD retraction.
%
% The factor matrix in each mode is computed from the corresponding
% unfolding of the same input tensor.  Thus this is the ordinary truncated
% HOSVD used in the current manuscript, rather than the old second-order
% initializer or a sequential HOSVD.

if ~isa(T, 'tensor')
    T = tensor(T);
end

T_norm = norm(T);
if ~isfinite(T_norm) || T_norm <= eps
    error('The T-HOSVD input has zero or nonfinite Frobenius norm.');
end

if numel(ranks) ~= 3
    error('Only third-order tensors are supported.');
end

U = cell(1,3);
for mode = 1:3
    unfolding = double(tenmat(T, mode));
    [left_vectors, ~, ~] = svd(unfolding, 'econ');
    if size(left_vectors, 2) < ranks(mode)
        error('The requested Tucker rank exceeds an unfolding dimension.');
    end
    U{mode} = left_vectors(:, 1:ranks(mode));
end

core = ttm(T, {U{1}', U{2}', U{3}'}, [1, 2, 3]);
X_retracted = ttm(core, U, [1, 2, 3]);

state = struct();
state.X = tensor(X_retracted);
state.U = U;
state.core = tensor(core);
state.ranks = ranks;
state.dims = size(state.X);
end
