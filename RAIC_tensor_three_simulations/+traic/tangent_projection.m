function projected = tangent_projection(Z, state)
%TANGENT_PROJECTION Project a full tensor onto the Tucker tangent space.
%
% This is the same tangent-space decomposition used in the original repo,
% written once as a reusable function and with I-UU' in place of null(U').

U = state.U;
S = state.core;
dims = state.dims;

Ut = {U{1}', U{2}', U{3}'};
projected_core = ttm(ttm(Z, Ut, [1, 2, 3]), U, [1, 2, 3]);

V = cell(1,3);
for mode = 1:3
    [V{mode}, ~] = qr(double(tenmat(S, mode))', 0);
end

W1 = kron(U{3}, U{2}) * V{1};
W2 = kron(U{3}, U{1}) * V{2};
W3 = kron(U{2}, U{1}) * V{3};

P1_perp = eye(dims(1)) - U{1} * U{1}';
P2_perp = eye(dims(2)) - U{2} * U{2}';
P3_perp = eye(dims(3)) - U{3} * U{3}';

Z1 = double(tenmat(Z, 1));
Z2 = double(tenmat(Z, 2));
Z3 = double(tenmat(Z, 3));

arm1_matrix = P1_perp * Z1 * W1 * W1';
arm2_matrix = P2_perp * Z2 * W2 * W2';
arm3_matrix = P3_perp * Z3 * W3 * W3';

arm1 = tensor(arm1_matrix, dims);
arm2 = permute(tensor(arm2_matrix, ...
    [dims(2), dims(1), dims(3)]), [2, 1, 3]);
arm3 = permute(tensor(arm3_matrix, ...
    [dims(3), dims(1), dims(2)]), [2, 3, 1]);

projected = projected_core + arm1 + arm2 + arm3;
end
