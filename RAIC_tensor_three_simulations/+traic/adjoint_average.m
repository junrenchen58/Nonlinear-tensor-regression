function Z = adjoint_average(A, weights, dims)
%ADJOINT_AVERAGE Compute m^{-1} sum_i weights_i A_i as a tensor.

m = size(A, 1);
if isa(A, 'single')
    vector = double(A' * single(weights(:))) / m;
else
    vector = (A' * weights(:)) / m;
end
Z = tensor(reshape(vector, dims));
end
