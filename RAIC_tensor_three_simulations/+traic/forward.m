function values = forward(A, X)
%FORWARD Apply the Gaussian tensor-design matrix to a tensor.

X_array = double(X);
x = X_array(:);
if isa(A, 'single')
    values = double(A * single(x));
else
    values = A * x;
end
end
