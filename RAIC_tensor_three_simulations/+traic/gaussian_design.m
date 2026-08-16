function A = gaussian_design(m, d, precision, block_rows)
%GAUSSIAN_DESIGN Generate an m-by-d i.i.d. standard Gaussian design.
%
% Rows are vectorized Gaussian tensor covariates. The default single
% precision substantially reduces memory, while all tensor iterates and
% reported errors remain in double precision.

if nargin < 4 || isempty(block_rows)
    block_rows = 256;
end

precision = lower(string(precision));
if precision == "single"
    A = zeros(m, d, 'single');
else
    A = zeros(m, d);
end

for first_row = 1:block_rows:m
    last_row = min(m, first_row + block_rows - 1);
    rows = first_row:last_row;
    if precision == "single"
        try
            A(rows, :) = randn(numel(rows), d, 'single');
        catch
            A(rows, :) = single(randn(numel(rows), d));
        end
    else
        A(rows, :) = randn(numel(rows), d);
    end
end
end
