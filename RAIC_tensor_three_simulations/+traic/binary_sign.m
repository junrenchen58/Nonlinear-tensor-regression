function y = binary_sign(x)
%BINARY_SIGN Sign map taking values in {-1,+1}, including at zero.

y = ones(size(x));
y(x < 0) = -1;
end
