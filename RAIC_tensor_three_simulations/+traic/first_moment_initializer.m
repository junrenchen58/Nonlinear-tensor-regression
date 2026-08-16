function state = first_moment_initializer(A, y, dims, ranks)
%FIRST_MOMENT_INITIALIZER First-moment T-HOSVD initializer.
%
% Forms m^{-1} sum_i y_i A_i and applies rank-(r1,r2,r3) T-HOSVD.

moment = traic.adjoint_average(A, y, dims);
state = traic.tucker_retract(moment, ranks);
end
