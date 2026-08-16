% Order-3 scalar-on-tensor regression via RGD.
% Output: iteration, time and error
% Inputs: A_mat: m-by-n^2 matrix, each row is a vectorization of the covariate tensor; 
% y: observation; X0: initialization of X; 
% X: true parameter of interest;  
% iter_max: the maximum number of iteration; succ_tol: stopping criteria
% This is the local contraction function for matrix phase retrieval
function [ error_matrix ,Xt] = vpr_localreal( A_mat,y,X0,X,n,iter_max)
Xt = X0;
Xt_err = min(norm(Xt - X,'fro'),norm(Xt + X,'fro'))/norm(X,'fro');
m = size(A_mat,1);
error_matrix = [Xt_err];
for iter = 1:iter_max    
    haty = abs(A_mat * Xt(:));
    Z = reshape((haty - y)'*((sign(A_mat * Xt(:)).*A_mat))/m,[n,n^2]);
    alpha_t = 1;
    tildeXt = Xt - alpha_t * Z;
    Xt_err_new = min(norm(tildeXt - X,'fro'),norm(tildeXt + X,'fro'))/norm(X,'fro');
    Xt_err = Xt_err_new;
    error_matrix =[error_matrix,Xt_err];
end
end