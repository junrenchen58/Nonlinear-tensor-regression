load hyper.mat;
X = reshape(X, 30, []);
X0 = reshape(X0, 30,[]);
sigma = 0.0; % noiseless case 

%% sensing process
n = 30; 
m = round(30^3/1.5); % measurement number
A_mat = randn(m, n^3);
epi = sigma * randn(m,1);
y = A_mat * X(:);
y = abs(y)+epi;

%% VPR
[error_matrix ,Xt] = vpr_localreal( A_mat,y,X0,X,n, 50);
X = reshape(X,[30,30,30]);
recoverimg = reshape(Xt,[30,30,30]);
