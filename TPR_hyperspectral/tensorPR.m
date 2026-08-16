load hyper.mat;
X = tensor(X);
U = randn(20,1); % not important!
sigma = 0.0; % noiseless case 

%% Initialization;
n = 30;
r = 10;
X0 = tensor(X0);
X0ho = hosvd(X0,norm(X0),'ranks',[r,r,r],'sequential',true,'verbosity',0);
U0 = X0ho.u;

%% sensing process
m = round(n^3/4); % measurement number
A = tensor(randn(n,n,n,m));
A_mat = tenmat(A, 4);
A_mat = A_mat.data;
epi = sigma * randn(m,1);
y = A_mat * X(:);
y = abs(y)+epi;

%% Local contraction
[errmat,recoverimg] = tpr_localreal(A_mat,y,X0,U0,X,U,n,n,n,r,r,r,50);
