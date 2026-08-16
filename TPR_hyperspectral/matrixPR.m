load hyper.mat;
X = reshape(X, 30,[]);
%X = [X(:,1:180);X(:,181:360);X(:,361:540);X(:,541:720);X(:,721:900)];
X0 = reshape(X0,30,[]);
%X0 = [X0(:,1:180);X0(:,181:360);X0(:,361:540);X0(:,541:720);X0(:,721:900)];
sigma = 0.0; % noiseless case 

%% sensing process
n = 30;
r = 10;
m = round(30^3/1.5); % measurement number
A_mat = randn(m, n^3);
epi = sigma * randn(m,1);
y = A_mat * X(:);
y = abs(y)+epi;

%% MPR
[error_matrix ,Xt] = mpr_localreal( A_mat,y,X0,X,r, 50);
X = reshape(X,[30,30,30]);
recoverimg = reshape(Xt,[30,30,30]);
