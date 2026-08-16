function [meanitererror,errl] = tpr_real(n,r,m,sigma,ini,numiter,seed,repe)
% Tensor Phase Retrieval of hyperspectral images 
errlrgd = zeros(repe,numiter+1); 
rng('default');
for i = 1:repe
    rng(seed);
    seed = seed + 1;
    p1 = n; p2 = n; p3 = n;
    r1 = r; r2 = r; r3 = r;
    %% data generation
    S = tensor(randn(r1, r2, r3));
    S = S/norm(S); % ensure parameter norm 1
    E1 = randn(p1, r1);
    E2 = randn(p2, r2);
    E3 = randn(p3, r3);
    [U1,~,~] = svds(E1,r1);
    [U2,~,~] = svds(E2,r2);
    [U3,~,~] = svds(E3,r3);
    A = tensor(randn(p1, p2, p3,m));
    U = {U1, U2, U3};
    X = ttm(S, U, [1:3]);
    A_mat = tenmat(A, 4);
    A_mat = A_mat.data;
    epi = sigma * randn(m,1);
    y = A_mat * X(:);
    y = abs(y)+epi; %% the nonlinearity
    
    %% initialization
    S0t = randn(p1, p2, p3);
    S0t = S0t/norm(S0t(:));
    S0t = tensor(S0t);
    Xt = ini*X + (1-ini)*S0t; % ini = 0: random inti.; ini = 1: correct
    Xtt = hosvd(Xt,norm(Xt),'ranks',[r1,r2,r3],'sequential',true,'verbosity',0);
    Ut = Xtt.u;
    Stt = Xtt.core; 
    Xt = ttm(Stt, Ut, [1:3]);
    Xt = Xt/norm(Xt);

    %% RGD for Tensor Single Index Models
    errlrgd(i,:) = tpr_local(A_mat,y,Xt, Ut,X, U, p1, p2, p3,r1,r2,r3,numiter);          
end

%% results
errl = errlrgd(:,numiter+1);
meanitererror = mean(errlrgd);
 
 

