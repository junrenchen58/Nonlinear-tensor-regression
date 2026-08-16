load Indian_pines_corrected.mat   
X = double(indian_pines_corrected);    
 
X = X(61:90, 61:90, 6:35);
X = X/norm(X,'fro'); 


 
for mode = 1:3
    Xmat = tenmat(tensor(X), mode);
    s = svd(double(Xmat),'econ');
    figure; semilogy(s,'-o','LineWidth',1.6,'MarkerSize',5); grid on;
    title(sprintf('Mode-%d singular values',mode));
    xlabel('Index'); ylabel('Singular value');
end

