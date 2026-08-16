for mode = 1:3
    
    Xmat = tenmat(img, mode);
    
    
    s = svd(double(Xmat));
    
     
    figure;
    semilogy(s, 'o-','LineWidth',1.5);
    xlabel(sprintf('Component index (mode-%d)', mode));
    ylabel('Singular value (log scale)');
    title(sprintf('Singular value decay along mode-%d', mode));
    grid on;
end