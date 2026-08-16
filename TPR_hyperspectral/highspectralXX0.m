load Indian_pines_corrected.mat   
Xindian = double(indian_pines_corrected);   
 
X = Xindian(61:90, 61:90, 6:35);
X = X/norm(X,'fro');
X0 = Xindian(61:90, 61:90, 36:65);
X0 = X0/norm(X0,'fro');
norm(X-X0,'fro')

% Imshow true
imgs = X / max(max(max(X)));
imgs = reshape(imgs, 30, 30, 1, 30);
montage(imgs, 'Size', [5 6]);

% Imshow initialization
figure
imgs = X0 / max(max(max(X0)));
imgs = reshape(imgs, 30, 30, 1, 30);
montage(imgs, 'Size', [5 6]);

save hyper.mat X X0

 
