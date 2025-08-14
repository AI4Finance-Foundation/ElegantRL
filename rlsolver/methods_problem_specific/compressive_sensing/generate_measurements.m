M = 100; N = 100; %% Dimension
S = 0.05; %% Signal Sparsity
A = rand(M, N); %% Measurement Matrix
Phi = dctmtx(N); %% DCT Matrix

A_Phi = A * Phi; %% Measurement Matrix on Latent Space
x_origin = rand(N, 1); %% Origin Signal
mask = randperm(N);   %% Generate random mask for sparsity
x_origin(mask(int8(N*S + 1):end)) = 0; %% Mask S*N to N index in x_origin to zero

save('.\cs_data\A',"A");
save('.\cs_data\Phi',"Phi");
save('.\cs_data\x_origin',"x_origin");
