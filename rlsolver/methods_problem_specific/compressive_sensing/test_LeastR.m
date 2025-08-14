%% Data Initialization 
N = 100; %% Dimension
S = 0.05; %% Signal Sparsity
%% x_origin
mat = load("cs_data\x_origin.mat");
x_origin = mat.x_origin;
%% DCT Matrix
mat = load("cs_data\Phi.mat");
Phi = mat.Phi;
%% Termination Settings
opts = [];
opts.tol = 1e-6;
opts.maxIter = 5000;
lambda = 0.001;
for M = 30:30:90
    %% Measurement Matrix
    mat = load("cs_data\A.mat");
    A = mat.A(1:M, :);
    %% Measurement Matrix on Latent Space
    A_Phi = A * Phi; %% Measurement Matrix on Latent Space
    y = A*x_origin();       
    %% Lasso
    [theta, funVal] = LeastR(A_Phi, y, lambda, opts);
    %% Lasso
    fprintf("M=%d, Number of Iterations: %d\n", M, length(funVal));
    %disp(funVal(end-10:end));
    fprintf("l2-norm(X_origin - X_recon): %f\n", norm(Phi*theta - x_origin));
    fprintf("l2-norm(X_recon): %f\n", norm(Phi*theta));
    fprintf("l2-norm(X_origin): %f\n", norm(x_origin));
end

