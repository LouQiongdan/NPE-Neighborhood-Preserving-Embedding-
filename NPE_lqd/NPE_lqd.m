function [ mappedX, mapping ] = NPE_lqd( X, new_lower_dimension, k )
%UNTITLED2 此处显示有关此函数的摘要
%Input:
%   X: the original data ,whose size is n*d , n is the number of samples and  d is the number of features.
%   new_lower_dimension:the new lower dimension of X we wanna get after NPE.
%   k:the number of nearest neighborhood.
%Output:
%   mappedX:the new X after the process of NPE,whose features' dimension is new_lower_dimension. 
%   mapping:the struct of mapping.
%   此处显示详细说明
%文章来自：Neighborhood Preserving Embedding . Xiaofei He , Deng Cai, Shuicheng Yan, Hong-Jiang Zhang

if size(X, 1) < size(X, 2)
    error('The number of samples shoule be higher than  number of dimensions');
end
if ~exist('new_lower_dimension','var')
    new_lower_dimension = 2;
end
if ~exist('k','var') 
    k = 10;
end

% Get dimensionality and number of dimensions
[ n , d ] = size(X);
mapping.mean = mean(X, 1);

 % Compute pairwise distances and find nearest neighbours (vectorized implementation)
 disp('Finding nearest neighbors...');
 [distance,neighborhood] = find_nn(X, k);%distance:N*N维度；neighborhood:N*k维度,放的是对应的坐标即样本编号
 max_k = size(neighborhood, 2);
 if nargout > 1 
    mapping.nbhd = distance;
 end
 
    % Find reconstruction weights for all points by solving the MSE problem 
    % of reconstructing a point from each neighbours . A used constraint is 
    % that the sum of the reconstruction weights for a point should be 1.
disp('Compute reconstruction weights...');
if k > d 
    tol = 1e-5;
else
    tol = 0;
end

X = X';
neighborhood = neighborhood';

% Construct reconstruction weight matrix
W = zeros(max_k, n);
for i = 1 : n 
    nbhd = neighborhood(:, i);
    if ischar(k)
        nbhd = nbhd(nbhd ~= 0)
    end
    kt = numel(nbhd);
    z = X(:,nbhd) - repmat(X(:, i), 1, kt);
    C = z' * z;
    C = C + eye (kt, kt) * tol * trace(C);
    wi = C \ ones(kt, 1);
    wi = wi / sum(wi);
    W(:, i) = [wi; nan(max_k - kt, 1)];
end

% Now that we have the reconstruction weights matrix, we define the 
% sparse cost matrix M = (I-W)'*(I-W).
M = sparse(1:n, 1:n, ones(1, n), n, n, 4 * max_k * n);
for i = 1 : n 
    w = W(:, i);
    j = neighborhood(: , i);
    w = w(j ~= 0);
    j = j(j ~= 0);
    M(i, j) = M(i, j) - w';
    M(j, i) = M(j, i) - w;
    M(j, j) = M(j, j) + w * w';
end

% For sparse datasets, we might end up with NaNs or Infs in M. We just set them to zero for now...
M(isnan(M)) = 0;
M(isinf(M)) = 0;

% Compute XWX and XX and make sure these are symmetric
X = X';
WP = X' * M * X;
DP = X' * X;
DP = (DP + DP') / 2;
WP = (WP + WP') / 2;

% Solve generalized eigenproblem
if size(X, 1) > 1500 && new_lower_dimension < (size(X, 1) / 10)
    option.disp = 0;
    options.issym = 1;
    options.isreal = 0;
    [eigvector, eigvalue] = eigs(WP, DP, new_lower_dimension, 'sa', options);
else
    [eigvector, eigvalue] = eig(WP, DP);
end

% Sort eigenvalues in descending order and get smallest eigenvectors
[eigvalue, ind] = sort(diag(eigvalue),'ascend');
eigvector = eigvector(:,ind(1:new_lower_dimension));

% Compute final linear basis and map data
mappedX = X * eigvector;
mapping.M = eigvector;

end

