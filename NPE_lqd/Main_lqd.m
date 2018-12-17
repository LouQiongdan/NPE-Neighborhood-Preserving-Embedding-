load('D:\Lqd_CX\日常降维学习算法\自己写的\NPE_lqd\data\usps_resampled.mat');
X=train_patterns';
no_dims=10;
k=9;
[mappedX, mapping] = NPE_lqd(X, no_dims, k);