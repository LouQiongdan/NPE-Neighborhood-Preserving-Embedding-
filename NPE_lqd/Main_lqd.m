load('D:\Lqd_CX\�ճ���άѧϰ�㷨\�Լ�д��\NPE_lqd\data\usps_resampled.mat');
X=train_patterns';
no_dims=10;
k=9;
[mappedX, mapping] = NPE_lqd(X, no_dims, k);