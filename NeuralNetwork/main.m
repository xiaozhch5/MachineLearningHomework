clc
clear all;
N = 100;
% 单个高斯函数
mu1 = [0, 0];
sigma1 = [3, -0.5; -0.5, 3];
r1 = mvnrnd(mu1, sigma1, N);
% 两个高
p=[0.7, 0.3];
% MU1 = [1, -1];
% MU2 = [1, 1];
 mu2 = [1 -1; 1 1];
SIGMA1 = [0.8 0.5; 0.5 0.8];
SIGMA2 = [0.9 -0.5; -0.5 0.9];
X = cat(3, SIGMA1, SIGMA2);
gm = gmdistribution(mu2, X, p);
[r2, compIdx] = random(gm, N);

options = statset('Display','final');
obj1 = gmdistribution.fit(r1,1,'Options',options);
% h1 = ezcontour(@(x1,y1)pdf(obj1,[x1 y1]),[-8 6],[-8 6]);
obj2 = gmdistribution.fit(r2,2,'Options',options);
% h2 = ezcontour(@(x1,y1)pdf(obj1,[x2 y2]),[-8 6],[-8 6]);
%画出最原始的两个高斯分布散点图
plot(r1(:, 1), r1(:, 2), '*', r2(:,1),r2(:,2),  '+');
hold on;
W1 = -1/2 * obj1.Sigma^-1;
w1 =obj1.Sigma^-1 * obj1.mu';
w10 = -1/2 * obj1.mu * obj1.Sigma^-1 * obj1.mu' - 1/2 * log(det(obj1.Sigma)) + log(1/2);
%
W2 = -1/2 * obj2.Sigma(:,:,1)^-1;
w2 =obj2.Sigma(:,:,1)^-1 * obj2.mu(1,:)';
w20 = -1/2 * obj2.mu(1,:) * obj2.Sigma(:,:,1)^-1 * obj2.mu(1,:)' - 1/2 * log(det(obj2.Sigma(:,:,1))) + log(1/2);
%
W3 = -1/2 * obj2.Sigma(:,:,2)^-1;
w3 =obj2.Sigma(:,:,2)^-1 * obj2.mu(2,:)';
w30 = -1/2 * obj2.mu(2,:) * obj2.Sigma(:,:,2)^-1 * obj2.mu(2,:)' - 1/2 * log(det(obj2.Sigma(:,:,2))) + log(1/2);

t2=[];
for t1=-2.5:2.5
    tt2 = fsolve('bayesian_fun',5,[],t1,W1,W2,W3,w1,w2,w3,w10,w20,w30,p);
    t2 = [t2, tt2];
end
plot(-2.5:2.5,t2,'b','LineWidth',3);
axis([-2.5 2.5 -2.5 2.5])
hold on;

%最优
W11 = -1/2 * sigma1^-1;
w11 =sigma1^-1 * mu1';
w110 = -1/2 * mu1 * sigma1^-1 * mu1' - 1/2 * log(det(sigma1)) + log(1/2);
%
W22 = -1/2 * SIGMA1^-1;
w22 =SIGMA1^-1 * mu2(1,:)';
w220 = -1/2 * mu2(1,:) * SIGMA1^-1 * mu2(1,:)' - 1/2 * log(det(SIGMA1)) + log(1/2);
%
W33 = -1/2 * SIGMA2^-1;
w33 =SIGMA2^-1 * mu2(2,:)';
w330 = -1/2 * mu2(2,:) * SIGMA2^-1 * mu2(2,:)' - 1/2 * log(det(SIGMA2)) + log(1/2);

t22=[];
for t11=-2.5:2.5
    tt22 = fsolve('bayesian_fun',5,[],t11,W11,W22,W33,w11,w22,w33,w110,w220,w330,p);
    t22 = [t22, tt22];
end
plot(-2.5:2.5,t22,'g','LineWidth',3);

