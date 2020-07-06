function [w, w0] = svm(traindata, trainclass, C)
% Simple SVM method for finding boundary
%
%     'traindata' - training data.
%    'trainclass' - classes for train data.
%             'C' - parameter that controls the penalty associated 
%                   with an incorrect classification
EPS = 0.0001;

N = size(trainclass, 2);
y = ones(N, 1); y(trainclass == 2) = -1;
x = traindata;

% Params for quadprog
H = zeros(N);
for i = 1:N
    for j = 1:N
        H(i, j) = y(i) * y(j) * x(:, i)' * x(:, j);
    end
end

f = -1*ones(N, 1);
Aeq = y';
beq = 0;
LB = zeros(N, 1);
UB = C * ones(N, 1);

lambda = quadprog(H, f, [], [], Aeq, beq, LB, UB);

% Calculate w
w = zeros(size(x, 1),1);
for i = 1:N
    w = w + lambda(i) * y(i) * x(:,i);
end

% Calculate w0
xk = 0;
for i = 1:N
    if (trainclass(i) == 1 && abs(lambda(i)) > EPS)
        abs(lambda(i))
        xk = x(:, i);
        break;
    end
end
w0 = 1 - xk'*w;

end