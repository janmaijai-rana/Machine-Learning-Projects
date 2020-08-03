function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid((X*theta));
a = size(theta,1);
k = zeros(a-1,1);
for i=2:a
  k(i-1)=theta(i).^2;
endfor  
J = (sum((-y.*log(h))-((1-y).*log(1-h)))/m)+(lambda/(2*m))*sum(k);

f = size(X,1);

x = sum((h-y).*X(1:f)')/m;
y = sum((h-y).*X)/m+(lambda/m)*(theta)';
grad(1) = x;
for i=2:a
  grad(i) = y(i);


% =============================================================

end
