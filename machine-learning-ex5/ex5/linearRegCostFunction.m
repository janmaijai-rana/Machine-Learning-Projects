function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X*theta;
j1 = (1/(2*m))*sum((h-y).^2);
theta1 = theta;
theta1(1) = 0;
j2 = (lambda/(2*m))*sum(theta1.^2);
J = j1+j2;

grad(1) = (1/m)*sum((h-y).*X(:,1));
theta2 = theta(2:end);
d = (lambda/m)*theta2;
x = X(:,2:end);
th = (1/m)*(x'*(h-y));
q = d + th;
grad(2:end) = q;
% =========================================================================

grad = grad(:);

end
