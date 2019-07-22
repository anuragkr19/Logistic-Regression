function [theta] = gradientDescent(X, y, theta, alpha, num_iters)
%gradientDescent performs gradient descent to learn theta

% Initialization
m = length(y); % number of training examples

%store gradient values
grad = zeros(size(theta));

%For each iteration compute the gradient on complete dataset and update the
%parameters by num_iters gradient steps with learning rate alpha

for iter = 1:num_iters
   h = sigmoid(X*theta);
   grad = (alpha./m).*(h - y)'*X;
   theta = theta - grad';
end

end
