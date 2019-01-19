function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X];         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m, 1) sigmoid(X*Theta1')];
h_t = sigmoid(a1*Theta2');

y_v = [1:num_labels];

for i = 1:m
yt = (y_v == y(i,1));
J = J + sum(-yt.*log(h_t(i,:))-(1-yt).*log(1-h_t(i,:)));
endfor
J= (1/m)*J;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
D2=0;
D3=0;
for i = 1:m
  z2= X(i,:)*Theta1';
  a2=[1  sigmoid(z2)];
  yt = (y_v == y(i,1));
  d3=(sigmoid(a2*Theta2')-yt);
  D3= D3 + d3'*a2;
  d2 =((d3*Theta2) .*[1  sigmoidGradient(z2)]);
   D2= D2 + d2(2:end)'*X(i,:);
endfor
Theta2_grad=(1/m)* D3;
Theta1_grad=(1/m)* D2;

%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
J=J+(lambda/(2*m))*(sum((Theta1(:,2:end).^2)(:))+sum((Theta2(:,2:end).^2)(:)));

for i = 1:rows(Theta1_grad)
  for j = 2:columns(Theta1_grad)
Theta1_grad(i,j) = Theta1_grad(i,j)+ (lambda/m)*Theta1(i,j);
endfor
endfor

for i = 1:rows(Theta2_grad)
  for j = 2:columns(Theta2_grad)
Theta2_grad(i,j) = Theta2_grad(i,j)+ (lambda/m)*Theta2(i,j);
endfor
endfor

%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
