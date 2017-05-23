function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% size(p) = 5000 x 1
% size(X) = 5000 x 401
% m = 5000
% num_labels = 10

% size(Theta1) = 25 x 401
% size(Theta2) = 10 x 26

a1 = [ones(m, 1) X]; % add a0(1) to input


a2 = sigmoid(a1 * Theta1');
a2 = [ones(size(a2, 1), 1) a2]; % add a0(2) to result of the hidden layer 

a3 = sigmoid(a2 * Theta2');


[W, p] = max(a3, [], 2);

% =========================================================================


end
