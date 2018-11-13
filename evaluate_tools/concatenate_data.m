function [X] = concatenate_data(multi_X)
% concatenate cell vector to matrix
X = [];
for i=1:length(multi_X)
    X = [X,multi_X{i}];
end
end

