function [label_vec] = label_mat2vec(label_mat)
% transform the labels from the matrix-form to vec-form
% 2017-12-8
% PengXu, Jiangnan University
% label_mat: n_examples * n_clusters
%            the rows of label_mat is examples
%            the cols of label_mat is the corresponding cluster, 1 or 0
% label_vec: n_examples * 1
%            each number of the label_vec represents for its cluster
[n_examples,n_clusters] = size(label_mat);
label_vec = zeros(n_examples,1);

for i = 1:n_clusters
    index  = label_mat(:,i) ==1;
    label_vec(index) = i;
end

end

