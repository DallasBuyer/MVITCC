function [label_mat] = label_vec2mat(label_vec)
% transform the labels from the vec-form to matrix-form-form
% 2017-12-8
% PengXu, Jiangnan University
% label_mat: n_examples * n_clusters
%            the rows of label_mat is examples
%            the cols of label_mat is the corresponding cluster, 1 or 0
% label_vec: n_examples * 1
%            each number of the label_vec represents for its cluster
n_clusters = length(unique(label_vec));
n_examples = length(label_vec);

label_mat = zeros(n_examples, n_clusters);
for i=1:n_clusters
    index  = label_vec==i;
    label_mat(index,i) = 1;
end

end

