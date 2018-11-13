function [purity] = compute_purity(true_label,predict_label)
% compute the purity for the clustering results
% reference to the code from Frank Lin (frank@cs.cmu.edu)
% note: computing purity don't need the corresponding relationship
%       between the true_label and preditc_label, value of purity
%       is the higher, the better.
% 2017-12-8
% PengXu, Jiangnan University
% true_label: n * 1
% predict_label: n * 1

confusion_matrix = compute_confusion_matrix(true_label,predict_label);

purity = sum(max(confusion_matrix,[],2))/length(true_label);
end

