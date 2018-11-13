function [confusion_matrix] = compute_confusion_matrix(true_label,predict_label)
% compute a confusion matrix for classification results
% reference to the code from Frank Lin (frank@cs.cmu.edu)
% note: here the confusion matrix is for classfication which is
%       totally different from the confusion matrix for clustering
%       where it is always called matching matrix.
% 2017-12-8
% PengXu, Jiangnan University
% true_label: n * 1
% predict_label: n * 1
% confusion_matrix: rows for predicted results and cols for truth.

confusion_matrix=zeros(max(true_label));

for i=1:length(true_label)
   confusion_matrix(predict_label(i),true_label(i))=...
       confusion_matrix(predict_label(i),true_label(i))+1; 
end

end

