% The main file for MV-ITCC(Multi-view Information-theoretic Co-clustering)
% The paper has been accepted by AAAI19.
% Peng Xu, Zhaohong Deng, Kup-Sze Choi, Longbing Cao, Shitong Wang. 2019.
% Multi-view Information-theoretic Co-clustering for Co-occurrence Data.

% 2018-11-13
% Peng Xu, Jiangnan University. pengxujnu@163.com

clc;
clear;

%% source3 dataset
data=importdata('data/sources3_3.mat');
name = 'sources3_3';
kf_vec = [4,8,4];
best_init_X = 8;

%% Caltech dataset
% data=importdata('data/Caltech_2.mat');
% name = 'Caltech_2';
% kf_vec = [80,50];
% best_init_X = 100;

multi_X = data.data;
for j=1:length(multi_X)
    multi_X{1,j} = normalize_data(multi_X{1,j});
end
Y = data.Y;

%% parameter settings
maxiter = 20;
run_times = 30;
kx = max(Y);
n_views = length(multi_X);
w_vec = (zeros(1,n_views)+1)*1/n_views;
lamda = 2.^(-6:1:6);

parameters = lamda;
n_parameters = length(parameters);
records = zeros(n_parameters, 3*2);

%% loop
for p=1:n_parameters
    for j=1:run_times
        indicators = mv_itcc(multi_X,Y,kx,...
            kf_vec,maxiter,w_vec,lamda(p),best_init_X);
        purity(j) = indicators(1);
        nmi(j) = indicators(2);
        randindex(j) = indicators(3);
        
        fprintf('****************parameters**************: %d-----%d\n',p,n_parameters);
        fprintf('*************************run_times**************************: %d\n',j);
    end
   
    records(p,1) = mean(purity);
    records(p,2) = std(purity);
    records(p,3) = mean(nmi);
    records(p,4) = std(nmi);
    records(p,5) = mean(randindex);
    records(p,6) = std(randindex);
    
    fprintf('****************parameters**************: %d-----%d\n',p,n_parameters);
    fprintf('max_purity: %.4f  max_nmi: %.4f  max_ri %.4f  \n',...
        max(records(:,1)),max(records(:,3)),max(records(:,5)));
end

%% save results

[max_value,max_index] = max(records);

purity_max = max_value(1);
purity_max_index = max_index(1);
purity_max_std = records(purity_max_index,2);

nmi_max = max_value(3);
nmi_max_index = max_index(3);
nmi_max_std = records(nmi_max_index,4);

randindex_max = max_value(5);
randindex_max_index = max_index(5);
randindex_max_std = records(randindex_max_index,6);

fprintf('purity_max: %f\n',purity_max);
fprintf('nmi_max:    %f\n',nmi_max);
fprintf('randindex_max:   %f\n',randindex_max);

results.purity_max = purity_max;
results.purity_max_std = purity_max_std;
results.nmi_max = nmi_max;
results.nmi_max_std = nmi_max_std;
results.randindex_max = randindex_max;
results.randindex_max_std = randindex_max_std;

results.name = name;
results.run_times = run_times;
results.maxiter = maxiter;
results.parameters = parameters;
results.records = records;
results.kf_vec = kf_vec;
results.best_init_X = best_init_X;

save(strcat('MVITCC_results_',name),'results');


