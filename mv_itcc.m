function indicators = mv_itcc(multi_X,Y,kx,kf_vec,maxiter,w_vec,lamda,best_init_X)
% Peng Xu, Jiangnan University. pengxujnu@163.com
% multi-view information-theoretic co-clustering function

% multi_X: 1*number_views,cell array of the data
% Y: number_samples*1, labels
% kx: number of clusters
% kf_vec: number of clusters for each views along the feature dimension
% maxiter: maximum iterations
% w_vec: initial weights for each view
% lamda: regularization parameter for the maximum entropy term
% best_init_X: number of clusters for concatenated data along the sample dimension

% indicators: [purity, NMI, RI]

[n_views,~,~] = calc_relevant_info(multi_X);

p_x_y_cell = init_P_X_Y(multi_X,n_views);
[CX,CY_cell]=init_CX_CYs_Coclustering(multi_X,Y,kx,kf_vec,best_init_X);
ph_x_y_cell = calc_Ph_X_Y(p_x_y_cell,CX,CY_cell);
f_old = calc_obj(p_x_y_cell,CX,CY_cell,w_vec,lamda);

iter=0;
while iter<maxiter
    CY_cell = update_CY_cell(p_x_y_cell,ph_x_y_cell,kf_vec,CX,CY_cell,w_vec,n_views);
    ph_x_y_cell = calc_Ph_X_Y(p_x_y_cell,CX,CY_cell);
    CX = update_CX(p_x_y_cell,ph_x_y_cell,kx,CX,CY_cell,w_vec,n_views);
    
    w_vec = update_w_vec(p_x_y_cell,CX,CY_cell,w_vec,lamda);
    
    ph_x_y_cell = calc_Ph_X_Y(p_x_y_cell,CX,CY_cell);
    f_new = calc_obj(p_x_y_cell,CX,CY_cell,w_vec,lamda);
    
    if abs(f_new-f_old) < 0.000001
        disp('iteration has been convergence')
        break;
    end
    if isnan(f_new)
        error('obj is NaN');
    end
    f_old = f_new;
    iter = iter + 1;
end
disp('the performance after iterations');
indicators = evaluate(CX,Y);
end

function [indicators]=evaluate(CX,Y)
indicators = evaluate_cluster_indicators(Y, label_mat2vec(CX));
end

function [CX]=update_CX(p_x_y_cell,ph_x_y_cell,kx,CX,CY_cell,w_vec,n_views)

p_y_l_x_cell = cell(1,n_views);
ph_y_l_xh_cell = cell(1,n_views);
for v=1:n_views
    p_x_y = p_x_y_cell{1,v};
    ph_x_y = ph_x_y_cell{1,v};
    CY = CY_cell{1,v};
    
    [n,d] = size(p_x_y);
    p_y = sum(p_x_y);
    p_xh_yh = calc_P_Xh_Yh(p_x_y,CX,CY);
    ph_xh_yh = calc_P_Xh_Yh(ph_x_y,CX,CY);
    ph_xh = sum(ph_xh_yh');
    p_yh = sum(p_xh_yh);
    
    cy = zeros(d,1);
    cx = zeros(n,1);
    for i=1:d
        cy(i) = find(CY(i,:)==1);
    end
    for i=1:n
        cx(i) = find(CX(i,:)==1);
    end
    
    ph_y_l_xh = zeros(d,kx);
    for i=1:d
        for j=1:kx
            ph_y_l_xh(i,j)=(p_y(i)/p_yh(cy(i)))*(ph_xh_yh(j,cy(i))/ph_xh(j));
        end
    end
    p_x = sum(p_x_y')';
    p_y_l_x = p_x_y./repmat(p_x,1,d);
    
    p_y_l_x_cell{1,v} = p_y_l_x;
    ph_y_l_xh_cell{1,v} = ph_y_l_xh;
end

CX = zeros(n,kx);

for i=1:n
    temp = zeros(kx,1);
    for j=1:kx
        for v=1:n_views
            p_y_l_x = p_y_l_x_cell{1,v};
            ph_y_l_xh = ph_y_l_xh_cell{1,v};
            w = w_vec(v);
            temp(j) = temp(j)+w * KL_divergence(p_y_l_x(i,:),ph_y_l_xh(:,j)');
        end
    end
    [~, min_index] = min(temp);
    CX(i,min_index) = 1;
end
end

function [CY_cell_new]=update_CY_cell(p_x_y_cell,ph_x_y_cell,kf_vec,CX,CY_cell,w_vec,n_views)

CY_cell_new = cell(1,n_views);
for v=1:n_views
    p_x_y = p_x_y_cell{1,v};
    ph_x_y = ph_x_y_cell{1,v};
    ky = kf_vec(1,v);
    CY = CY_cell{1,v};
    
    [nx,ny] = size(p_x_y);
    p_y = sum(p_x_y);
    p_x = sum(p_x_y');
    
    p_xh_yh = calc_P_Xh_Yh(p_x_y,CX,CY);
    p_xh = sum(p_xh_yh');
    
    ph_xh_yh = calc_P_Xh_Yh(ph_x_y,CX,CY);
    ph_yh = sum(ph_xh_yh);
    
    cx = zeros(nx,1);
    cy = zeros(ny,1);
    for i=1:nx
        cx(i) = find(CX(i,:)==1);
    end
    for i=1:ny
        cy(i) = find(CY(i,:)==1);
    end
    
    p_x_l_y = p_x_y./repmat(p_y,nx,1);
    index1 = isnan(p_x_l_y);
    p_x_l_y(index1) = 0;
    
    ph_x_l_yh = zeros(nx,ky);
    for i=1:nx
        for j=1:ky
            ph_x_l_yh(i,j) = (p_x(i)/p_xh(cx(i)))*(ph_xh_yh(cx(i),j)/ph_yh(j));
        end
    end
    
    CY = zeros(ny,ky);
    for i=1:ny
        temp = zeros(ky,1);
        for j=1:ky
            temp(j) = KL_divergence(p_x_l_y(:,i),ph_x_l_yh(:,j));
        end
        [~, min_index] = min(temp);
        CY(i,min_index) = 1;
    end
    
    CY_cell_new{1,v} = CY;
end

end

function [obj_sum]=calc_obj(p_x_y_cell,CX,CY_cell,w_vec,lamda)
n_views = length(CY_cell);
ph_x_y_cell = calc_Ph_X_Y(p_x_y_cell,CX,CY_cell);
obj = zeros(1,n_views);
for i=1:n_views
    p_x_y = p_x_y_cell{1,i};
    ph_x_y = ph_x_y_cell{1,i};
    p1 = p_x_y(:);
    ph1 = ph_x_y(:);
    
    index = find(p1~=0);
    p1 = p1(index);
    ph1 = ph1(index);
    obj(i) = sum(p1.*log(p1./ph1));
end
obj_sum = sum(obj.*w_vec);
entropy = 0;
for i=1:n_views
    w = w_vec(i);
    entropy = entropy + w*log(w+1e-10);
end
obj_sum = obj_sum + lamda*entropy;
end

function [w_vec]=update_w_vec(p_x_y_cell,CX,CY_cell,w_vec,lamda)
n_views = length(CY_cell);
ph_x_y_cell = calc_Ph_X_Y(p_x_y_cell,CX,CY_cell);
temp_denominator=0;
for i=1:n_views
    p_x_y = p_x_y_cell{1,i};
    ph_x_y = ph_x_y_cell{1,i};
    p1 = p_x_y(:);
    ph1 = ph_x_y(:);
    
    index = find(p1~=0);
    p1 = p1(index);
    ph1 = ph1(index);
    obj(i) = sum(p1.*log(p1./ph1));
    
    temp_denominator = temp_denominator+exp(-obj(i)/lamda);
end

for i=1:n_views
    if temp_denominator==0
        w_vec(i)=0.5;
    else
        w_vec(i) = exp(-obj(i)/lamda)/temp_denominator;
    end
end
end

function [n_views,n_examples,d_vec] = calc_relevant_info(multi_X)

n_views = length(multi_X);
n_examples = size(multi_X{1,1},1);
d_vec = zeros(1,n_views);
for i=1:n_views
    d_vec(i) = size(multi_X{1,i},2);
end

end

function [p_x_y_cell] = init_P_X_Y(multi_X,n_views)
p_x_y_cell = cell(1,n_views);
for i=1:n_views
    temp_pxy = multi_X{1,i};
    p_x_y_cell{1,i} = temp_pxy/sum(sum(temp_pxy));
end

end

function [CX,CY_cell]=init_CX_CYs_Coclustering(multi_X,Y,kx,kf_vec,best_init_X)

[n_views,~,~] = calc_relevant_info(multi_X);
CY_cell = cell(1,n_views);
x = [];
for i=1:n_views
    temp_data = multi_X{1,i};
    fprintf('view for feature clustering: %d\n',i);
    [~,temp_CY,~] = ib_co_clustering(temp_data,multi_X,Y,kx,kf_vec(i),20);
    CY_cell{1,i} = temp_CY;
    x = [x,temp_data];
end

best_kf = best_init_X;

disp('initial performance on multi-view dataset');
[CX,~,~] = ib_co_clustering(x,multi_X,Y,kx,best_kf,10);


end

function [ph_x_y_cell]=calc_Ph_X_Y(p_x_y_cell,CX,CY_cell)
n_views = length(CY_cell);
ph_x_y_cell = cell(1,n_views);

for i=1:n_views
    p_x_y = p_x_y_cell{1,i};
    CY = CY_cell{1,i};
    
    p_xh_yh = calc_P_Xh_Yh(p_x_y,CX,CY);
    p_x = sum(p_x_y');
    p_y = sum(p_x_y);
    p_xh = sum(p_xh_yh');
    p_yh = sum(p_xh_yh);
    
    kx = size(CX,2);
    ky = size(CY,2);
    
    p_x_l_xh = zeros(size(p_x));
    for j=1:kx
        index = find(CX(:,j)==1);
        p_x_l_xh(index) = p_x(index)/p_xh(j);
    end
    
    p_y_l_yh = zeros(size(p_y));
    for j=1:ky
        index = find(CY(:,j)==1);
        p_y_l_yh(index) = p_y(index)/p_yh(j);
    end
    
    ph_x_y = zeros(size(p_x_y));
    for j=1:kx
        for k=1:ky
            index_x = find(CX(:,j)==1);
            index_y = find(CY(:,k)==1);
            ph_x_y(index_x,index_y) = p_xh_yh(j,k);
        end
    end
    for j=1:size(ph_x_y,1)
        ph_x_y(j,:) = ph_x_y(j,:).*p_y_l_yh;
    end
    for j=1:size(ph_x_y,2)
        ph_x_y(:,j) = ph_x_y(:,j).*p_x_l_xh';
    end
    
    ph_x_y_cell{1,i} = ph_x_y;
end

end

function [p_xh_yh]=calc_P_Xh_Yh(p_x_y,CX,CY)

[~,kx] = size(CX);
[~,ky] = size(CY);
p_xh_yh = zeros(kx,ky);
for i=1:kx
    for j=1:ky
        index_x = find(CX(:,i)==1);
        index_y = find(CY(:,j)==1);
        temp = p_x_y(index_x,index_y);
        p_xh_yh(i,j) = sum(sum(temp));
    end
end
end

function [dis] = KL_divergence(p,q)
index = find(p ~= 0);
p = p(index);
q = q(index);
temp = p./q;
temp = log(temp);
temp = p.*temp;
dis = sum(temp);
end