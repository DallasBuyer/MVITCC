function [CX,CY,indicators] = ib_co_clustering(X,multi_X,Y,kx,kf,maxiter)
% X: the co-occurence matrix such as the text-term frequency
% for each document
% Y: labels for X with the shape of n*1
% kx: number of clusters for examples of X
% kf: number of clusters for features of X
% maxiter: number of iteration during the algorithm

p_x_y = X/sum(sum(X));
[CX,CY] = init_CX_CY_kmeans(X,Y,kx,kf);
ph_x_y = calc_Ph_X_Y(p_x_y,CX,CY); 
f_old = calc_obj(p_x_y,CX,CY);

iter=0;
while iter<maxiter
    CY = update_CY(p_x_y,ph_x_y,kf,CX,CY);
    CX = update_CX(p_x_y,ph_x_y,kx,CX,CY);
    ph_x_y = calc_Ph_X_Y(p_x_y,CX,CY); 
    f_new = calc_obj(p_x_y,CX,CY);
    if f_new>f_old
        break;
    end
    if abs(f_new - f_old) < 0.000001
        break;
    end
    f_old = f_new;
    iter = iter + 1;
end
indicators = evaluate(CX,Y);
end

function [indicators]=evaluate(CX,Y)
indicators = evaluate_cluster_indicators(Y, label_mat2vec(CX));
end

function [CX] = update_CX(p_x_y,ph_x_y,kx,CX,CY)
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

CX = zeros(n,kx);
for i=1:n
    temp = zeros(kx,1);
    for j=1:kx
        temp(j) = KL_divergence(p_y_l_x(i,:),ph_y_l_xh(:,j)');
    end
    [~, min_index] = min(temp);
    CX(i,min_index) = 1;
end

end

function [CY] = update_CY(p_x_y,ph_x_y,ky,CX,CY)

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

end

function [obj] = calc_obj(p_x_y,CX,CY)

ph_x_y = calc_Ph_X_Y(p_x_y,CX,CY);
p1 = p_x_y(:);
ph1 = ph_x_y(:);
index = find(p1~=0);
p1 = p1(index);
ph1 = ph1(index);
obj = sum(p1.*log(p1./ph1));
end

function [CX,CY]=init_CX_CY_kmeans(X,Y,kx,kf)
[n_examples,n_features] = size(X);

% clustering for X
index_x = kmeans(X,kx);
CX = zeros(n_examples,kx);
for i=1:n_examples
    CX(i,index_x(i)) = 1;
end

% clustering for Y
if kf>n_features
    error('kf is greater than the dimensions of features');
end
index_y = kmeans(X',kf);
CY = zeros(n_features,kf);
for i=1:n_features
    CY(i,index_y(i)) = 1;
end

end

function [ph_x_y]=calc_Ph_X_Y(p_x_y,CX,CY)

% calculate the joint distribution p(xh,yh) for current view
p_xh_yh = calc_P_Xh_Yh(p_x_y,CX,CY);
p_x = sum(p_x_y');
p_y = sum(p_x_y);
p_xh = sum(p_xh_yh');
p_yh = sum(p_xh_yh);

kx = size(CX,2);
ky = size(CY,2);

% calculate the conditional distribution p(x|xh)
p_x_l_xh = zeros(size(p_x));
for j=1:kx
    index = find(CX(:,j)==1);
    p_x_l_xh(index) = p_x(index)/p_xh(j);
end
% calculate the conditional distribution p(y|yh)
p_y_l_yh = zeros(size(p_y));
for j=1:ky
    index = find(CY(:,j)==1);
    p_y_l_yh(index) = p_y(index)/p_yh(j);
end

% calculate the joint distribution ph(x,y)
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