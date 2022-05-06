function [Pm,search_area]=Bayesian_DSP2018_paper(X,Snap,resolution,position,etc)
search_area=[-90:resolution:90];
Rx=X*X'/Snap;
Y=Rx(:);
[M2,T]=size(Y);
M=sqrt(M2);
K_hat=length(search_area);
reslu=search_area(2)-search_area(1);
In=eye(M);
In=In(:);
pos_all=round(log(kron(exp(-position),exp(position))));
W=kron(Rx.',Rx)/Snap;
W_sq=sqrtm(inv(W));
Y=W_sq*Y;

%% Initialization
d=0.01;
maxiter=300;
tol=1e-5;
beta0=1;
delta=ones(K_hat+1,1);
a_search=search_area*pi/180;
A=exp(-1i*pi*pos_all'*sin(a_search));
B=-1i*pi*pos_all'*cos(a_search).*A;
A_w=W_sq*A;
B_w=W_sq*B;
I_w= W_sq* In;
Phi=[A_w, I_w];
V_temp=  1/beta0*eye(M2) +  Phi *diag(delta) * Phi';
Sigma = diag(delta) -diag(delta) * Phi' * (V_temp\Phi) *  diag(delta);
mu = beta0*Sigma * Phi' * Y;

converged = false;
iter = 0;
while (~converged) || iter<=100
    
    iter = iter + 1;
    delta_last = delta;
    %% Calculate mu and Sigma
    V_temp=  1/beta0*eye(M2) +  Phi *diag(delta) * Phi';
    Sigma = diag(delta) -diag(delta) * Phi' * (V_temp\Phi) *  diag(delta);
    mu = beta0*(Sigma * (Phi' * Y));
    
    %% Update delta
    temp=sum( mu.*conj(mu), 2) + T*real(diag(Sigma));
    delta= ( -T+ sqrt(  T^2 + 4*d* real(temp) ) ) / (  2*d   );
    
    %% Stopping criteria
    erro=norm(delta - delta_last)/norm(delta_last);
    if erro < tol || iter >= maxiter
        converged = true;
    end
    
    %% Grid refinement
    [~, idx] = sort(delta(1:end-1), 'descend');
    idx = idx(1:etc);
    BHB = B_w' * B_w;
    P = real(conj(BHB(idx,idx)) .* (mu(idx,:) * mu(idx,:)' + Sigma(idx,idx)));
    v =  real(diag(conj(mu(idx))) * B_w(:,idx)' * (Y - A_w * mu(1:end-1)-mu(end)*I_w))...
        -  real(diag(B_w(:,idx)' * A_w * Sigma(1:end-1,idx))  +    diag(Sigma(idx,K_hat+1))*B_w(:,idx).'*conj(I_w));
    eigP=svd(P);
    if eigP(end)/eigP(1)>1e-5
        temp1 =  P \ v;
    else
        temp1=v./diag(P);
    end
    temp2=temp1'*180/pi;
    if iter<100
        ind_small=find(abs(temp2)<reslu/100);
        temp2(ind_small)=sign(temp2(ind_small))*reslu/100;
    end
    ind_large=find(abs(temp2)>reslu);
    temp2(ind_large)=sign(temp2(ind_large))*reslu/100;
    angle_cand=search_area(idx) + temp2;
    search_area(idx)=angle_cand;
%     for iii=1:etc
%         if angle_cand(iii)>= search_mid_left(idx(iii))  && (angle_cand(iii) <= search_mid_right(idx(iii)))
%             search_area(idx(iii))=angle_cand(iii);
%         end
%     end
    A_ect=exp(-1i*pi*pos_all'*sin(search_area(idx)*pi/180));
    B_ect=-1i*pi*pos_all'*cos(search_area(idx)*pi/180).*A_ect;
    A_w(:,idx) =W_sq*A_ect;
    B_w(:,idx) =W_sq*B_ect;
    Phi(:,idx)= A_w(:,idx);
 
end
Pm=delta(1:end-1);
% [search_area,sort_s]=sort(search_area);
% Pm=Pm(sort_s);
% insert=(search_area(1:end-1)+search_area(2:end))/2;
% search_area_2=zeros(length(search_area)*2-1,1);
% search_area_2(1:2:end)=search_area;
% search_area_2(2:2:end)=insert;
% Pm_2=zeros(length(Pm)*2-1,1);
% Pm_2(1:2:end)=Pm;
% search_area=search_area_2;
% Pm=Pm_2;