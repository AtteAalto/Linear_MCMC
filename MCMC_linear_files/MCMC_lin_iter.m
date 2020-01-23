function [Plink,chain,xstore,state,stats]=MCMC_lin_iter(data,state,parameters)

stats.start_time=clock;

if match_check(data.Tsam,data.ts)
    error('Sampling times are not consistent with the time series data!')
end

%Set the state of the sampler
q=state.q;
r=state.r;
xs_old=state.xs;
Pold=state.P;
Jold=state.J;
Sold=state.S;
M=state.M;


%Set parameters
ee=parameters.etraj;
eq=parameters.eq;
er=parameters.er;
em=parameters.em;
its=parameters.its;

%Set the heuristic temperature (default: 1)
Theur=1;
if isfield(parameters,'Theur')
    Theur=parameters.Theur;
end

%Reform data
Tsam=data.Tsam;
ts=data.ts;
u=[];
if isfield(data,'input')
    if match_check(data.input,data.ts)
        error('Input size is not consistent with the time series data!')
    end
    input=data.input;
    u=input{1}(:,1:end-1);
end

%TS data is put in one matrix y. 
%Matrix Ser contains the indices corresponding to different experiments.
y=ts{1};
Ser=zeros(4,size(ts,2));
Ser(1,1)=1;
Ser(2,1)=size(y,2);
for j=2:size(ts,2)
    y=[y,ts{j}];
    Ser(1:2,j)=[Ser(2,j-1)+1;Ser(2,j-1)+size(ts{j},2)];
    if isfield(data,'input')
        u=[u,input{j}(:,1:end-1)];
    end
end

%Check dimension variables
n=size(y,1);
n_in=size(u,1);
nstep=(size(xs_old,2)-size(Ser,2))/(size(y,2)-size(Ser,2));

%Complete the Ser-matrix
Ser(3:4,1)=[1; nstep*(Ser(2,1)-1)+1];
for jser=2:size(Ser,2)
    Ser(3:4,jser)=[Ser(4,jser-1)+1; Ser(4,jser-1)+nstep*(Ser(2,jser)-Ser(1,jser))+1];
end


%Set the indicator prior (default: 1/n)
link_pr=1/n;
if isfield(parameters,'link_pr')
    link_pr=parameters.link_pr;
end

%Check for prior information. S_aux indicates which elements in S are
%allowed to change (default: all ones)
S_aux=zeros(n,n+n_in);
if isfield(data,'sure')
    if size(data.sure,2)>n+.5
        S_aux=data.sure;
    else
        S_aux(1:n,1:n)=data.sure;
    end
end
Sold=max(Sold,S_aux);
Sold=min(Sold,1+S_aux);
S_aux=ones(n,n+n_in)-abs(S_aux);

%Upsample the input using zero-order hold.
for j=1:size(u,2)
    u=[u(:,1:(j-1)*nstep),u(:,(j-1)*nstep+1)*ones(1,nstep),u(:,(j-1)*nstep+2:end)];  
end

%Prepare the nomiss indicator matrix, showing the measurements that are NOT
%missing (default: all ones)
nomiss=ones(size(y));
if isfield(data,'missing')
    if norm(size(data.missing)-size(data.ts))>.1
        error('Missing measurements indices are not consistent with the time series data!')
    end   
    for i=1:n
        for jser=1:size(Ser,2)
            miss=find(abs(data.missing{jser}(:,1)-i)<.5);
            nomiss(i,data.missing{jser}(miss,2)+Ser(1,jser)-1)=zeros(1,length(miss));
        end
    end
end

%Compute indices that are used in the derivative computation (derind) and
%those that give the measured values from the finer trajectory, that is,
%y=x(yind), and compute the time steps in the finer disretisation (d).
derind=(1:nstep*(Ser(2,1)-Ser(1,1)));
yind=1+(0:Ser(2,1)-1)*nstep;
d=(Tsam{1}(2:end)-Tsam{1}(1:end-1))/nstep;
for jser=2:size(Ser,2)
    derind=[derind,(1:nstep*(Ser(2,jser)-Ser(1,jser)))+1+derind(end)];
    yind=[yind,yind(end)+1+(0:Ser(2,jser)-Ser(1,jser))*nstep];
    d=[d,(Tsam{jser}(2:end)-Tsam{jser}(1:end-1))/nstep];
end
d=reshape(ones(nstep,1)*d,1,[]);


% Compute different size estimates from the data
nry=0;
totvar=0;
for l=1:size(Ser,2)
    nry=nry+sum(((y(:,Ser(1,l)+1:Ser(2,l))-y(:,Ser(1,l):Ser(2,l)-1)).^2./(Tsam{l}(2:end)-Tsam{l}(1:end-1))),2)/sum(d); 
    totvar=totvar+sum(abs(y(:,Ser(1,l)+1:Ser(2,l))-y(:,Ser(1,l):Ser(2,l)-1)),2)/sum(d);
end 
Scale=zeros(n,1);
vary=zeros(n,1);
Sc_u=zeros(n_in,1);
if isfield(data,'input')
    Sc_u=Sc_u+sum(u.^2.*d);  
end
for j=1:size(Ser,2)
    Scale=Scale+(sum(y(:,Ser(1,j)+1:Ser(2,j)-1).^2.*(Tsam{j}(3:end)-Tsam{j}(1:end-2)),2)/2+(y(:,Ser(1,j)).^2*(Tsam{j}(2)-Tsam{j}(1))+y(:,Ser(2,j)).^2*(Tsam{j}(end)-Tsam{j}(end-1)))/2);
    
    %Estimate on the L^2 norm of the outputs based on y. Used for scaling of q and M.
    vary=vary+sum((y(:,Ser(1,j)+1:Ser(2,j))-y(:,Ser(1,j):Ser(2,j)-1)).^2./(Tsam{j}(2:end)-Tsam{j}(1:end-1)),2);     
end
Scale=[Scale;Sc_u].^-.5;


%Embedding of the measurements to a piecewise linear function (denoted by
%m[Y] in the paper).
mm=max(Ser(4,:)-Ser(3,:))+1;
Pr=zeros(mm,max(Ser(2,:)-Ser(1,:))+1);
Pr(1:nstep,1)=flipud((1:nstep)'/nstep);
Pr(mm-nstep+1:mm,end)=(1:nstep)'/nstep;
for j=2:max(Ser(2,:)-Ser(1,:))
    Pr((j-2)*nstep+2:(j-1)*nstep+1,j)=(1:nstep)'/nstep;
    Pr((j-1)*nstep+1:j*nstep,j)=flipud((1:nstep)'/nstep);
end
Pintc=sin((1:nstep-1)'*(1:nstep-1)/nstep*pi)./(pi*ones(nstep-1,1)*(1:nstep-1))*2^.5;

%Initialise variables
Plink=zeros(size(Sold));    %Sum of indicator samples (\bar{P} in the paper)
chain=0;                    %Nr. of samples
acctraj=0;                  %Nr. of accepted trajectory-samples
xstore=0*xs_old;            %Posterior mean of trajectory
acctop=zeros(n,1);          %Nr. of accepted indicator samples (row-wise)
acchyp=zeros(n,1);          %Nr. of accepted M-samples
accr=zeros(n,1);            %Nr. of accepted r-samples
yold=xs_old(:,yind);        %Measurement from the current trajectory sample

%% Iterations
tic; time_mark=toc;
for k=1:its
    %===== INDICATOR SAMPLING =====
    
    %Compute the Gramian XX and the bracket terms YX
    XX=[xs_old(:,derind);u]*(d.*[xs_old(:,derind);u])';
    YX=(xs_old(:,derind+1)-xs_old(:,derind))*[xs_old(:,derind);u]';
    YX=YX*diag(Scale);
    XX=diag(Scale)*XX*diag(Scale);
    
    for i=1:n
        %Get the old row of S
        S=Sold(i,:);
        
        %Check which elements are allowed to change
        inds=find(S_aux(i,:)>.5);
        
        %Choose from two different types of moves
        topc=(rand>.5)*(sum(S(inds))>.5)*(sum(S(inds))<sum(S_aux(i,:))-.5);
        
        %Perform the move
        if (1-topc)
            %Change one entry in S
            indc=randi(sum(S_aux(i,:)),1);
            S(inds(indc))=1-S(inds(indc)); 
        else
            %Change one zero into one and one one into zero
            ind1=find(S(inds)>.5);
            ind0=find(S(inds)<.5);
            indc01=ind0(randi(length(ind0)));
            indc10=ind1(randi(length(ind1)));
            S(inds(indc01))=1;
            S(inds(indc10))=0;
        end
        
        ind=find(S>.5);
        
        %Part of the Wiener measure that is not in the CN sampler
        nrY=0;
        for l=1:size(Ser,2)
            nrY=nrY+sum(((yold(i,Ser(1,l)+1:Ser(2,l))-yold(i,Ser(1,l):Ser(2,l)-1)).^2./(Tsam{l}(2:end)-Tsam{l}(1:end-1))),2);     
        end 
        
        %Cost function value
        J1=(nrY-YX(i,ind)*((q(i)/M(i)*eye(length(ind))+XX(ind,ind))\YX(i,ind)'))/2/q(i)+.5*log(det(eye(length(ind))+M(i)/q(i)*XX(ind,ind)));
        PS=sum(S)*log(link_pr);
        
        %Acceptance of row i
        if exp((PS-Pold(i)+Jold(i)-J1)/Theur) > rand
            Sold(i,:)=S;
            Pold(i)=PS;
            Jold(i)=J1;
            acctop(i)=acctop(i)+1;
        end
        
        
        %Sampling hyperparameters M_i
        ind=find(Sold(i,:)>.5);
        Mtr=M(i)+em*randn;
        Mtr=min(max(Mtr,1e-5),20*vary(i));
        J1=(nrY-YX(i,ind)*((q(i)/Mtr*eye(length(ind))+XX(ind,ind))\YX(i,ind)'))/2/q(i)+.5*log(det(eye(length(ind))+Mtr/q(i)*XX(ind,ind)));
        if Mtr/vary(i)*(20-Mtr/vary(i))/(M(i)/vary(i)*(20-M(i)/vary(i)))*exp(1/vary(i)*(M(i)-Mtr)+Jold(i)-J1) > rand
            M(i)=Mtr;
            Jold(i)=J1;
            acchyp(i)=acchyp(i)+1;
        end 
        
        
        %Sampling of r(i)
        rtr=r(i)+er*randn;
        rtr=1e-8+abs(rtr-1e-8);
        if (r(i)/rtr)^(1+sum(nomiss(i,:))/2)*exp(.00001./r(i)-.00001./rtr+sum((y(i,find(nomiss(i,:)>.5))-yold(i,find(nomiss(i,:)>.5))).^2)/2*(1/r(i)-1/rtr))>rand
            r(i)=rtr;
            accr(i)=accr(i)+1;
        end
    end
        
    
    %===== TRAJECTORY SAMPLING =====
    qtr=q+eq*randn(size(q));
    qtr=.5e-5+abs(qtr-.5e-5);
    
    %Sample the trajectory
    for l=1:size(Ser,2)
        if isfield(data,'missing')  %Missing measurements
            for i=1:n
                Csam=missing_data_sampler(data.missing{l},Tsam{l},r(i),qtr(i),i);
                coef=(1-ee^2)^.5*nomiss(i,Ser(1,l):Ser(2,l))+(qtr(i)/q(i))^(0*.5)*(1-nomiss(i,Ser(1,l):Ser(2,l)));
                yhat(i,Ser(1,l):Ser(2,l))=y(i,Ser(1,l):Ser(2,l))+coef.*(yold(i,Ser(1,l):Ser(2,l))-y(i,Ser(1,l):Ser(2,l)))+ee*randn(1,Ser(2,l)-Ser(1,l)+1)*Csam';
            end
        else
            yhat(:,Ser(1,l):Ser(2,l))=y(:,Ser(1,l):Ser(2,l))+(1-ee^2)^.5*(yold(:,Ser(1,l):Ser(2,l))-y(:,Ser(1,l):Ser(2,l)))+ee*diag(r.^.5)*randn(n,Ser(2,l)-Ser(1,l)+1);   
        end        
        xs(:,Ser(3,l):Ser(4,l))=sparse(diag((qtr./q).^.5))*(1-ee^2)^.5*xs_old(:,Ser(3,l):Ser(4,l))+(yhat(:,Ser(1,l):Ser(2,l))-sparse(diag((qtr./q).^.5))*(1-ee^2)^.5*yold(:,Ser(1,l):Ser(2,l)))*Pr(1:(Ser(4,l)-Ser(3,l)+1),1:(Ser(2,l)-Ser(1,l)+1))'; 
        xs(:,Ser(3,l)+1:Ser(4,l))=xs(:,Ser(3,l)+1:Ser(4,l))+ee*sparse(diag(qtr.^.5))*((nstep*d((Ser(3,l):Ser(4,l)-1)-l+1)).^.5.*reshape([Pintc*randn(nstep-1,n*(Ser(2,l)-Ser(1,l)));zeros(1,n*(Ser(2,l)-Ser(1,l)))],[],n)'); 
    end
    
    
    %Initialise the cost function value
    J1=zeros(n,1);
    
    %Compute the Gramian XX and the bracket terms YX
    XX=[xs(:,derind);u]*(d.*[xs(:,derind);u])';
    YX=(xs(:,derind+1)-xs(:,derind))*[xs(:,derind);u]';
    YX=YX*diag(Scale);
    XX=diag(Scale)*XX*diag(Scale);
    
    for i=1:n
        ind=find(Sold(i,:)>.5);
        
        %Part of the Wiener measure that is not in the CN sampler
        nrY=0;
        for l=1:size(Ser,2)
            nrY=nrY+sum(((yhat(i,Ser(1,l)+1:Ser(2,l))-yhat(i,Ser(1,l):Ser(2,l)-1)).^2./(Tsam{l}(2:end)-Tsam{l}(1:end-1))),2);     
        end 
        
        %Cost function value
        J1(i)=(nrY-YX(i,ind)*((qtr(i)/M(i)*eye(length(ind))+XX(ind,ind))\YX(i,ind)'))/2/qtr(i)+.5*log(det(eye(length(ind))+M(i)/qtr(i)*XX(ind,ind)));    
    end
       
    %Accept or reject the new sample
    P_aux_q=exp(sum(.001./q-.001./qtr+(log(q)-log(qtr)).*(1.001+.5*(size(y,2)-size(Ser,2)))));
    if P_aux_q*exp(sum(Jold-J1)) > rand
        Jold=J1;
        q=qtr;
        acctraj=acctraj+1;
        xs_old=xs; 
        yold=yhat;
    end
    
    %Notify the user if the simulation is going to take a long time (>15 min).
    if k==100
        time_now=toc;
        left=(time_now-time_mark)*(its-k)/100;
        if left>900
            h_left=floor(left/3600);
            left=left-3600*h_left;
            min_left=floor(left/60);
            disp(['NOTE! Estimated time remaining for completion: ' num2str(h_left) ' h ' num2str(min_left) ' min'])
        end       
    end
    
    %Take every 10th sample to the actual distribution (thinning)
    if mod(k,10)<.5
        chain=chain+1;
        Plink=Plink+Sold;
        xstore=xstore+xs_old;
        
        %Report progression
        if mod(k,10000)<.5
            time_now=toc;
            left=(time_now-time_mark)*(its-k)/10000;
            h_left=floor(left/3600);
            left=left-3600*h_left;
            min_left=floor(left/60);
            left=floor(left-60*min_left);
            disp(['Iteration: ' num2str(k) ', Estimated time remaining: ' num2str(h_left) ' h ' num2str(min_left) ' min ' num2str(left) ' sec'])
            time_mark=time_now;
        end
    end
end

xstore=xstore/(its/10);
stats.acctraj=acctraj;
stats.acctop=acctop;
stats.acchyp=acchyp;
stats.accr=accr;
stats.fin_time=clock;

state.q=q;
state.r=r;
state.xs=xs_old;
state.P=Pold;
state.J=Jold;
state.S=Sold;
state.M=M;


