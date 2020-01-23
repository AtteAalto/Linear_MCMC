function [data,state,parameters]=MCMC_lin_init(data)

%This file sets some parameters of the MCMC sampler and initialises the
%state of the sampler. Some modifications are made to the data if it is not
%exactly in the form required by the sampler.


%% Set the parameters of the sampler.

%Step length of the Crank-Nicolson sampler (should be >.05)
parameters.etraj=.075;

%Other step length parameters:
parameters.em=.1;
parameters.er=.00001;
parameters.eq=.0001;

%link_pr = p / (1-p) where p is the prior probability for the existence of
%a link. This controls the sparsity level of the network. Default is 1/n if
%it is not given.
% parameters.link_pr=1/100;


%Heuristic temperature variable to speed up the topology sampling. Value 1
%is default option if it is not set, and it corresponds to exactly correct
%sampling.
parameters.Theur=1;

%Number of iterations (in the burn-in)
parameters.its=3000;

%The number of steps in the trajectory between measurements. A good default
%is 8, but the sampling gets (slightly) faster if it is reduced.
nstep=8;   

%The cell array Tsam in the data-structure contains the sampling times of
%the measurements. If the user doesn't specify these, this code creates 
%them, assuming constant sampling frequency (1 sample / time unit). 
if isfield(data,'Tsam')
    if size(data.Tsam,2)<1.5
        if size(data.Tsam{1},2)<1.5
            Tsam={};
            for j=1:size(data.ts,2)
                Tsam={Tsam{1:size(Tsam,2)}, data.Tsam{1}(1,1)*(0:size(data.ts{j},2)-1)};
            end
            data.Tsam=Tsam;
        end
    else
        for j=1:size(data.ts,2)
            if size(data.Tsam{j},2)<1.5
                data.Tsam{j}=data.Tsam{j}(1,1)*(0:size(data.ts{j},2)-1);
            end
        end
    end
else
    Tsam={};
    for j=1:size(data.ts,2)
        Tsam={Tsam{1:size(Tsam,2)}, 4*(0:size(data.ts{j},2)-1)};
    end
    data.Tsam=Tsam;
end



%Perform linear interpolation if there are missing measurements. Note that
%the code also takes into account the missing measurements in the sampling
%by increasing the sample variance to obtain a Brownian bridge sample
%between the previous and the next non-missing sample.
if isfield(data,'missing')
    data.ts=missing_data_interpolation(data);
end



%% Initialize the state of the sampler. Do not touch this part!

n=size(data.ts{1},1);
n_in=0;
if isfield(data,'input')
    n_in=size(data.input{1},1);
end

%Initial connectivity guess
state.S=rand(n,n+n_in)>1-2/n;

%Initial trajectory x is obtained by linear interpolation of the time series data.
Ser=zeros(4,size(data.ts,2));
Ser(1,1)=1;
Ser(2,1)=size(data.ts{1},2);
Ser(3:4,1)=[1; nstep*(Ser(2,1)-1)+1];
for jser=2:size(data.ts,2)
    Ser(1:2,jser)=[Ser(2,jser-1)+1;Ser(2,jser-1)+size(data.ts{jser},2)];
    Ser(3:4,jser)=[Ser(4,jser-1)+1; Ser(4,jser-1)+nstep*(Ser(2,jser)-Ser(1,jser))+1];
end
for jser=1:size(Ser,2)
    for jj=1:(Ser(2,jser)-Ser(1,jser))
        xs(:,Ser(3,jser)+(jj-1)*nstep:Ser(3,jser)+jj*nstep-1)=data.ts{jser}(:,jj)*(1-(0:nstep-1)/nstep)+data.ts{jser}(:,jj+1)*(0:nstep-1)/nstep;
    end
    xs(:,Ser(4,jser))=data.ts{jser}(:,end);
end
state.xs=xs;

%Compute reasonable initial values for q and M
nry=0;
nry_q=0;
Ttot=0;
for l=1:size(Ser,2)  
    nry=nry+sum((data.ts{l}(:,2:end)-data.ts{l}(:,1:end-1)).^2./(data.Tsam{l}(2:end)-data.Tsam{l}(1:end-1)),2);
    nry_q=nry_q+sum((data.ts{l}(:,2:end)-data.ts{l}(:,1:end-1)).^2,2);
    Ttot=Ttot+data.Tsam{l}(end)-data.Tsam{l}(1);    
end 
nry=nry/Ttot;
nry_q=nry_q/Ttot;


%Initialise other hyperparameters
state.r=.0006*.1*ones(n,1);       
state.q=nry_q/20;
state.M=nry/3;

%Initialise cost function values
state.P=-1e8*ones(n,1);
state.J=1e8*ones(n,1);


