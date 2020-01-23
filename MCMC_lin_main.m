%% === Linear MCMC ===
%This file runs linear MCMC method as described in the acompanying
%readme-file. This file should be run one section at a time. Below is the 
%implementation producing the results of the numerical example described in 
%the article.

addpath('./MCMC_linear_files')

[data,state,parameters]=GPDM_init(data);

% MCMC Burn-in
[~,chain,~,state,stats]=GPDM_ko(data,state,parameters);
disp_stats(' BURN-IN COMPLETE',stats,chain,parameters.its)

% Actual sampling
parameters.its=20000;
[Plink,chain,xstore,state,stats]=GPDM_ko(data,state,parameters);
disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)

%The end result is Plink/chain.


%% Collect more samples

chain_old=chain;
Plink_old=Plink;

parameters.its=50000;
xstore_old=xstore;
[Plink,chain,xstore,state,stats]=GPDM(data,state,parameters);

xstore=chain_old/(chain+chain_old)*xstore_old+chain/(chain+chain_old)*xstore;
Plink=Plink_old+Plink;
chain=chain_old+chain;
disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)



%% Numerical example of the article 

%This takes a long time to run if all replicates are done at once. However,
%the replicates can be run in parallel simply by replacing the for-loop
%with parfor.

addpath('./MCMC_linear_files')
load('paper_data.mat')

%Run the algorithm for the ten replicates
for rep=1:10
    clear('data')
    
    % Case 1
    data.ts={all_data(rep).y1,all_data(rep).y2};
    data.Tsam={.5};
    
    % Case 2
    %data.ts={all_data(rep).y1(:,1:2:end),all_data(rep).y2(:,1:2:end)};
    %data.Tsam={1};
    
    % Case 3
    %data.ts={all_data(rep).y1};
    %data.Tsam={.5};
    
    [data,state,parameters]=MCMC_lin_init(data);
    parameters.Theur=1.2;
    
    % MCMC Burn-in
    [~,chain,~,state,stats]=MCMC_lin_iter(data,state,parameters);
    disp_stats(' BURN-IN COMPLETE',stats,chain,parameters.its)

    % Actual sampling
    parameters.its=150000;
    [Plink,chain,xstore,state,stats]=MCMC_lin_iter(data,state,parameters);
    disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)
    
    res(rep).Plink=Plink;
    res(rep).chain=chain;
    res(rep).xstore=xstore;
    res(rep).state=state;
    disp(['Replicate ' num2str(rep) ' complete'])
end

%% More samples

for rep=1:10
    clear('data')
    
    % Case 1
    data.ts={all_data(rep).y1,all_data(rep).y2};
    data.Tsam={.5};
    
    % Case 2
    % data.ts={all_data(rep).y1(:,1:2:end),all_data(rep).y2(:,1:2:end)};
    % data.Tsam={1};
    
    % Case 3
    % data.ts={all_data(rep).y1};
    % data.Tsam={.5};
   
    [data,state,parameters]=MCMC_lin_init(data);
    
    chain_old=res(rep).chain;
    Plink_old=res(rep).Plink;
    state=res(rep).state;

    parameters.its=30000;
    [Plink,chain,xstore,state,stats]=MCMC_lin_iter(data,state,parameters);
    Plink=Plink_old+Plink;
    chain=chain_old+chain;
    disp_stats(' SAMPLING COMPLETE',stats,chain,parameters.its)


    res(rep).Plink=Plink;
    res(rep).chain=chain;
    res(rep).xstore=xstore;
    res(rep).state=state;
    disp(['Replicate ' num2str(rep) ' complete'])
end


%% Compute the AUROC/AUPR values

aucs=zeros(10,2);
for rep=1:10
    links=res(rep).Plink/res(rep).chain;
    ROC_nodiag
    aucs(rep,:)=[AUROC,AUPR];
end


