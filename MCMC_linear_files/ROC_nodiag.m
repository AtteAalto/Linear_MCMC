%Matrix A used in the simulations
n=100;
A=-1.0*eye(n)+diag(ones(n-1,1),1);
A(n,1)=1;
A(20,61)=.8;
A(60,21)=.8;
A(61,61)=-1.8;
A(21,21)=-1.8;
A(28,30)=1;
A(30,30)=-2;
A(80,40)=.3;
A(40,40)=-1.3;

%Ground truth network
net=abs(A)>.01;

%Plot the AUROC/AUPR curves?
pl=0;
   
inds=1:100;
prob=links(inds,inds)/max(max(links(inds,inds)));



%
net=net.*(ones(n,n)-eye(n));
prob=prob.*(ones(n,n)-eye(n));

% ROC curve
prs=.9999:-.0001:.0001;

pos=0*prs;
false=0*prs;

for j=1:length(prs)
    pos(j)=sum(sum((prob>prs(j)).*net(inds,inds)));
    false(j)=sum(sum(prob>prs(j)))-sum(sum((prob>prs(j)).*net(inds,inds))); 
end
    
ntr=sum(sum(net(inds,inds)));
nfa=length(inds)*(length(inds)-1)-ntr;

TPR=[0,pos/ntr,1];
FPR=[0,false/nfa,1];

%AUROC
AUROC=0;
for j=2:length(FPR)
    AUROC=AUROC+(FPR(j)-FPR(j-1))*(TPR(j-1)+TPR(j))/2;
end
    
if pl
    figure
    hold on
    plot(FPR,TPR,'k--','LineWidth',2)
    grid
    xlabel('False positive rate','FontSize',18)
    ylabel('True positive rate','FontSize',18)
    set(gca,'FontSize',16)
end


% Precision-recall

ind=min(min(find(pos>.5)),min(find(false>.5)));

prec=[pos(ind)./(pos(ind)+false(ind)),pos(ind:end)./(pos(ind:end)+false(ind:end))];
rec=[0,pos(ind:end)./ntr];

if pl
    figure
    plot(rec,prec,'k','LineWidth',2)
    grid
    set(gca,'FontSize',16)
    xlabel('Precision','FontSize',18)
    ylabel('Recall','FontSize',18)
end

%AUPR
AUPR=0;
for j=2:length(rec)
    AUPR=AUPR+(rec(j)-rec(j-1))*(prec(j-1)+prec(j))/2;
end


