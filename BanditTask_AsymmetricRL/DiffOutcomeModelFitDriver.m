%RUNS FMINCON
%this script is run once

load MTurkHoneyData; % the data to fit
Nsubjects = size(dataTest.stimChosen,1); 
global V_end;
V_end = zeros(Nsubjects,6);

clear Fit 
Fit.Nparms = 3;
Fit.LB = [0 0 1e-6];
Fit.UB = [1 1 30];

for s = 1:Nsubjects;
    fprintf('Fitting subject %d out of %d...\n',s,Nsubjects)
    % preprocessing the data a bit
    T = dataTraining.trialtype(s,:);
    T = T+1;
    C = dataTraining.stimChosen(s,:) - 64;
    Acc = dataTraining.accuracy(s,:);
    R = dataTraining.outcome(s,:);
    
    for iter = 1:10;   % run 10 times from random initial conditions, to get best fit
        fprintf('Iteration %d...\n',iter)
        
        % determining initial condition
        Fit.init(s,iter,:) = rand(1,length(Fit.LB)).*(Fit.UB-Fit.LB)+Fit.LB; % random initialization
        
        % running fmincon to fit the free parameters of the model
        [res,lik] = ... 
            fmincon(@(x) FitModel_DiffOutcome(C,R,x(1),x(2),x(3),s),...
            squeeze(Fit.init(s,iter,:)),[],[],[],[],Fit.LB,Fit.UB,[],...
            optimset('TolX', 0.00001, 'TolFun', 0.00001, 'MaxFunEvals', 9e+9, 'Algorithm', 'interior-point'));
        % GradObj = 'on' to use gradients, 'off' to not use them *** ask us about this if you are interested *** 
        % DerivativeCheck = 'on' to have fminsearch compute derivatives numerically and check the ones I supply
        % LargeScale = 'on' to use large scale methods, 'off' to use medium
        Fit.Result.EtaPos(s,iter) = res(1); 
        Fit.Result.EtaNeg(s,iter) = res(2);
        Fit.Result.Beta(s,iter) = res(3); 
        Fit.Result.Lik(s,iter) = lik;
        Fit.Result.Lik  % to view progress so far
    end
    
end

% find the best fit results for each subject
%Displays subject id, eta, beta, loglikelihood
[a,b] = min(Fit.Result.Lik,[],2);
for s = 1:Nsubjects
    Fit.Result.BestFit(s,:) = [s,...
    Fit.Result.EtaPos(s,b(s)),... 
    Fit.Result.EtaNeg(s,b(s)),...
    Fit.Result.Beta(s,b(s)),...
    Fit.Result.Lik(s,b(s))];
end
Fit.Result.BestFit
Fit = Fit.Result.BestFit;
save('DiffOutcomeModelFits','Fit','V_end')