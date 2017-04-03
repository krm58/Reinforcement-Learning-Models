%%
%Code that fits data from a Credit Assignment Reinforcement Learning task,
%with the hypothesis that learners are learning stimulus values according
%to an Eligibility Trace model. See Sutton & Barto, 1998.

%Note: this code was adapted from an SRNDNA 2015 Computational Modeling
%workshop. Immense gratitude to SRNDNA for amazing talks and workshops!

%Assumes data is loaded into workspace. Calls ETFit.m

files = dir('*.mat');
Nsubjects = length(files);
clear Fit
Fit.Nparms = 3; %alpha, beta, lambda
Fit.LB = [0 1e-6 0];
Fit.UB = [1 30 1];

for s = 1:Nsubjects;
    load(files(s).name)
    fprintf('Fitting subject %d out of %d...\n',s,Nsubjects)
    for iter = 1:10;   % run from random initial conditions, to get best fit
        fprintf('Iteration %d...\n',iter)
        
        % determining initial condition
        Fit.init(s,iter,:) = rand(1,length(Fit.LB)).*(Fit.UB-Fit.LB)+Fit.LB; % random initialization
        
        % running fmincon to fit the free parameters of the model
        [res,lik] = ...
            fmincon(@(x) ETFit(type,response,flowers,x(1),x(2),x(3)),...
            squeeze(Fit.init(s,iter,:)),[],[],[],[],Fit.LB,Fit.UB,[],...
            optimset('maxfunevals',5000,'maxiter',2000,'GradObj','off','DerivativeCheck','off','LargeScale','on','Algorithm','active-set'));
        % GradObj = 'on' to use gradients, 'off' to not use them *** ask us about this if you are interested ***
        % DerivativeCheck = 'on' to have fminsearch compute derivatives numerically and check the ones I supply
        % LargeScale = 'on' to use large scale methods, 'off' to use medium
        Fit.Result.Alpha(s,iter) = res(1);
        Fit.Result.Beta(s,iter) = res(2);
        Fit.Result.Lambda(s,iter) = res(3);
        Fit.Result.Lik(s,iter) = lik;
        Fit.Result.Lik  % to view progress so far
        
    end
    [a, b] = min(Fit.Result.Lik(s, :));
    Fit.Result.BestFit(s,:) = [s,...
        Fit.Result.Alpha(s,b),...
        Fit.Result.Beta(s,b),...
        Fit.Result.Lambda(s,b),...
        Fit.Result.Lik(s,b)];
end
Fit.Result.BestFit