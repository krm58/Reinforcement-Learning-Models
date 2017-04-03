function [lik] = FitModel_RW(CSchosen,Rewards,eta,beta,s)
%LOG LIKELIHOOD FUNCTION, run many times. 
%eta = learning rate

% Find the log likelihood of the choice data under a TD model with learning rate
% eta, softmax temperature beta

% Outputs:
% Lik = - log likelihood of the data

lik    = 0;   % log likelihood

%in each pair, even is good, odd is bad
%Trial type 0 is positive, 1 is negative, 2 is mixed
CS_num = [1 2; 3 4; 5 6]; % The stimuli used for each trial type
V = [0, 0, 0, 0, 0, 0];  % initial values for the 6 stimuli

global V_end;

for t = 1:length(CSchosen)  % t = trial number
    c = 0;
    n = 0;
    %pick the current belief on the rewards to come from each action
    if (isnan(CSchosen(t))) 
        continue
    end
    if (CSchosen(t) == 1) %if they chose A, they didn't choose B
        c = 1;
        n = 2;
    elseif (CSchosen(t) == 2)
        c = 2;
        n = 1;
    elseif (CSchosen(t) == 3)
        c = 3;
        n=4;
    elseif (CSchosen(t) == 4)
        c = 4;
        n=3;
    elseif (CSchosen(t) == 5)
        c = 5;
        n = 6;
    elseif (CSchosen(t) == 6)
        c = 6;
        n=5;
    end
    
    qt = [V(c), V(n)];
    
    % adding this choice to the log likelihood
    x = beta*V(c) - log(sum(exp(beta*qt)));
    lik = lik + x;
    
    % updating the value of the chosen stimulus according to the reward received (on choice as well as single trials)
    PE = Rewards(t) - V(c);
    V(c) = V(c) + PE*eta;
end

% OPTIONAL: putting a prior on the parameters (so we are looking for the MAP and not the ML solution)
lik = lik + log(betapdf(eta,1.1,1.1));
lik = lik + log(gampdf(beta,2,3));
lik = -lik;  % so we can minimize the function rather than maximize
V_end(s,:) = V;