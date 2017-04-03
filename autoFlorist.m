%This code generates simulated code designed for a Credit Assignment
%Reinforcement Learning task. This script creates an automated learner that
%chooses among four stimuli according to an eligibility trace RL model (see
%Sutton & Barto, 1998) and outputs the subjective value of each of the four
%stimuli. User can set the number of trials of the experiment, the learning
%rate (alpha), the softmax parameter (beta), and the eligibility trace
%parameter (lambda).

function [V, choices, selected, reward] = autoFlorist(numTrial,alpha,beta,lambda)

CS_num = [2 1; 3 1; 4 1; 2 3; 4 2; 4 3; 1 1; 2 2; 3 3; 4 4];
V = [0, 0, 0, 0];  % initialize values for the 4 stimuli
et = [0, 0, 0, 0]; %eligibility trace vector
reward = zeros(1,numTrial);
selected = zeros(1,numTrial);

% chosen(1,i) = CS_num(choices(i),1);
% notchosen(1,i) = CS_num(choices(i),2);

%make random CS pairing presentations
choices = zeros(1,numTrial);
for i = 1:numTrial
    x = randperm(10);
    choices(i) = x(1);
end

temp = 1/beta;

for t = 1:numTrial  % t = trial number
    qt = [V(CS_num(choices(t),1)); V(CS_num(choices(t),2))];
    
    smp       = exp(qt/temp) ./ sum(exp(qt/temp));
    smp = smp';
    [dum,arm] = histc(rand(1),[0,cumsum(smp)]); clear dum;
    
    %Determine reward
    if (arm == 1)
        chosen = CS_num(choices(t),1);
    elseif (arm == 2)
        chosen = CS_num(choices(t),2);
    end
    selected(t) = chosen;
    if (chosen == 1)
        reward(t) = reward(t) + 2;
    elseif (chosen == 2)
        reward(t) = reward(t) + 4;
    elseif (chosen == 3)
        if ((t + 2) <= numTrial)
            reward(t+2) = reward(t+2) + 4;
        end
    elseif (chosen == 4)
        if ((t + 2) <= numTrial)
            reward(t+2) = reward(t+2) + 8;
        end
    end
    
    rewReceived = reward(t);
    % updating the value of the chosen stimulus according to reward received
    PE = rewReceived - V(chosen);
    et(chosen) = et(chosen) + 1;
    
    for j = 1:4 %for all 4 states
        V(j) = V(j) + alpha*PE*et(j);
        et(j) = lambda*et(j);
    end
end