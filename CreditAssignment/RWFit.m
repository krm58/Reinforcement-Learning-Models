function [lik, V] = RWFit(type,response,flowers,alpha,beta)

lik = 0;   % log likelihood

%Stimuli choice pairs
CS_num = [2 1; 3 1; 4 1; 2 3; 4 2; 4 3; 1 1; 2 2; 3 3; 4 4];
chosen = zeros(200,2);
for i = 1:size(type,1)
    if type(i,1) == 'A'
        if type(i,2) == 'A'
            chosen(i,:) = CS_num(7,:);
        elseif type(i,2) == 'B'
            chosen(i,:) = CS_num(1,:);
        elseif type(i,2) == 'C'
            chosen(i,:) = CS_num(2,:);
        elseif type(i,2) == 'D'
            chosen(i,:) = CS_num(3,:);
        end
    elseif type(i,1) == 'B'
        if type(i,2) == 'A'
            chosen(i,:) = CS_num(1,:);
        elseif type(i,2) == 'B'
            chosen(i,:) = CS_num(8,:);
        elseif type(i,2) == 'C'
            chosen(i,:) = CS_num(4,:);
        elseif type(i,2) == 'D'
            chosen(i,:) = CS_num(5,:);
        end
    elseif type(i,1) == 'C'
        if type(i,2) == 'A'
            chosen(i,:) = CS_num(2,:);
        elseif type(i,2) == 'B'
            chosen(i,:) = CS_num(4,:);
        elseif type(i,2) == 'C'
            chosen(i,:) = CS_num(9,:);
        elseif type(i,2) == 'D'
            chosen(i,:) = CS_num(6,:);
        end
    elseif type(i,1) == 'D'
        if type(i,2) == 'A'
            chosen(i,:) = CS_num(3,:);
        elseif type(i,2) == 'B'
            chosen(i,:) = CS_num(5,:);
        elseif type(i,2) == 'C'
            chosen(i,:) = CS_num(6,:);
        elseif type(i,2) == 'D'
            chosen(i,:) = CS_num(10,:);
        end
    end
end

V = zeros(1,4);  % initial stimuli values
chosenStim = zeros(1,200); %chosen stimulus
nonchosenStim = zeros(1,200); %not-chosen stimulus

for t = 1:200  % t = trial number
    if (response(1,t) == chosen(t,1))
        chosenStim(1,t) = chosen(t,1);
        nonchosenStim(1,t) = chosen(t,2);
    elseif response(1,t) == chosen(t,2)
        chosenStim(1,t) = chosen(t,2);
        nonchosenStim(1,t) = chosen(t,1);
    end
end

for i = 1:200
    c = chosenStim(1,i);
    n = nonchosenStim(1,i);
    
    qt = [V(c); V(n)];
    x = beta*V(c) - log(sum(exp(beta*qt))); % adding this choice to the log likelihood
    lik = lik + x;
    
    % updating the value of the chosen stimulus according to the reward received (on choice as well as single trials)
    PE = flowers(i) - V(c);
    
    V(c) = V(c) + alpha*PE;
    
end

% % OPTIONAL: putting a prior on the parameters
lik = lik + log(betapdf(alpha,1.1,1.1));
lik = lik + log(gampdf(beta,2,3));
lik = -lik;  % so we can minimize the function rather than maximize