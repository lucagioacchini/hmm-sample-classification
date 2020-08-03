clear all
close all
%%
rng('default');
tstart=tic;
Kquant=8;% number of quantization levels
Nstates=8;% number of states in the HMM
ktrain=[1,2,3,4,5,6,7];% indexes of patients for training
ktest=[8,9,10];% indexes of patients for testing
tol = 1e-3;
maxiter = 200;

[hq,pq]=pre_process_data(Nstates,Kquant,ktrain);% generate the quantized signals
telapsed = toc(tstart);
disp(['first part, elapsed time ',num2str(telapsed),' s'])

%% HMM training phase circular transition matrix
% hmm parameters
p = 0.9;
q = (1-p)/(Nstates-1);
% define seq
hq_train = hq(ktrain);
pq_train = pq(ktrain);
% initial matrices
transition_circular = q*ones(Nstates, Nstates);
for i=1:Nstates
    for j=1:Nstates
        if (i == j)
            transition_circular(i, j) = p;
        end
    end
end
transition_circular = circshift(transition_circular, -1);

transition_rand = rand(Nstates);
transition_rand = bsxfun(@rdivide,transition_rand,sum(transition_rand,2));

emission_rand = rand(Nstates);
emission_rand = bsxfun(@rdivide,emission_rand,sum(emission_rand,2));

% First Machine - Healthy
[transH_1, emissH_1] = hmmtrain(hq_train, transition_circular, emission_rand,'Algorithm', 'BaumWelch', 'Tolerance', tol, 'Maxiterations', maxiter);
[transH_2, emissH_2] = hmmtrain(hq_train, transition_rand, emission_rand,'Algorithm', 'BaumWelch', 'Tolerance', tol, 'Maxiterations', maxiter);
% Second Machine - Diseased
[transP_1, emissP_1] = hmmtrain(pq_train, transition_circular, emission_rand,'Algorithm', 'BaumWelch', 'Tolerance', tol, 'Maxiterations', maxiter);
[transP_2, emissP_2] = hmmtrain(pq_train, transition_rand, emission_rand,'Algorithm', 'BaumWelch', 'Tolerance', tol, 'Maxiterations', maxiter);

%% HMM testing phase....

full_set = cat(2,pq,hq);

% sensitivity init
tpr_1 = 0;
tpr_2 = 0;
% specificity init
tnr_1 = 0;
tnr_2 = 0;

% ground truth
gt = zeros(20, 1);
gt(11:20) = 1;

for i=1:20
    % First Machine - Healthy
    [postH_1, logH_1] = hmmdecode(full_set{:, i}, transH_1, emissH_1);
    [postH_2, logH_2] = hmmdecode(full_set{:, i}, transH_2, emissH_2);
    % Second Machine - Healthy
    [postP_1, logP_1] = hmmdecode(full_set{:, i}, transP_1, emissP_1);
    [postP_2, logP_2] = hmmdecode(full_set{:, i}, transP_2, emissP_2);
        
    % classification
    % Positive --> Healthy
    % Negative --> Diseased
    if logH_1>logP_1
        class_1(i) = 1;
    else
        class_1(i) = 0;
    end
    
    if logH_2>logP_2
        class_2(i) = 1;
    else
        class_2(i) = 0;
    end
end


% distinguish between healthy and diseased patients
classP_1 = class_1(1:10);
classH_1 = class_1(11:20);
classP_2 = class_2(1:10);
classH_2 = class_2(11:20);

% distinguish between train and test
class_test_1 = cat(2, classP_1(ktest), classH_1(ktest));
class_test_2 = cat(2, classP_2(ktest), classH_2(ktest));
class_train_1 = cat(2, classP_1(ktrain), classH_1(ktrain));
class_train_2 = cat(2, classP_2(ktrain), classH_2(ktrain));

% sensitivity and specificity init
tpr_test_1 = 0;
tnr_test_1 = 0;
tnr_test_2 = 0;
tpr_test_2 = 0;
tpr_train_1 = 0;
tnr_train_1 = 0;
tnr_train_2 = 0;
tpr_train_2 = 0;

%% SENSITIVITY AND SPECIFICITY TEST
% Sensitivity and Specificity Test Circular transition matrix
for i=1:size(class_test_1, 2)/2
    if class_test_1(i) == 0
        tnr_test_1 = tnr_test_1 + 1; 
    end   
end
for i=size(class_test_1, 2)/2:size(class_test_1, 2)
    if class_test_1(i) == 1
        tpr_test_1 = tpr_test_1 + 1; 
    end   
end
% Sensitivity and Specificity Test Random transition matrix
for i=1:size(class_test_2, 2)/2
    if class_test_2(i) == 0
        tnr_test_2 = tnr_test_2 + 1; 
    end   
end
for i=size(class_test_2, 2)/2:size(class_test_2, 2)
    if class_test_2(i) == 1
        tpr_test_2 = tpr_test_2 + 1; 
    end   
end

%% SENSITIVITY AND SPECIFICITY TRAIN
% Sensitivity and Specificity Train Circular transition matrix
for i=1:size(class_train_1, 2)/2
    if class_train_1(i) == 0
        tnr_train_1 = tnr_train_1 + 1; 
    end   
end
for i=size(class_train_1, 2)/2:size(class_train_1, 2)
    if class_train_1(i) == 1
        tpr_train_1 = tpr_train_1 + 1; 
    end   
end
% Sensitivity and Specificity Train Random transition matrix
for i=1:size(class_train_2, 2)/2
    if class_train_2(i) == 0
        tnr_train_2 = tnr_train_2 + 1; 
    end   
end
for i=size(class_train_2, 2)/2:size(class_train_2, 2)
    if class_train_2(i) == 1
        tpr_train_2 = tpr_train_2 + 1; 
    end   
end

tnr_test_1 = tnr_test_1/size(ktest, 2) % TNR TEST, CIRCULAR
tpr_test_1 = tpr_test_1/size(ktest, 2) % TPR TEST, CIRCULAR
tnr_test_2 = tnr_test_2/size(ktest, 2) % TNR TEST, RANDOM
tpr_test_2 = tpr_test_2/size(ktest, 2) % TPR TEST, RANDOM

tnr_train_1 = tnr_train_1/size(ktrain, 2) % TNR TEST, CIRCULAR
tpr_train_1 = tpr_train_1/size(ktrain, 2) % TPR TEST, CIRCULAR
tnr_train_2 = tnr_train_2/size(ktrain, 2) % TNR TEST, RANDOM
tpr_train_2 = tpr_train_2/size(ktrain, 2) % TPR TEST, RANDOM
