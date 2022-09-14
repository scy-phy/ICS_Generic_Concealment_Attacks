%%
%load data
%preprocessing
%train LTI
%train rolling_stat
%detection with LTI
%detection with rolling stat
%combine
%compute scores
%%
clear all;

%%Load Train Data
train = readtable('../Spoofing Framework/BATADAL/train_dataset_datetime.csv');
train(:,1)  = [];
ground_truth_train = table2array(train(:,44));
train(:,44) = [];

train_rolling = train(:,(1:31));
Train_rolling = table2array(train_rolling);
%Train_rolling = Train_rolling(20000:end,:);

output = train(:,(1:7));
train_lti = train(:,(8:31));
Train = table2array(train_lti);
Output = table2array(output);
%Output = Output(20000:end,:);
%Train = Train(20000:end,:);
Min_Train=min(Train);
Max_Train=max(Train);
Min_Output=min(Output);
Max_Output=max(Output);

Norm_train=rescale(Train, 'InputMin',Min_Train,'InputMax',Max_Train);
Norm_output=rescale(Output, 'InputMin',Min_Output,'InputMax',Max_Output);
%scaler = MinMaxScaler();
%scaler.fit(Train);
%Norm_train = scaler.transform(Train);
%Norm_test=rescale(Test, 'InputMin', Min_Train, 'InputMax', Max_Train);
%scaler2 = MinMaxScaler();
%scaler2.fit(Output);
%Norm_output = scaler2.transform(Output);
train_data = iddata(Norm_output,Norm_train,1);
%% Load Test Data

test = readtable('../Spoofing Framework/BATADAL/test_dataset_1_datetime.csv');

test(:,1)  = [];
ground_truth_test = table2array(test(:,44));
test(:,44) = [];

test_rolling = test(:,(1:31));
Test_rolling = table2array(test_rolling);

output_test = test(:,(1:7));
input_test = test(:,(8:31));
Test = table2array(input_test);
Output_test = table2array(output_test);

%Norm_test = scaler.transform(Test);
Norm_test=rescale(Test, 'InputMin', Min_Train, 'InputMax', Max_Train);
%Norm_output_test = scaler2.transform(Output_test);
Norm_output_test=rescale(Output_test, 'InputMin', Min_Output, 'InputMax', Max_Output);
test_data = iddata(Norm_output_test,Norm_test,1);

%%
climits = [10,9,8,7,6,5];
mshifts = [8,7,6,5];
outs = [1,2,3,4,5,6,7];
LTIdetector = LTI_detector(11, climits, mshifts, outs, train_data, ground_truth_train);
ROLLINGdetector = ROLLING_detector(Train_rolling, 5);
%%
disp('Original Attacks');
figure;
subplot(3,1,1);
LTIpredictions = LTIdetector.anomaly_detection(ground_truth_test, test_data, true, true);
subplot(3,1,2);
ROLLINGpredictions = ROLLINGdetector.anomaly_detection(ground_truth_test, Test_rolling,true, true);
overall = double(LTIpredictions | ROLLINGpredictions);
subplot(3,1,3);
plot(ground_truth_test, 'LineWidth',4);
hold on;
area(overall);
[accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth_test, overall);
fprintf('Test Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);

%% TEST Evasion attacks from file validation
%% Unconstrained

random_replay = readtable('../Spoofing Framework/BATADAL/unconstrained_spoofing/test_dataset_1_random_replay.csv');
stale = readtable('../Spoofing Framework/BATADAL/unconstrained_spoofing/test_dataset_1_stale.csv');
acsac = readtable('../BATADAL_BLACK_BOX_ATTACKS_ACSAC/unconstrained_spoofing/test_dataset_1_unconstrained_newAE.csv');
generic_attacks = {stale,random_replay, acsac};
generic_attacks_names = { 'Stale', 'Random Replay', 'Learning-based'};
for attack_index = 1:length(generic_attacks)
    attack = generic_attacks{attack_index};
    attack_name = generic_attacks_names{attack_index};
    disp(attack_name);
    attack(:,1)  = [];
    attack(:,44) = [];
    attack_rolling = attack(:,1:31);
    Attack_rolling = table2array(attack_rolling);

    output_attack_LTI = attack(:,(1:7));
    attack = attack(:,(8:31));
    Attack = table2array(attack);
    Output_attack = table2array(output_attack_LTI);

    %Norm_attack = scaler.transform(Attack);
    Norm_attack=rescale(Attack, 'InputMin', Min_Train, 'InputMax', Max_Train);
    %Norm_output_attack = scaler2.transform(Output_attack);
    Norm_output_attack=rescale(Output_attack, 'InputMin', Min_Output, 'InputMax', Max_Output);
    attack_data = iddata(Norm_output_attack,Norm_attack,1);

    figure;
    subplot(3,1,1);
    LTIpredictions_attack = LTIdetector.anomaly_detection(ground_truth_test, attack_data, true, true);
    subplot(3,1,2);
    ROLLINGpredictions_attack = ROLLINGdetector.anomaly_detection(ground_truth_test, Attack_rolling,true,true);
    overall_attack = double(LTIpredictions_attack | ROLLINGpredictions_attack);
    subplot(3,1,3);
    plot(ground_truth_test, 'LineWidth',4);
    hold on;
    [accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth_test, overall_attack);
    area(overall_attack);
    fprintf('Evasion Attack Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
end
%% Constrained 
generic_attacks_names = {'Replay', 'Learning-based'};
for attack_index = 1:length(generic_attacks_names)
    for i = [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]
        disp(generic_attacks_names(attack_index))
        fprintf('MAX %i\n', i)
        if strcmp(generic_attacks_names(attack_index),'Replay')
            attack = readtable(strcat('../Spoofing Framework/BATADAL/constrained_spoofing/test_dataset_1_constrained_replay_allowed_', int2str(i), '.csv'));
        end
        if strcmp(generic_attacks_names(attack_index),'Learning-based')
            attack = readtable(strcat('../BATADAL_BLACK_BOX_ATTACKS_ACSAC/constrained_spoofing/test_dataset_1_constrained_newAE_', int2str(i), '.csv'));
        end
        attack(:,1)  = [];
        attack(:,44) = [];
        attack_rolling = attack(:,1:31);
        Attack_rolling = table2array(attack_rolling);

        output_attack_LTI = attack(:,(1:7));
        attack = attack(:,(8:31));
        Attack = table2array(attack);
        Output_attack = table2array(output_attack_LTI);

        Norm_attack=rescale(Attack, 'InputMin', Min_Train, 'InputMax', Max_Train);
        Norm_output_attack=rescale(Output_attack, 'InputMin', Min_Output, 'InputMax', Max_Output);
        attack_data = iddata(Norm_output_attack,Norm_attack,1);

        
        LTIpredictions_attack = LTIdetector.anomaly_detection(ground_truth_test, attack_data, true, false);
        ROLLINGpredictions_attack = ROLLINGdetector.anomaly_detection(ground_truth_test, Attack_rolling, true, false);
        overall_attack = double(LTIpredictions_attack | ROLLINGpredictions_attack);
        
        [accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth_test, overall_attack);
        fprintf('Generic Concealment Attack Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
    end
end
%%
figure;
plot(ground_truth_test, 'LineWidth',4);
hold on;
area(overall_attack);