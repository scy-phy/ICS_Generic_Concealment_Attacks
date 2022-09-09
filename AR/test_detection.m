clear all;
close all;
clc;

Train =  readtable('../Spoofing Framework/BATADAL/train_dataset_datetime.csv');
Test_1 = readtable('../Spoofing Framework/BATADAL/test_dataset_1_datetime.csv');

Attack_stale_1 = readtable('../Spoofing Framework/BATADAL/unconstrained_spoofing/test_dataset_1_stale.csv');
Attack_random_replay_1 = readtable('../Spoofing Framework/BATADAL/unconstrained_spoofing/test_dataset_1_random_replay.csv');
Attack_acsac = readtable('../BATADAL_BLACK_BOX_ATTACKS_ACSAC/unconstrained_attack/test_dataset_1_unconstrained_newAE.csv');

disp('Test');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(15, 5.5, 1, Train, Test_1);
disp('Random Replay');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(15, 5.5, 1, Train, Attack_random_replay_1);
disp('Stale');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(15, 5.5, 1, Train, Attack_stale_1);
disp('Learning-based');
[accuracy_1, precision_1, recall_1, f1_1, fpr_1] = AR_detection(15, 5.5, 1, Train, Attack_acsac);
