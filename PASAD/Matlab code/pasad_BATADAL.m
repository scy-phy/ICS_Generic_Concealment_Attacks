% A Matlab implementaion of PASAD (Process-Aware Stealthy-Attack
% Detection), which is a real-time intrusion detection system for control
% systems. The file contains three code sections to be executed separately
% in the order given. The sections correspond to the training phase,
% detecton phase, and a plot of the result respectively.
close all;
clear all;
clc;

Train = readtable('../../Spoofing Framework/BATADAL/train_dataset_datetime.csv');
Test = readtable('../../Spoofing Framework/BATADAL/test_dataset_1_datetime.csv');
Stale = readtable('../../Spoofing Framework/BATADAL/unconstrained_spoofing/test_dataset_1_stale.csv');
Replay  = readtable('../../Spoofing Framework/BATADAL/unconstrained_spoofing/test_dataset_1_replay.csv');
Random_replay = readtable('../../Spoofing Framework/BATADAL/unconstrained_spoofing/test_dataset_1_random_replay.csv');
Acsac = readtable('../../BATADAL_BLACK_BOX_ATTACKS_ACSAC/unconstrained_attack/test_dataset_1_unconstrained_newAE.csv');
%%
%BATADAL
train = table2array(Train(:, 15));
test = table2array(Test(:,15));
stale = table2array(Stale(:,15));
random_replay = table2array(Random_replay(:,15));
acsac = table2array(Acsac(:,15));
generic_attacks = {random_replay, stale, acsac};
generic_attacks_names = {'Random Replay', 'Stale', 'Learning-based'};
ground_truth = table2array(Test(:, 45));
params = [250, 125];
statistical_dim = 18;

%% Training Phase
% Obtaining a mathematical representation of the process behavior by
% determining the statistical dimension and the principal components of the
% signal subspace.

s = train;
testing = stale;

I = params;
N = I(1); L = I(2);
T = length(testing);%s);
K = N-L+1;

% Defining custom colors
bk = [.24 .24 .36]; 
rd = [1 .5 .5]; 
gr = [.8 1 .1]; 
bl = [.7 .85 1]; 

% Range vector corresponding to the sensor measurements affected by the
% attack (Should be adapted according to the entered time series. Defaulted
% to TE non-hf series).
atck_rg = find(ground_truth == 1);

% Constructing the (Hankel) trajectory matrix and solving its Singular
% Value Decomposition (SVD).
X = hankel(s(1:L),s(L:N));
disp('SVD decomposition started ...');tic
[t,e,~] = svd(X); 
ev = diag(e);
disp('SVD decomposition complete');toc

% Determining the statistical dimension of the time series.
es = (ev(2:end)./sum(ev(2:end)))*100;
figure
plot(es,'color',[.4 .4 .4],'linewidth',2),hold on,
plot(es,'rx','color',[1 .4 .2]);
xlabel('Number of eigenvalues');
ylabel('Eigenvalue share')
title('Scree plot');
set(gca,'fontsize',16);
r = statistical_dim;
close gcf
disp('Training PASAD is complete.');

% Constructing the matrix whose columns form an orthonormal basis for the
% signal subspace.
U = t(:,(1:r));

% Computing the centroid of the cluster formed by the training lagged
% vectors in the signal subspace.
c = mean(X,2);
%disp(c);
utc = U'*c;

% A vector containing the normalization weights for computing the squared
% weighted Euclidean distance in the detection phase.
nev = sqrt(ev(1:r)./sum(ev(1:r)));

% Reconstring the approximate signal using the diagonal averaging step in
% Singular Spectrum Analysis (SSA).
disp('Reconstructing signal ...');tic
ss = U*(U'*X);

sig = zeros(N,1);  

for k = 0:L-2
    for m = 1:k+1
        sig(k+1) = sig(k+1)+(1/(k+1))*ss(m,k-m+2);
    end
end

for k = L-1:K-1
    for m = 1:L
        sig(k+1) = sig(k+1)+(1/(L))*ss(m,k-m+2);
    end
end

for k = K:N
    for m = k-K+2:N-K+1
        sig(k+1) = sig(k+1)+(1/(N-k))*ss(m,k-m+2);
    end
end
disp('Signal reconstruction complete');toc


%% Validation Phase
% Tracking the distance from the centroid by iteratively computing the
% departure score for every test vector.

disp('Validation started...');tic
d = zeros(T-N,1);

% Constructing the first test vector.
x = s(N-L+1:N);

    for i = length(s)-N+1:length(s)
        % Constructing the current test vector by shifting the elements to
        % the left and appending the current sensor value to the end.
        x = x([2:end 1]);
        x(L) = s(i);  
        
        % Computing the difference vector between the centroid of the
        % cluster and the projected version of the current test vector.
        y = utc - U'*x;
        
        % Computing the weighted norm of the difference vector.
        y = nev.*y;
        d(i-N) = y'*y;
    end
disp('Testing complete.');toc
threshold = max(d)+0.01;

%% Testing and Generic Attacks
d_original = detection(test, T, L, utc, U, nev);
disp('Original')
plot_results('Original', d_original, d_original, threshold, atck_rg, test, Test, bk, rd, ground_truth);
%%
for attack = 1:length(generic_attacks)
    attack_name = generic_attacks_names{attack};
    disp(attack_name);
    attack_data = generic_attacks{attack};
    d = detection(attack_data, T, L, utc, U, nev);
    plot_results(attack_name, d_original, d, threshold, atck_rg, attack_data, Test, bk, rd, ground_truth);
end

%% Detection Phase
% Tracking the distance from the centroid by iteratively computing the
% departure score for every test vector.
function d = detection(data, T, L, utc, U, nev)
    %disp('Testing started...');tic
    d = zeros(T,1);

    % Constructing the first test vector.
    x = data(1:L);%s(N-L+1:N);

        for i = 1:T
            % Constructing the current test vector by shifting the elements to
            % the left and appending the current sensor value to the end.
            x = x([2:end 1]);
            x(L) = data(i);%s(i); 

            % Computing the difference vector between the centroid of the
            % cluster and the projected version of the current test vector.
            y = utc - U'*x;

            % Computing the weighted norm of the difference vector.
            y = nev.*y;
            d(i) = y'*y;

        end

    %disp('Testing complete.');toc
end

%% Plof of the result
function plot_results(plot_name, d_original, d, threshold, atck_rg, testing, Test, bk, rd, ground_truth)
    indexes = atck_rg(diff(atck_rg) > 1);
    intervals = zeros([length(indexes)+1 2]);
    intervals(1, :) =  [1 find(atck_rg == indexes(1))];
    for i = 2:length(indexes)
        intervals(i,:) = [find(atck_rg==indexes(i-1))+1 find(atck_rg == indexes(i))];
    end
    intervals(end, :) = [find(atck_rg == indexes(end))+1 length(atck_rg)];
    s = testing;
    figure
    ax1 = subplot(2,1,1);hold on
    plot(Test.DATETIME, s,'color',bk);
    %SWAT
    %ylim([600 1200]);

    %plot(s(1:N),'color',bl,'linewidth',1);
    %plot(sig,'color',gr,'linewidth',1);
    for dim = 1:size(intervals, 1)
        plot(Test.DATETIME(atck_rg(intervals(dim, 1) : intervals(dim,2))), s(atck_rg(intervals(dim, 1): intervals(dim,2))), 'color', rd);%atck_rg, s(atck_rg),'color',rd);
    end

    ylabel('Sensor Meas.');
    set(gca,'fontsize',26,'linewidth',1.5);
    legend('Normal','Attack');
    ax2 = subplot(2,1,2);hold on 
    trans = plot(Test.DATETIME, d_original,'color', [0.75 0.75 0.75],'linewidth',2);
    %alpha(trans, .5);


    plot(Test.DATETIME, d,'color','b','linewidth',2);
    %SWAT
    %plot(Test.DATETIME, ones(size(Test))*3000000, '--r');
    ylim([0 1000]);
    %BATADAL
    plot(Test.DATETIME, ones(size(Test))*threshold, '--r');

    xlabel('Observation Index');
    ylabel('Departure Score');
    legend('Original', plot_name, 'Threshold');
    set(gca,'fontsize',26,'linewidth',1.5);

    linkaxes([ax1,ax2],'x');
    %SWAT
    %detection_indexes = (d>=3000000);
    %BATADAL
    detection_indexes = (d>=threshold);
    conf_matrix = confusionmat(ground_truth, double(detection_indexes), 'Order', [1,0]);
    hold off;
    %ax3 = subplot(3,1,3);
    %confusionchart(conf_matrix);
    tp = conf_matrix(1, 1);
    tn = conf_matrix(2, 2);
    fp = conf_matrix(2, 1);
    fn = conf_matrix(1, 2);

    accuracy = (tp+tn)/(tp+tn+fp+fn);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f1score = 2*((precision*recall)/(precision+recall));
    fpr = fp/(fp+tn);
    fprintf('Recall: %.3f Precision: %.3f F1-score: %.3f Acuracy: %.3f FPR: %.3f\n', recall, precision, f1score, accuracy, fpr);
end