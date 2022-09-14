classdef ROLLING_detector
    %ROLLING_DETECTOR Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        features
        bin
        stds_of_zero_mean
    end
    
    methods
        function detector = ROLLING_detector(data, bin)
            detector.features = [];
            detector.stds_of_zero_mean = [];
            detector.bin = bin;
            for sensor = 1:length(data(1,:))
                
                hankel = transpose(createRollingWindow(data(:,sensor),  detector.bin));
                rolling_mean = mean(hankel);
                rolling_std = std(hankel);
                

                mean_of_zero_variances = mean(rolling_mean(rolling_std==0));
                std_of_zero_variances = std(rolling_mean(rolling_std==0));
              
                
                if mean_of_zero_variances == 0 || isnan(mean_of_zero_variances) %isnan is to check if there is no window where there was no update (i.e. the sensor updaates at leat onece in each window)
                    %find parameters for detection
                    [percentile, max_prc] = binary_search(rolling_std, 0, 100);
                    detector.stds_of_zero_mean = [detector.stds_of_zero_mean; prctile(rolling_std,percentile)];
                    detector.features = [detector.features; sensor];
                end
            end
        end
        
        function predictions = anomaly_detection(detector, ground_truth, data, print, plot)
            predictions = zeros([length(ground_truth) 1]);
            min_std = 1000000000+zeros([1 length(ground_truth)-(detector.bin)+1]); %+1 for correct dimensions at line 54
            for j = 1:length(detector.features)
                sensor = detector.features(j);
                hankel = transpose(createRollingWindow(data(:,sensor),  detector.bin));
                rolling_mean = mean(hankel);
                rolling_std = std(hankel);
                rolling_std_norm = rolling_std/detector.stds_of_zero_mean(j);
                min_std(rolling_mean>0) = min(rolling_std_norm(rolling_mean>0), min_std(rolling_mean>0)); 
                for i = detector.bin:length(rolling_mean)
                    if rolling_mean(i) > 0 && rolling_std(i) < detector.stds_of_zero_mean(j)
                        predictions(i) = 1;
                    end
                end
            end
  
            max_std = ones([1 length(min_std)])./min_std;
            max_std = min(1.8+zeros([1 length(max_std)]), max_std);
            if plot == true
                plotchart(max_std, 1);
            end
            if print == true
                [accuracy, precision, recall, f1, fbeta, fpr] = compute_scores(ground_truth, predictions);
                fprintf('Sliding Accuracy: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
            end
        end
    end
end

%% Class related functions

 function rolling_vector = createRollingWindow(vector, n)
 % https://de.mathworks.com/matlabcentral/answers/171154-create-rolling-window-matrix-from-vector#answer_166697
 % CREATEROLLINGWINDOW returns successive overlapping windows onto a vector
 %   OUTPUT = CREATEROLLINGWINDOW(VECTOR, N) takes a numerical vector VECTOR
 %   and a positive integer scalar N. The result OUTPUT is an MxN matrix,
 %   where M = length(VECTOR)-N+1. The I'th row of OUTPUT contains
 %   VECTOR(I:I+N-1).
    l = length(vector);
    m = l - n + 1;
    rolling_vector = vector(hankel(1:m, m:l));
 end
 
function [min_range, max_range] = binary_search(data, min_range, max_range)
    increment = (max_range - min_range)/2;
    for perc = min_range:increment:max_range
        percentile = prctile(data,perc);
        if percentile > 0
            if increment <= 0.1
                min_range = perc;
                max_range = perc; 
                return;
            else
                min_index = perc-increment;
                if min_index < 0
                    min_index = 0;
                end
                [min_range, max_range] = binary_search(data, min_index, perc);
                return;
            end
        end
    end
end


function  plotchart(min_std, threshold)
    newplot
    n = length(min_std);
    plot(1:numel(min_std),min_std(:));
    line([1 n],threshold*[1 1],'LineStyle',':');
end
