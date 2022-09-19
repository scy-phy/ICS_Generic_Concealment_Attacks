function [accuracy, precision, recall, f1, fpr] = AR_detection(column, climit, mshift, Train, Test) 
    %%
    train = Train(:, column);
    train = table2array(train);
    train_idd = iddata(train, [], 1);
    sys = ar(train_idd, 20, 'ls');
    test = Test(:, column);
    test = table2array(test);
 
    
    %%
    figure;
    compare(test, sys, 1)
    [e_train,r_train] = resid(train, sys);
    [e_test,r_test] = resid(test, sys);
    try
        mfnc = mean(e_train.y);
        sfnc = std(e_train.y);
    catch
        mfnc = mean(e_train);
        sfnc = std(e_train);
    end
    figure;
    resid(train,sys);
    figure;
    resid(test,sys);
    try 
        [iupper_train, ilower_train] = cusum(e_train.y,climit,mshift,mfnc,sfnc, 'all');
    catch
        [iupper_train, ilower_train] = cusum(e_train,climit,mshift,mfnc,sfnc, 'all');
    end
    ground_truth_train = zeros([length(train) 1]);
    prediction_train = merge_cusum_results(ground_truth_train, iupper_train, ilower_train);
    [accuracy, precision, recall, f1, fpr] = compute_scores(ground_truth_train, prediction_train);
    
    fprintf('Accuracy Train: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
    
    %% TEST
    try
        [iupper_test, ilower_test] = cusum(e_test.y,climit,mshift,mfnc,sfnc, 'all');
    catch
        [iupper_test, ilower_test] = cusum(e_test,climit,mshift,mfnc,sfnc, 'all');
    end
    ground_truth_test = table2array(Test(:, 45));
    
    prediction_test = merge_cusum_results(ground_truth_test, iupper_test, ilower_test);

    [accuracy, precision, recall, f1, fpr] = compute_scores(ground_truth_test, prediction_test);
    fprintf('Accuracy Test: %f F1-score: %f Precision: %f Recall: %f FPR: %f\n', accuracy, f1, precision, recall, fpr);
    fprintf('& %.2f & %.2f & %.2f & %.2f & %.2f \\\\ \n', recall, precision, f1, accuracy, fpr);
    
    figure; 
    try
        cusum(e_test.y,climit,mshift,mfnc,sfnc, 'all');
    catch
        cusum(e_test,climit,mshift,mfnc,sfnc, 'all');
    end
    hold on;
    plot(ground_truth_test, 'g','LineWidth',5);

    function prediction = merge_cusum_results(ground_truth, iupper, ilower)
        prediction = zeros([length(ground_truth) 1]);
        for i = 1:length(ground_truth)
            prediction(i) = ismember(i, iupper);
            if prediction(i)==0
                prediction(i) = ismember(i, ilower);
            end
        end
        
   function [accuracy, precision, recall, f1, fpr] = compute_scores(ground_truth, prediction)
    conf_matrix = confusionmat(ground_truth, prediction, 'Order', [1,0]);
    tp = conf_matrix(1, 1);
    tn = conf_matrix(2, 2);
    fp = conf_matrix(2, 1);
    fn = conf_matrix(1, 2);
    accuracy = (tp+tn)/(tp+tn+fp+fn);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f1 = 2*((precision*recall)/(precision+recall));
    fpr = fp/(fp+tn);