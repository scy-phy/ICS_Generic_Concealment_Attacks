function [accuracy, precision, recall, f1,fbeta, fpr] = compute_scores(ground_truth, prediction)
    conf_matrix = confusionmat(ground_truth, prediction, 'Order', [1,0]);
    tp = conf_matrix(1, 1);
    tn = conf_matrix(2, 2);
    fp = conf_matrix(2, 1);
    fn = conf_matrix(1, 2);
    accuracy = (tp+tn)/(tp+tn+fp+fn);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f1 = 2*((precision*recall)/(precision+recall));
    beta=2;
    fbeta = ((1 + beta^0.9) * precision * recall) / (beta^0.9 * precision + recall);
    fpr = fp/(fp+tn);
 end


