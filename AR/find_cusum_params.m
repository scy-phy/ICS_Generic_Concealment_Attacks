clear all;
close all;
clc;

best = [0,0,0,0];
Train =  readtable('../Spoofing Framework/BATADAL/train_dataset_datetime.csv');
Test = readtable('../Spoofing Framework/BATADAL/test_dataset_1_datetime.csv');

set(0,'DefaultFigureVisible','off')
column = 15;
for climit = linspace(1,6,11)
    for mshift = 1:4
        disp(column);
        disp(climit);
        disp(mshift);
        [accuracy, precision, recall, f1, fpr] = AR_detection(column, climit, mshift, Train, Test);
        if f1 > best(4)
            best = [column, climit, mshift, f1];
            disp(best);
        end
    end
end
disp('BEST PARAMETERS')
disp(best);
exit;