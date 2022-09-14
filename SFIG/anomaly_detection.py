from anytree import AnyNode
from anytree.importer import JsonImporter
from anytree.cachedsearch import findall
import json
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter
from itemset_generation import Itemset
from predicate_generation import Predicates
from sklearn.preprocessing import MinMaxScaler
from get_data import get_attack_data
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, recall_score
import numpy as np

import pickle

from anytree.exporter import JsonExporter
import json
import matplotlib.pyplot as plt
import multiprocessing

import warnings
warnings.filterwarnings('ignore') 

rules = None


def parallelize_dataframe(df, func):
    num_cores = multiprocessing.cpu_count()-1
    num_partitions = num_cores  # number of partitions to split dataframe
    df_split = np.array_split(df, num_partitions)
    pool = multiprocessing.Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def compute_scores(Y, Yhat):

    # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    # TP = np.diag(confusion_matrix)
    # TN = confusion_matrix.values.sum() - (FP + FN + TP)

    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)
    # Specificity or true negative rate
    # TNR = TN/(TN+FP)
    # Precision or positive predictive value
    # PPV = TP/(TP+FP)
    # Negative predictive value
    # NPV = TN/(TN+FN)
    # Fall out or false positive rate
    # FPR = FP/(FP+TN)
    # False negative rate
    # FNR = FN/(TP+FN)
    # False discovery rate
    # FDR = FP/(TP+FP)

    # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    fpr, recall, _ = roc_curve(Y, Yhat)
    return [accuracy_score(Y, Yhat), f1_score(Y, Yhat), precision_score(Y, Yhat), recall[1], fpr[1]]


def detection(itemsets):
    itemsets['ATT_FLAG'] = 0
    itemsets['REASON'] = 0

    for index, itemset in itemsets.iterrows():
        for rule in rules:
            if set(rule[0]).issubset(itemset.values):
                anomalous = not(set(rule[1]).issubset(itemset.values))
                if anomalous:
                    itemsets.loc[index, 'REASON'] = rule[1][0]
                    itemsets.loc[index, 'ATT_FLAG'] = 1
                    #print(itemsets[['ATT_FLAG', 'REASON']])
                    break

    return itemsets[['ATT_FLAG', 'REASON']]


if __name__ == "__main__":
    data_path = '../Spoofing Framework/'
    dataset = 'BATADAL'
    model_folder = './model_used_for_evaluation/'#'./reproduce_results/'

    importer = JsonImporter()

    with open(model_folder+'mined_rules_'+dataset+'.json', 'r') as f:
        data = f.read()
    root = importer.import_(data)
    
    found_childs = findall(root, lambda node: node.name != "root", maxlevel=2)
    rules = []
    for child in found_childs:
        leaves = child.leaves
        for leave in leaves:
            rule = [[node.name, node.support]
                    for node in leave.ancestors if node.name != 'root']
            rule.append([leave.name, leave.support])
            min_support = np.inf
            counter = 0
            position = 1
            
            if (position == 1):
                position = len(rule)-1
            rule = [[rule[i][0] for i in range(0, position)], [
                rule[i][0] for i in range(position, len(rule))]]
            rules.append(rule)
            
    print("Number of Rules: " + str(len(rules)))
    f = open(model_folder+"predicates_"+dataset+".pickle", 'rb')
    predicates = pickle.load(f)
    predicates.to_string()

    results = pd.DataFrame(
        columns=['accuracy', 'f1_score', 'precision', 'recall', 'fpr'])

    experiments = ['test_dataset_1_constrained_replay_allowed_', 
                   'test_dataset_1_constrained_newAE_']
    n = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40]
    list_experiments = []
    list_experiments += ['test_dataset_1_datetime', 'test_dataset_1_stale', 'test_dataset_1_random_replay','test_dataset_1_unconstrained_newAE']
    for experiment in experiments:
        for m in n:
            list_experiments.append(experiment+str(m))

    print('List of Experiments: '+str(list_experiments))

    for experiment in list_experiments:
        if 'newAE' in experiment:
            df_test, actuator_columns, sensor_columns, escape_columns = get_attack_data(
            '../BATADAL_BLACK_BOX_ATTACKS_ACSAC', '', experiment)
        else:
            df_test, actuator_columns, sensor_columns, escape_columns = get_attack_data(
                data_path, dataset, experiment)
        scaler = MinMaxScaler()
        df_test[sensor_columns] = pd.DataFrame(scaler.fit_transform(
            df_test[sensor_columns]), index=df_test.index, columns=[sensor_columns])

        def difference(x):
            return (x[1] - x[0])

        rolling_sensor_time_series = df_test[sensor_columns].rolling(
            window=2).apply(difference, raw=True)

        itemset = Itemset(predicates, df_test,
                          rolling_sensor_time_series, actuator_columns)

        detect = parallelize_dataframe(itemset.itemset, detection)
        # detect=detection(itemset.itemset)

        #print(detect['REASON'].value_counts(dropna=False))
        results.loc[experiment] = compute_scores(
            df_test['ATT_FLAG'].iloc[1:], detect['ATT_FLAG'])
        print(results)
        results.to_csv(model_folder+"results.csv")

