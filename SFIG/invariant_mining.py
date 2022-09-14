from itemset_generation import Itemset
from predicate_generation import Predicates
import subprocess

from get_data import get_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from anytree import Node, RenderTree, AsciiStyle
from anytree.cachedsearch import findall
import os

import csv
import time
import random

import pickle

import warnings
warnings.filterwarnings('ignore') 

random.seed(1234)


def MIS_computation(support, beta, LS):  # LS is Least support for item i
    support = support
    support = (support*beta).astype(int)
    MIS = np.maximum(support, LS)

    return MIS


def calculate_minimum_supports(beta, LS):
    MIS_supports = pd.Series()
    supports = pd.Series()
    for c in itemset.columns.values:
        support = itemset[c].value_counts()
        support.index = support.index.astype(int)
        support = support.rename('support')
        MIS = MIS_computation(support, beta, LS)
        MIS_supports = MIS_supports.append(MIS)
        supports = supports.append(support)
    pd.DataFrame(supports).to_csv(model_folder+'supports_'+dataset+'.csv')
    pd.DataFrame(MIS_supports).to_csv(model_folder+
        'MIS_'+dataset+'.csv', sep=" ", header=False)
    
def mine_data(beta, LS, itemset, prob, duration):
    calculate_minimum_supports(beta, LS)
    # prepare the itemset file for the java libarary
    def remove_nan(array): return array[~np.isnan(array)].astype(int).tolist()
    itemset = [remove_nan(rule) for rule in itemset.values]

    with open(model_folder+'itemset_'+dataset+'_nospace.csv', "w", newline="") as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(itemset)

    # call java library
    from subprocess import Popen, PIPE, STDOUT
    p = Popen(["java", "-jar", "spmf.jar", "run",  "CFPGrowth++",
               model_folder+'itemset_'+dataset+'_nospace.csv',  '/dev/stdout', 'MIS_'+dataset+'.csv'], stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    root = Node("root", support=1)

    # parse and process java output
    start_time = None
    iterations = 0
    for line in p.stdout:

        if any(word in str(line) for word in ['=', 'Transactions', 'Max', 'Frequent', 'Total']):
            print(line.decode("utf8").strip())
            continue
        if 'spmf.jar' not in str(line) and start_time == None:
            start_time = time.time()

        if 'spmf.jar' not in str(line):
            
            if random.uniform(0, 1) < prob:  # 0.00002:
                elapsed_time = time.time() - start_time
                if elapsed_time > duration:
                    break
                if iterations > 1000:
                    iterations = 0
                    print("Elapsed time[s]/Time budget[s]: {} / {} ".format(elapsed_time, duration))
                    print("Number of mined rules: {}".format(len(root.leaves)))
                iterations = iterations + 1
                # print(len(root.leaves))
                candidate_itemset = str(line.decode("utf8").strip()).split(' ')
                last_elem = len(candidate_itemset) - 2
                support = int(candidate_itemset[-1])
                candidate_itemset = list(map(int, candidate_itemset[0: last_elem]))
                depth = 0
                last_parent = root

                found_enough = findall(
                    root, lambda node: node.support == support and node.name == candidate_itemset[-1])
                if len(found_enough) >= 1:
                    continue
                    
                for elem in candidate_itemset:      
                    found_parent = findall(
                        last_parent, lambda node: node.name == elem and node.support >= support)  # (map(lambda x: x.support == support, node.leaves))
                    if found_parent:
                        found_parent = found_parent[0]
                    else:
                        found_parent = None
                    if found_parent == None:
                        child = Node(elem, support=support)
                        child.parent = last_parent
                        last_parent = child
                    else:
                        last_parent = found_parent
                    depth = depth + 1
                last_parent.support = int(support)

    #print(RenderTree(root, style=AsciiStyle()).by_attr('name'))
    #print(RenderTree(root, style=AsciiStyle()).by_attr('support'))
    print(len(root.leaves))
    from anytree.exporter import JsonExporter
    import json
    with open(model_folder+'mined_rules_'+dataset+'.json', 'w') as f:
        exporter = JsonExporter(indent=2)
        f.write(exporter.export(root))
    


if __name__ == "__main__":



    data_path = "../Spoofing Framework/"
    dataset = 'BATADAL'
    model_folder = './reproduce_results/'
    forlder_exists =  os.path.exists(model_folder)
    if forlder_exists:
        print('++++++++++++++WARNING: FOLDER ALREADY EXISTS CONTENT WILL BE OVERWRITTEN!++++++++++++++')
    else:
        os.mkdir(model_folder)

    try:
        itemset = pd.read_csv(model_folder+'itemset_'+dataset+'.csv', sep=" ", skiprows = 1, header=None)

    except FileNotFoundError:

        df_train_orig, actuator_columns, sensor_columns, escape_columns = get_data(
            data_path, dataset)

        scaler = MinMaxScaler()
        df_train_orig[sensor_columns] = pd.DataFrame(scaler.fit_transform(
            df_train_orig[sensor_columns]), index=df_train_orig.index, columns=[sensor_columns])

        print("Predicate Generation")
        predicates = Predicates(
            df_train_orig, sensor_columns, actuator_columns)
        print("Generated the following number of predicates: ")
        predicates.to_string()
        
        with open(model_folder+"predicates_"+dataset+".pickle", "wb") as output_file:
            pickle.dump(predicates, output_file)
        

        def difference(x):
            x = x.reset_index(drop=True)
            return (x[1] - x[0])

        rolling_sensor_time_series = df_train_orig[sensor_columns].rolling(
            window=2).apply(difference)

        itemset = Itemset(predicates, df_train_orig,
                          rolling_sensor_time_series, actuator_columns)
        pd.DataFrame(itemset.itemset).to_csv(
            model_folder+'itemset_with_header_'+dataset+'.csv', index=False, header=True)
        pd.DataFrame(itemset.itemset.values).to_csv(
            model_folder+'itemset_'+dataset+'.csv', sep=" ", index=False, header=False)
        itemset = pd.read_csv(model_folder+'itemset_'+dataset+'.csv', sep=" ", header=None)
        
    beta = 0.9
    LS = int(0.32*len(itemset)) 
    prob = 0.001
    duration = 600 #3600*12 #12 hours to reproduce the detector considered in the paper
    mine_data(beta, LS, itemset, prob, duration)
    
