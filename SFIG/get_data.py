import pandas as pd
import numpy as np


def get_data(data_path, dataset):
    if dataset == 'BATADAL':
        df_train_orig = pd.read_csv(
            data_path + dataset + "/train_dataset_datetime.csv", parse_dates=['DATETIME'], dayfirst=True)
        try:
            df_train_orig = df_train_orig.drop(columns=['Unnamed: 0'])
        except KeyError:
            pass
        actuator_columns = df_train_orig.filter(
            regex=("STATUS")).columns.tolist()
        escape_columns = ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']
        sensor_columns = [
            col for col in df_train_orig.columns if col not in actuator_columns + escape_columns]

    if dataset == 'WADI':
        df_train_orig = pd.read_csv(data_path + dataset+"/14_days_clean.csv",
                                    parse_dates={'DATETIME': ['Date', 'Time']}, dayfirst=True)
        #resampling
        #df_train_orig = df_train_orig.iloc[::5, :]
        df_train_orig = df_train_orig.reset_index()
        
        actuator_columns = df_train_orig.filter(
            regex=("STATUS")).columns.tolist()
        escape_columns = ['Row', 'Date', 'Time', 'DATETIME', '2_MV_001_STATUS',
                          '2_LT_001_PV', '2_MV_002_STATUS']
        sensor_columns = [
            col for col in df_train_orig.columns if col not in actuator_columns + escape_columns]

    if dataset == 'SWAT':
        df_train_orig = pd.read_csv(
            data_path + dataset+'/Normal_v0.csv', dayfirst=True, header=1, sep=';', decimal=",")
        df_train_orig = df_train_orig.iloc[16000:]
        #resampling
        df_train_orig = df_train_orig.iloc[::20, :]
        df_train_orig = df_train_orig.reset_index()
        
        actuator_columns = df_train_orig.filter(
            regex=("(MV|P[0-9]|UV)")).columns.tolist()
        escape_columns = [' Timestamp', 'Normal/Attack']
        sensor_columns = [
            col for col in df_train_orig.columns if col not in actuator_columns + escape_columns]

    return df_train_orig, actuator_columns, sensor_columns, escape_columns

def get_attack_data(data_path, dataset, data):
    if dataset == 'BATADAL' or dataset == '':
        try:
            df_test_1 = pd.read_csv(
                data_path + dataset + "/constrained_spoofing/" 
                + data +".csv", parse_dates=['DATETIME'], dayfirst=True)
        except FileNotFoundError:
            try:
                df_test_1 = pd.read_csv(
                    data_path + dataset + "/unconstrained_spoofing/" 
                    + data +".csv", parse_dates=['DATETIME'], dayfirst=True)
            except FileNotFoundError:
                df_test_1 = pd.read_csv(
                    data_path + dataset + '/' 
                    + data +".csv", parse_dates=['DATETIME'], dayfirst=True)
        try:
            df_test_1 = df_test_1.drop(columns=['Unnamed: 0'])
        except KeyError:
            pass
        actuator_columns = df_test_1.filter(
            regex=("STATUS")).columns.tolist()
        escape_columns = ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']
        sensor_columns = [
            col for col in df_test_1.columns if col not in actuator_columns + escape_columns]
        df_test = df_test_1
        
    return df_test, actuator_columns, sensor_columns, escape_columns
