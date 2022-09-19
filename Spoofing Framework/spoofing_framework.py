import pandas as pd
import os
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)

def identify_attacks(test_data, days):
    """
    Given the test_data identifies the attack intervals and creates a pandas DataFrame where those spoofing is going to be applied.
    
    Returns
    -------
    DataFrame
        summary of the attack intervals
    """
    # find attacks among data
    attacks = test_data.loc[test_data['ATT_FLAG'] == 1]
    prev_datetime = attacks.index[0]  # find first timing
    start = prev_datetime
    count_attacks = 0

    # find attacks bounds
    attack_intervals = pd.DataFrame(
        columns=['Name', 'Start', 'End', 'Replay_Copy'])
    for index, _ in attacks.iterrows():
        if (index - prev_datetime > datetime.timedelta(minutes=15)):
            count_attacks = count_attacks + 1
            interval = pd.DataFrame([['attack_'+str(count_attacks), start, prev_datetime, (start - (
                prev_datetime - start)) - datetime.timedelta(days=days)]], columns=['Name', 'Start', 'End', 'Replay_Copy'], index = [count_attacks])
            attack_intervals = attack_intervals.append(interval)
            start = index
        prev_datetime = index
    count_attacks = count_attacks + 1
    interval = pd.DataFrame([['attack_'+str(count_attacks), start, prev_datetime, start - (
        prev_datetime - start) - datetime.timedelta(days=350)]], columns=['Name', 'Start', 'End', 'Replay_Copy'], index = [count_attacks])
    attack_intervals = attack_intervals.append(interval)
    print('_________________________________ATTACK INTERVALS___________________________________\n')
    print(attack_intervals)
    print('____________________________________________________________________________________')
    return attack_intervals


def spoof(spoofing_technique, attack_intervals, eavesdropped_data, test_data, constraints=None):
    
    """
    Given a spoofing_technique to be applied, the attack_intervals, eavesdropped_data and test_data, it builds the dataset containing sensor spoofing.
    
    Returns
    -------
    DataFrame
        Dataset with spoofed sensor readings.
    """
    prev_end = test_data.index[0]
    df2 = pd.DataFrame()
    # spoof data
    for index, row in attack_intervals.iterrows():
        #normal behavior
        if prev_end == test_data.index[0]:
             df2 = df2.append(test_data.loc[prev_end : row['Start']-datetime.timedelta(seconds=1)])
        else:
            df2 = df2.append(test_data.loc[prev_end+datetime.timedelta(seconds=1) : row['Start']-datetime.timedelta(seconds=1)])
        df = pd.DataFrame(columns=eavesdropped_data.columns)
        if constraints:
            #print( constraints[index-1])
            df = spoofing_technique(df,
                                    row, eavesdropped_data, test_data, attack_intervals, constraints[index-1])
        else:
            df = spoofing_technique(df,
                                    row, eavesdropped_data, test_data, attack_intervals)
        df['ATT_FLAG'] = '1'
        df3 = pd.DataFrame(data=df.values, columns=df.columns,
                           index=test_data.loc[row['Start']: row['End']].index)  # update datetime
        df2 = df2.append(df3, ignore_index=False)[df3.columns.tolist()]
        prev_end = row['End']
    df2 = df2.append(
        test_data.loc[prev_end+datetime.timedelta(seconds=1): test_data.last_valid_index()])
    return df2


def replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    
    Applies replay attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    df = df.append(eavesdropped_data.loc[row['Replay_Copy']: row['Replay_Copy']+(
        row['End']-(row['Start']))])[test_data.columns.tolist()]  # append replayed row
    return df

def random_replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    
    Applies random replay attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    df = df.append(eavesdropped_data.loc[row['Replay_Copy']: row['Replay_Copy']+(
        row['End']-(row['Start']))].sample(frac=1, random_state = 531))[test_data.columns.tolist()]  # append random replayed row
    return df


def stale(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    
    Applies stale attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    lenght = len(test_data.loc[row['Start']:row['End']])
    stale=df.append(test_data.loc[row['Start']-datetime.timedelta(minutes=15)])[test_data.columns.tolist()]
    df = pd.concat([stale]*lenght)
    return df

def constrained_replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    
    """
    
    Applies constrained replay attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """

    constraints = args[0]
    check_constraints(constraints)
    df = df.append(test_data.loc[row['Start']: row['End']])
    df[constraints] = eavesdropped_data[constraints].loc[row['Replay_Copy']
        :row['Replay_Copy']+(row['End']-(row['Start']))].values
    return df

def constrained_random_replay(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    
    """
    
    Applies constrained random replay to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """

    constraints = args[0]
    check_constraints(constraints)
    df = df.append(test_data.loc[row['Start']: row['End']])
    df[constraints] = eavesdropped_data[constraints].loc[row['Replay_Copy']
        :row['Replay_Copy']+(row['End']-(row['Start']))].sample(frac=1, random_state = 531).values
    return df


def constrained_stale(df, row, eavesdropped_data, test_data, attack_intervals, *args):
    """
    
    Applies constrained stale attack to the input data
    
    Returns
    -------
    DataFrame
        data with applied replay attack
    """
    constraints = args[0]
    check_constraints(constraints)
    length = len(test_data.loc[row['Start']:row['End']])
    stale=df.append(test_data.loc[row['Start']-datetime.timedelta(minutes=15)])[test_data.columns.tolist()]
    df = df.append(test_data.loc[row['Start']: row['End']])
    for column in constraints:
        value = stale[column].values
        df[column] = [value[0]]*length
    return df

def check_constraints(constraints):
    """
    Check if constraints are provided
    """

    if constraints == None:
        print('Provide constraints')
        import sys
        sys.exit()
    else:
        pass

if __name__ == "__main__":
    
    """
    Artifact of the paper 
    Assessing Model-free Anomaly Detection in Industrial Control Systems Against Generic Concealment Attacks
    In proceedings ACSAC 2022, Austin TX, USA

    Spoofing framework tool.
    
    The spoofing framework requires:
    test_data : data organized in a .csv file where the the first row represent the features names, and all the others contains the sampled data at every time step.
    eavesdropped_data : data used to train the spoofing technique.
    
    In case of constrained attacks the program requires the files containing the constraints to be applied, organized in.
    
    The script saves the dataset with applied spoofing into the file system
    """

    data_folder_name = "BATADAL"
    test_data_name = "test_dataset_1"
    eavesdropped_data_name ="train_dataset_datetime"
    storing_folder = "BATADAL"
    days_back = 350
    unconstrained_spoofing_techniques = [random_replay, replay, stale]
    constrained_spoofing_techniques = [constrained_random_replay, constrained_replay, constrained_stale]

    test_data = pd.read_csv('./{}/{}_datetime.csv'.format(data_folder_name,test_data_name),
                            index_col=['DATETIME'], parse_dates=True)
    eavesdropped_data = pd.read_csv("./{}/{}.csv".format(data_folder_name,eavesdropped_data_name), 
                            index_col=['DATETIME'], parse_dates=True)
    
    attack_intervals = identify_attacks(test_data, days_back)

    if unconstrained_spoofing_techniques:
        actuator_columns = test_data.filter(
                regex=("STATUS")).columns.tolist()

        for spoofing_technique in unconstrained_spoofing_techniques:
            print('_________________')
            print(spoofing_technique.__name__)
            print('_________________')
            spoofed_data = spoof(spoofing_technique, attack_intervals,
                                    eavesdropped_data, test_data)
            assert len(spoofed_data) == len(test_data)
            try: 
                spoofed_data.to_csv('./{}/unconstrained_spoofing/{}_{}.csv'.format(storing_folder, test_data_name, spoofing_technique.__name__))
            except (FileNotFoundError, OSError):
                if not(os.path.isdir(storing_folder)):
                    os.mkdir(storing_folder)
                if not(os.path.isdir('./{}/unconstrained_spoofing/'.format(storing_folder))):
                    os.mkdir('./{}/unconstrained_spoofing/'.format(storing_folder))
                spoofed_data.to_csv('./{}/unconstrained_spoofing/{}_{}.csv'.format(storing_folder, test_data_name, spoofing_technique.__name__))

            
    if constrained_spoofing_techniques:

        for i in [2,3,4,5,6,7,8,9,10,15,20,25,30,35,40]:
            constraints=[]
            for att_num in [1,2,3,4,5,6,7]:
                s = open('./{}/constraints/constraint_variables_attack_{}.txt'.format(data_folder_name,att_num), 'r').read()
                dictionary =  eval(s)
                constraints.append(dictionary[i])
            
            #print(constraints)

            actuator_columns = test_data.filter(
                regex=("STATUS")).columns.tolist()

            for spoofing_technique in constrained_spoofing_techniques:
                print('_________________')
                print(spoofing_technique.__name__)
                print('constraints = '+str(i))
                print('_________________')
                spoofed_data = spoof(spoofing_technique, attack_intervals,
                                    eavesdropped_data, test_data, constraints)
                assert len(spoofed_data) == len(test_data)
                try:
                    spoofed_data.to_csv('./{}/constrained_spoofing/{}_{}_allowed_{}.csv'.format(storing_folder, test_data_name, spoofing_technique.__name__,i))
                except (FileNotFoundError, OSError):
                    if not(os.path.isdir(storing_folder)):
                        os.mkdir(storing_folder)
                    if not(os.path.isdir('./{}/constrained_spoofing/'.format(storing_folder))):
                        os.mkdir('./{}/constrained_spoofing/'.format(storing_folder))
                    spoofed_data.to_csv('./{}/constrained_spoofing/{}_{}_allowed_{}.csv'.format(storing_folder, test_data_name, spoofing_technique.__name__,i))
