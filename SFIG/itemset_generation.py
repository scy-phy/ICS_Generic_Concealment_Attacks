import pandas as pd
from predicate_generation import Predicates
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# for every


class Itemset():
    """
    Check which predicates hold every time step
    """

    def __init__(self, predicates, df_train_orig, rolling_sensor_time_series, actuator_columns):
        self.itemset = pd.DataFrame()
        self.get_categorical_items( df_train_orig[1:], actuator_columns)
        self.get_distibution_driven_items(
            predicates.distribution_driven_predicates, rolling_sensor_time_series[1:])
        self.get_event_driven_items(
            predicates.event_driven_predicates, df_train_orig[1:])
        index = 0
        for c in self.itemset.columns.values:
            self.itemset[c] = self.itemset[c].add(index*10)
            if c in  self.itemset.filter(regex=("(ED_)")).columns:
                self.itemset[c] = self.itemset[c].replace(index*10, '')
            
            index = index + 1
        

    def get_distibution_driven_items(self, distribution_driven_predicates, data):
        """
        For every time step predict the cluster for the given sensor update

        Parameters
        ----------
        distribution_driven_predicates : list
            list of distribution driven predicates objects
        data : pandas DataFrame
        """

        for distribution in distribution_driven_predicates:
            column = pd.DataFrame(data=distribution.gmm.predict(
                data[distribution.sensor].values.reshape(-1, 1)), index=data.index, columns=['DD_'+str(distribution.sensor)])
            self.itemset = pd.concat([self.itemset, column], axis=1)

    def get_event_driven_items(self, event_driven_predicates, data):
        """
        For every timestep, for every trigger compute prediction with the linear model.

        (it gives 1 only when an event is occurred)

        Parameters
        ----------
        event_driven_predicates : list
            list of event driven predicates objects
        data : pandas DataFrame
        """
        count = 0
        for event_trigger in event_driven_predicates:
            count = count + 1
            y = data[event_trigger.target_sensor]
            preds = event_trigger.model.predict(
                data[event_trigger.training_features])
            column = pd.DataFrame(data=1*(np.sqrt(np.square(y.values - preds)) < len(y)*[event_trigger.epsilon]),
                                  index=data.index, columns=['ED_'+str(event_trigger.target_sensor)+'_'+str(count)])
            
            self.itemset = pd.concat([self.itemset, column], axis=1)

    def get_categorical_items(self, data, actuator_columns):
        """
        For every timestep return the state of actuators

        Parameters
        ----------
        row : [type]
            [description]
        """
        self.itemset = pd.concat([self.itemset, data[actuator_columns]], axis=1)
 


if __name__ == "__main__":
    data_path = "./Spoofing Framework/"
    dataset = 'BATADAL'
    if dataset == 'BATADAL':
        df_train_orig = pd.read_csv(
            data_path + dataset + "/train_dataset_datetime.csv", parse_dates=['DATETIME'], dayfirst=True)
        df_train_orig = df_train_orig.drop(columns=['Unnamed: 0'])
        actuator_columns = df_train_orig.filter(
            regex=("STATUS")).columns.tolist()
        escape_columns = ['Unnamed: 0', 'DATETIME', 'ATT_FLAG']
        sensor_columns = [
            col for col in df_train_orig.columns if col not in actuator_columns + escape_columns]

    scaler = MinMaxScaler()
    df_train_orig[sensor_columns] = pd.DataFrame(scaler.fit_transform(
        df_train_orig[sensor_columns]), index=df_train_orig.index, columns=[sensor_columns])

    predicates = Predicates(df_train_orig, sensor_columns, actuator_columns)
    print("Generated the following number of predicates: ")
    predicates.to_string()

    def difference(x):
        return (x[1] - x[0])

    rolling_sensor_time_series = df_train_orig[sensor_columns].rolling(
        window=2).apply(difference)

    # for index, row in time_series[1:].iterrows():
    itemset = Itemset(predicates, df_train_orig, rolling_sensor_time_series, actuator_columns)

    # print(itemset.itemset.describe())
    pd.DataFrame(itemset.itemset.values).to_csv( 'itemset_'+dataset+'.csv', sep=[' '] , header=False)
    supports = pd.Series()
    for c in itemset.itemset.columns.values:
        series = itemset.itemset[c].value_counts()
        series = series.rename('count')
        supports = supports.append(series)
    pd.DataFrame(supports).to_csv('supports.csv', sep=[' '] , header=False)