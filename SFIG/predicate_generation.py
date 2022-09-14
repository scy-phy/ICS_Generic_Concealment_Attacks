import numpy as np
import itertools
import math

from scipy import linalg
import matplotlib as mpl
from scipy.stats import norm


import matplotlib.pyplot as plt
from sklearn import mixture, datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import pandas as pd


class CategoricalPredicate():
    """
    Categorical predicates class, to model actuators states
    """
    def __init__(self, data, actuator_columns):
        """
        Parameters
        ----------
        data: pandas DataFrame
            the multivariate temporal series
        
        actuator_columns: list
            the list of actuator columns         
        
        Attributes
        ----------
        allowed_states : dict
            for every sensor records the categorical values that are allowed
        """
        states = data[actuator_columns].drop_duplicates()
        self.allowed_values = {}
        for column in actuator_columns:
            self.allowed_values[column] = np.sort(
                states[column].drop_duplicates().values)
    
    def to_string(self):
        #print("-----------------------")
        print("Categorical Predicates:")
        #print(self.allowed_values)
        count = 0
        for column in self.allowed_values.keys():
            count = count + len(self.allowed_values[column])
        print(count)
        #print("-----------------------")
        


class DistributionDrivenPredicate():
    """
    Distribution Driven Predicates
    ...
    Methods
    -------
    find_gmm(time_series)
        Finds the Gaussian Mixture Model with the lowest BIC score for a given sensor
    """
    def __init__(self, sensor, time_series):
        """
        Parameters
        ----------
            sensor : str
                sensor name
            time_series : pandas DataFrame
                time series of differences in sensor updates
        """

        self.sensor = sensor
        self.gmm = self.find_gmm(time_series)
        
    def to_string(self):
        print("-----------------------")
        print("Distribution Driven Predicates:")
        print(self.sensor)
        print(self.gmm.n_components)
        print("-----------------------")

    def find_gmm(self, time_series):
        """
        Find the Gaussian Mixture Model with the minimum BIC score. 
        
        Parameters
        ----------
        time_series :  pandas DataFrame
             time series of differences in sensor updates
        
        Returns
        -------
        mixture.GaussianMixture
            gaussian mixture model with lowest BIC 
        """
        X = time_series[1:].values.reshape(-1, 1)
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 4) #TODO change maximum
        cv_types = ['spherical', 'tied', 'diag', 'full']
        for cv_type in cv_types:
            for n_components in n_components_range:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components,
                                              covariance_type=cv_type)
                gmm.fit(X)
                bic.append(gmm.bic(X))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm

        bic = np.array(bic)

        clf = best_gmm
        return clf


class EventDrivenPredicate():
    """
    Event Driven Predicates
    
    for every event occurred in the system, we define that an event is triggered by sensor x if it possible to predict x given the other sensor readings at the same time step.
    """
    def __init__(self, actuator, event, sensor):
        """
        Parameters
        ----------
        actuator : string
            name of the reference actuator that changed state generating an event
        event : tuple
            transition occurred over the actuator
        sensor : string
           sensor value that we want to predict
           
        Attributes
        ----------
        actuator : string
        event : tuple
        target_sensor : string
        training_features : list
            the sensor used to predict the targer_sensor values
        model : None | linear_model.Lasso
            the fitted model
        R_event : ndarray
            the array of coefficient of the fitted model, it is sparse given Lasso model
        epsilon : int
            threshold used to define if the given object is a trigger for the event
        trigger : bool
        """
        self.actuator = actuator
        self.event = event
        self.target_sensor = sensor
        self.training_features = None
        self.model = None
        self.R_event = None
        self.epsilon = 0.05
        self.trigger = True
        
    def to_string(self):
        print("-----------------------")
        print("Event Driven Predicates:")
        print(self.actuator)
        print("-----------------------")

    def fit_model(self, train_dataset, sensor_columns):
        """
        fit the lasso model for a given sensor given an event
        
        Parameters
        ----------
        train_dataset : pandas DataFrame
            dataset containing all the data related to the event considered
        sensor_columns : list
        """
        y = train_dataset[self.target_sensor]
        self.training_features = [
            col for col in sensor_columns if col not in self.target_sensor]
        x = train_dataset[self.training_features]

        reg = linear_model.Lasso(alpha=0.1, max_iter=10000)
        reg.fit(x, y)
        self.R_event = reg.coef_
        preds = reg.predict(x)
        error = math.sqrt(mean_squared_error(y, preds))
        if (np.sqrt(np.square(y.values - preds)) < len(y)*[self.epsilon]).all() and not(error == 0):
            self.model = reg
        else:
            self.trigger = False


class Predicates():
    """
    Predicates generation for the given dataset
    """
    def __init__(self, df_train_orig, sensor_columns, actuator_columns):
        """        
        Parameters
        ----------
        df_train_orig : pandas DataFrame
            data that are used to extract predicates
        sensor_columns : list
        actuator_columns : list
        """

        self.categorical_predicates = CategoricalPredicate(df_train_orig, actuator_columns)
        self.distribution_driven_predicates = self.distribution_driven_strategy(
            df_train_orig, sensor_columns)
        self.event_driven_predicates = self.event_driven_strategy(
            df_train_orig, sensor_columns, actuator_columns)
    
    def to_string(self):
        self.categorical_predicates.to_string()
        sum = 0
        for predicate in  self.distribution_driven_predicates:
            sum = sum + predicate.gmm.n_components
        print("Distribution Driven Predicates: ")
        print(sum)
        print("Event Driven Predicates: ")
        print(len(self.event_driven_predicates))
            

    def distribution_driven_strategy(self, df_train_orig, sensor_columns):
        """
        Extract Distribution Driven Predicates
        
        Parameters
        ----------
        df_train_orig : pandas DataFrame
        sensor_columns : list
        
        Returns
        -------
        list
            for every sensor the extracted GMM with minimum BIC score
        """
        print('Distribution Driven Strategy')

        def difference(x):
            return (x[1] - x[0])

        time_series = df_train_orig[sensor_columns].rolling(
            window=2).apply(difference, raw=True)
        return [DistributionDrivenPredicate(sensor, time_series[sensor]) for sensor in sensor_columns]

    def event_driven_strategy(self, df_train_orig, sensor_columns, actuator_columns):
        """
        Extract Event Driven Predicates
        
        Parameters
        ----------
        df_train_orig :  pandas DataFrame
        sensor_columns : list
        actuator_columns : [list
        Returns
        -------
        list
            trigger for events extracted for the dataset
        """
        print('Event Driven Strategy')
        # find events

        def difference(x):
            return (x[1] - x[0])
        events = df_train_orig[actuator_columns].rolling(
            window=2).apply(difference, raw=True)

        # remove rows without events
        events = events.loc[(events != 0).any(axis=1)]

        # track events and occurrences
        # for every event extract the indexes when it occurred,
        # to create the dataset for the specific event
        event_dict = {}
        for index, row in events[1:].iterrows():
            for actuator in actuator_columns:
                if not(row[actuator] == 0):
                    if not(actuator in event_dict.keys()):
                        event_dict[actuator] = {}
                    new_transition = tuple([
                        df_train_orig.at[index-1, actuator], df_train_orig.at[index, actuator]])
                    try:
                        event_dict[actuator][new_transition].append(index)
                    except KeyError:
                        event_dict[actuator][new_transition] = [index]
        event_objs = []
        for actuator in event_dict.keys():
            # print('----------------------')
            #print('\t' + actuator)
            # print('----------------------')
            for event in event_dict[actuator].keys():
                # print('--\t'+str(event)+'\t--')
                train_dataset = df_train_orig[sensor_columns].loc[event_dict[actuator][event]]
                # TODO this dataset may contain only few samples
                for sensor in sensor_columns:
                    event_obj = EventDrivenPredicate(actuator, event, sensor)
                    event_obj.fit_model(train_dataset, sensor_columns)
                    if event_obj.trigger:
                        event_objs.append(event_obj)
                       
                        break  #to create only a trigger for a specified event
        return event_objs


if __name__ == "__main__":
    data_path = "../Spoofing Framework/"
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
    predicates.to_string()