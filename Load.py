from nilmtk import DataSet
import matplotlib.pyplot as plt
import pandas as pd

'''
    This class has been created to reunite the datasets and to get a specific 
    signal in a easy way. The __path is needed to be setted according to the
    location of the .h5 files.

'''

class Appliance:
    __path = r"Database"
    def __init__(self, name, building, dataset):
        
        self._dataset = 0
        self._name = 0
        self._data = pd.DataFrame()
        self._building = building
        
        self.name = name
        
        self.dataset = dataset
        
    
    # Getters and setters      
    @property
    def dataset(self):
        return self._dataset     
    @dataset.setter
    def dataset(self, value):       
        if value.lower() == "redd".lower():
            temp = DataSet(Appliance.__path + r'\redd.h5')
        elif value.lower() == "synd".lower():
            temp = DataSet(Appliance.__path + r'\SynD.h5')
        else:
            raise Exception("Invalid dataset. Please insert a valid argument.")
        
        # If self.building is list (or a tuple), self.data will be a list of series, taking every instance.
        # If self.building is int, self.data will be a series, taking just one signal.
        if type(self.building) == int:
            try:
                # Verifying if it has more than one instance. If true, it takes the first one
                instance = temp.buildings[self.building].elec[self.name].instance()
                if type(instance) == int:
                    self.data = next(temp.buildings[self.building].elec[self.name].power_series())
                elif type(instance) == tuple:
                    self.data = next(temp.buildings[self.building].elec[self.name][instance[0]].power_series())
            except KeyError:
                raise KeyError("Please insert a valid appliance name or house number.")   
        
        elif type(self.building) == list or type(self.building) == tuple:
            self.data = []
            for h in range(len(self.building)):    
                try:
                    instance = temp.buildings[self.building[h]].elec[self.name].instance()
                    if type(instance) == int:
                        self.data.append(next(temp.buildings[self.building[h]].elec[self.name].power_series()))
                    elif type(instance) == tuple:
                        for i in range(len(instance)):
                            self.data.append(next(temp.buildings[self.building[h]].elec[self.name][instance[i]].power_series()))
                except KeyError:
                    raise KeyError("Please insert a valid appliance name or house number.")
        else:
            raise KeyError("Please inser a valid house number format.")
    
    @property
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value
    
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, value):
        self._data = value
    
    @property
    def building(self):
        return self._building
    @building.setter
    def building(self, value):
        self._building = value
    
    # Ploting the signal
    def plot(self):
        plt.plot(self.data)
    
    # Path setter
    @staticmethod
    def set_path(value):
        # It should be a String
        if type(value) is not str:                          
            raise ValueError("Please, insert the __path as a String.")
        else:
            import os.path
            # It should be a valid existing __path
            if os.path.isdir(value):
                Appliance.__path = value
            else:
                raise ValueError("Please, choose a valid __path.")
    
    # Path getter
    @staticmethod
    def show_path():
        return Appliance.__path
    
    def split(self, train_size=None, test_size=None):
        # This method splits the signal in train and test
        
        from sklearn.model_selection import train_test_split
        
        # Catching errors
        if train_size == None and test_size == None:
            raise ValueError ("Invalid values of train and/or test.")
        if train_size != None and test_size != None:
            if train_size + test_size != 1:
                raise ValueError ("Invalid values of train and/or test.")
        
        # Assign values
        if train_size == None:
            train_size = 1 - test_size
        if test_size == None:
            test_size = 1 - train_size
        
        train, test = train_test_split(self.data, test_size=test_size, train_size=train_size)
        return train, test
    
    
    
    