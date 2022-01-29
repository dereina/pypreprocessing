import units.unit as unit
#import units.jit_optimizations.energy_equalizer as energy_equalizer
import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report,confusion_matrix
import os
import pickle
#import psyco
#psyco.full()

class Unit(unit.Unit):
    def __init__ (self, context, unit_config= None, input_map_id_list = None): #the data schema of the input map is checked by this Unit
        unit.Unit.__init__(self, context, unit_config, input_map_id_list)
        print("__init__ Predictor")
        #load data from sources, read configuration file
        #an empty list means no inputs... the default value is provided...
        self.df_list = self.getInputOrDefault(type(pandas.DataFrame()), []) #inputs are always lists, should be a struct with meta data from the sender...
        self.load_predictor = self.getConfigOrDefault("load_predictor", False)
        self.hidden_layer_sizes = self.getConfigOrDefault('hidden_layer_size', [144,])
        self.activation=self.getConfigOrDefault('activation', 'tanh')
        self.solver=self.getConfigOrDefault('solver','lbfgs')
        self.learning_rate=self.getConfigOrDefault('learning_rate','adaptive')
        self.max_iter=self.getConfigOrDefault('max_iter', 1000)
        self.max_fun=self.getConfigOrDefault('max_fun', 50000)
        self.drop_columns = self.getConfigOrDefault('drop_columns', [])
        self.target_column = self.getConfigOrDefault('target_column', 'category')

    def run(self):
        k=0
        split_test=False
        #predictors = ["n"+str(x) for x in range(self.affination // 2)]
        #predictors += ["height", "width", 'mode', 'median', 'std', 'direct', 'inverse', 'min', 'max', 'pmode', 'pmedian', 'pstd', 'pdirect', 'pinverse', 'pmin', 'pmax']
        print("run predictor ")
        predictor = None
        if self.load_predictor:
                out_path = self.getPathOutputForFile("predictor0.pkcls") # mal.....
                file_present = os.path.isfile(out_path)
                if file_present:
                    infile = open(out_path,'rb')
                    predictor = pickle.load(infile)
                    infile.close() 
                    
        if predictor is None:    
            if self.df_list is not None and len(self.df_list) > 0:
                for i in range(len(self.df_list)):
                    df = self.df_list[i]
                    if self.target_column not in df.columns:
                        continue
                    
                    drop_columns = []
                    for column in self.drop_columns:
                        if column in df.columns:
                            drop_columns.append(column)
                    
                    if len(drop_columns) > 0:
                        df = df.drop(columns = drop_columns)

                    #df = df.drop(df.index[[0,1]]) #drop first lines of the csv... you could check lines with non numerical values to drop... also for columns...
                    #predictors = list(set(list(df.columns))-set(drop_columns))
                    #print(df[predictors].head())

                    y = df[self.target_column].values
                    df = df.drop(columns = [self.target_column])
                    predictors = df.columns
                    print("predictors")
                    print(predictors)  
                    #dlk
                    X = df[predictors].values
                    print(y)
                    print(X)
                    #print(X)
                    #print("jou")
                    #print(y)
                    #df[predictors] = df[predictors]/df[predictors].max() # normalization
                    #print(entry)
                    #print(df.shape)
                    #print(df.transpose())
                    #test_size = 0.20
                    #X_train, X_test, y_train, y_test = None, None, None, None
                    #if not split_test:
                    #    X_train, X_test, y_train, y_test = X,X,y,y
                    #else:
                    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=3141592)
        
                    #if not split_test:
                    #    X_test = X_train
                    #    y_test = y_train
        
                    filename = self.getPathOutputForFile("predictor"+str(i)+'.pkcls')
        
        
                    #print(X_train.shape); 
                    #print(y_train.ravel().shape)
                    mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation, solver=self.solver, learning_rate=self.learning_rate, max_iter=self.max_iter, max_fun=self.max_fun)
                    mlp.fit(X,y.ravel())
        
                    with open(filename,'wb') as outfile:
                        pickle.dump(mlp,outfile)
                    outfile.close()
        
        
                    infile = open(filename,'rb')
                    predictor = pickle.load(infile)
                    infile.close()
        
                    predict_train = predictor.predict(X)
                    #predict_test = mlp.predict(X_test)
                    print('TRAIN')
                    print(confusion_matrix(y,predict_train))
                    print(classification_report(y, predict_train,  zero_division="warn"))
                    #print('TEST')
                    #print(confusion_matrix(y_test,predict_test))
                    #print(classification_report(y_test, predict_test, zero_division="warn"))
                    print(X) 
                    return True

        if predictor is not None:
            #do a test with the embedded input data
            #predictor doing test 
            if self.df_list is not None and len(self.df_list) > 0:
                for i in range(len(self.df_list)):
                    df = self.df_list[i]
                    if self.target_column not in df.columns:
                        continue
                    
                    drop_columns = []
                    for column in self.drop_columns:
                        if column in df.columns:
                            drop_columns.append(column)
                    
                    if len(drop_columns) > 0:
                        df = df.drop(columns = drop_columns)

                    #df = df.drop(df.index[[0,1]]) #drop first lines of the csv... you could check lines with non numerical values to drop... also for columns...
                    #predictors = list(set(list(df.columns))-set(drop_columns))
                    #print(df[predictors].head())

                    y = df[self.target_column].values
                    df = df.drop(columns = [self.target_column])
                    predictors = df.columns
                    print("predictors")
                    print(predictors)  
                    #dlk
                    X = df[predictors].values
                    print(y)
                    print(X)
        
                    predict_test = predictor.predict(X)
                    #predict_test = mlp.predict(X_test)
                    print('TEST')
                    print(confusion_matrix(y,predict_test))
                    print(classification_report(y, predict_test,  zero_division="warn"))
                    #print('TEST')
                    #print(confusion_matrix(y_test,predict_test))
                    #print(classification_report(y_test, predict_test, zero_division="warn"))
                    print(X) 
                    return True

        return False