import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import BayesianOptimization

from midi_conversion import pq2arr



def train_test_split(split=1):
    '''
    Loading and splitting the data into a train and test part.
    Split defines how many tracks are contained in the test set
    '''
    arr_tracks, arr_track_lengths=pq2arr('1024x128',(1024,128))
    #arr_tracks=arr_tracks[:2,:,:] # testing
    arr_tracks_train=arr_tracks[:-split,:,:]
    arr_tracks_test=arr_tracks[-split:,:,:]
    return arr_tracks_train, arr_tracks_test



def produce_timeseries(arr_tracks,tracks_indexes=None): 
    '''
    Given arr_tracks, each individual track is aggregated into 
    further subsequences that, with a lag of n_timesteps and a
    lead of 1
    '''
    
    if tracks_indexes:
        arr_tracks=arr_tracks[tracks_indexes,:,:].reshape(1,-1,n_features)

    arr_X=np.empty(shape=(0,n_timesteps,n_features))
    arr_y=np.empty(shape=(0,n_features))   
    
    for i in range(arr_tracks.shape[0]):
        arr=arr_tracks[i,:,:]
        timeseries = timeseries_generation(arr, n_lag=n_timesteps, n_lead=1).values
        X=timeseries[:,:-n_features]
        y=timeseries[:,-n_features:]
        X = X.reshape((X.shape[0], n_timesteps, n_features))
        arr_X = np.append(arr_X, X, axis=0)
        arr_y = np.append(arr_y, y, axis=0)
    return arr_X, arr_y


def timeseries_generation(data, n_lag=1, n_lead=1, dropnan=True):
    '''
    Converts timeseries in dataset with n_lag timesteps for each sample.
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    
	# input sequence (t-n, ... t-1)
    for i in range(n_lag, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        
	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_lead):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            
	# put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg




def save_train_test(X_train, y_train, X_test, y_test): 
    '''
    Function for saving train and test arrays in parquet format
    '''
    X_train=X_train.reshape(-1, n_features)
    df_X_train=pd.DataFrame(X_train, dtype='float64')
    df_X_train.columns=[str(col) for col in df_X_train.columns]
    df_X_train.to_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','X_train'), compression='gzip')
    
    X_test=X_test.reshape(-1, n_features)
    df_X_test=pd.DataFrame(X_test, dtype='float64')
    df_X_test.columns=[str(col) for col in df_X_test.columns]
    df_X_test.to_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','X_test'), compression='gzip')
    
    df_y_train=pd.DataFrame(y_train, dtype='float64')
    df_y_train.columns=[str(col) for col in df_y_train.columns]
    df_y_train.to_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','y_train'), compression='gzip')
    
    df_y_test=pd.DataFrame(y_test, dtype='float64')
    df_y_test.columns=[str(col) for col in df_y_test.columns]
    df_y_test.to_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','y_test'), compression='gzip')
 
  
def load_train_test():  
    '''
    Function for loading train and test data
    '''
    X_train=pd.read_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','X_train')).values
    X_train=X_train.reshape(-1,n_timesteps, n_features)
    
    X_test=pd.read_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','X_test')).values
    X_test=X_test.reshape(-1,n_timesteps, n_features)
    
    y_train=pd.read_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','y_train')).values
    y_test=pd.read_parquet('data/train_test/{}/{}.parquet.gzip'.format('1024x128','y_test')).values
    return X_train, y_train, X_test, y_test



def train_model(X_train,y_train):
    '''
    In train_model a LSTM network is initiated and trained
    '''
   
    mc = ModelCheckpoint(
        'best_model.h5',
        monitor='loss',
        mode='min',
        save_best_only=True)
    
    callbacks=[mc]
    
    n_neurons=1024

    # Building of the neural network
    model = Sequential()
    model.add(LSTM(n_neurons, 
                   input_shape=(n_timesteps, n_features),
                   return_sequences=True,
                   name='lstm_1'))
    
    model.add(Dropout(0.2, name='dropout_1'))
    
    model.add(LSTM(n_neurons,
                   input_shape=(n_timesteps, n_features),
                   name='lstm_2'))
    
    model.add(Dense(n_features,activation='relu', name='dense_1'))
    
    model.compile(loss='mae', optimizer=Adam(learning_rate=0.001))
       
    
    # fit network
    _start=time()
    
    history = model.fit(
        X_train, 
        y_train,
        #validation_split=0.2,
        epochs=n_epoch,
        batch_size=n_batch,
        callbacks=callbacks,
        verbose=2,
        shuffle=False)
    
    # fit network with generator as input
    '''
    history = model.fit(
        x=data_generator(arr_tracks,n_epoch),
        epochs=n_epoch,
        steps_per_epoch=n_batch,
        callbacks=callbacks,
        verbose=2,
        shuffle=False)
    '''
    
    _stop=time()
    print(_stop-_start)
    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()   
    return model


def build_hp_model(hp):
    '''
    build_hp_model is a helper function for building a neural
    network that is the input of a hyperparamter optimization instanc
    '''
    # Number of layers that should be tested
    hp_layers = hp.Int(
        'n_layers', 
        min_value=1,
        max_value=3,
        step=1,
        )
    
    # number of neurons that should be tested
    hp_neurons = hp.Int(
        'n_units', 
        min_value=128,
        max_value=1024,
        default=256,
        step=128,
        )
      

    model = Sequential()    
    for i in range(hp_layers):
        model.add(
            LSTM(
                # This step provides the possibility of having
                # a different amount of neurons per layer
                hp.Int(
                    'units_layer_{}'.format(i),
                    min_value=128,
                    max_value=1024,
                    step=128
                    ),
                input_shape=(n_timesteps, n_features),
                return_sequences=True,
                name='lstm_{}'.format(i)))
        model.add(Dropout(0.2, name='dropout_{}'.format(i)))
        
    model.add(LSTM(hp_neurons,
                   input_shape=(n_timesteps, n_features),
                   name='lstm_default'))
    
    model.add(Dense(units=n_features, activation='relu',name='dense_default'))
    model.compile(loss='mae', optimizer='adam')
    print(model.summary())
    return model




def hyperparam_search(X_train,y_train):
    '''
    Performing of a hyperparameter search using Bayesian optimization
    and saving of the best performing model
    '''

    tuner = BayesianOptimization(
        build_hp_model,
        objective='loss',
        max_trials=100,                     # how many combinations of parameters should be tried
        executions_per_trial=1,             # how often a certain combination is run
        directory=os.path.normpath('C:/keras_tuning'),
        project_name='lstm_tuning_configs',
        overwrite=True,
        seed=seed
        )
    
    # works same as model.fit()
    tuner.search(
        X_train, y_train,
        epochs=n_epoch,
        batch_size=n_batch,
        verbose=1
        )
    
    
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('best_model.h5')
    best_params = tuner.get_best_hyperparameters(1)[0]
    
    print(best_model.summary())
    print()
    print(best_params.values)
    
    return best_model, best_params




# adapted from https://sailajakarra.medium.com/lstm-for-time-series-predictions-cc68cc11ce4f
def forecast(model, forecast_start_points, n_preds):
    '''
    Forecasting a timeseries given some starting points
    '''
    n_batch=1
    y_pred=np.zeros(shape=(n_preds,n_features))
    first_eval_batch = forecast_start_points
    current_batch = first_eval_batch.reshape((n_batch, n_timesteps, n_features))
    for i in range(n_preds):
        # [0] flattens second dimension with length 1
        pred=model.predict(current_batch)[0]
        y_pred[i,:]=pred
        # remove first element of batch and add current prediction
        current_batch = np.append(current_batch[:,1:,:],[[pred]],axis=1)     
    return y_pred



def pad_train_test_pred_for_plots(y_train, y_test, y_pred):
    '''
    For stacking a predictions array with the array representing
    a song they have to be padded to have the same size
    '''
    n_train=len(y_train)
    # shift train predictions for plotting
    arr_train=np.zeros(shape=(n_train+n_preds,n_features))
    arr_train[:, :] = np.nan
    arr_train[0:n_train, :] = y_train
    arr_train[n_train:n_train+len(y_test),:]=y_test
       

    arr_pred=np.zeros(shape=(n_train+n_preds,n_features))
    arr_pred[:, :] = np.nan
    arr_pred[n_train:, :] = y_pred
    
    return arr_train, arr_pred



def plot_preds(arr_train, arr_pred):
    '''
    plot_preds stacks arr_train and arr_pred by masking all 
    values in arr_pred that are 0
    '''
    arr_train=arr_train.transpose()
    arr_pred=arr_pred.transpose()
    cmap = colors.ListedColormap(['red'])
    arr_pred=np.ma.masked_where(arr_pred == 0, arr_pred)
    fig, ax = plt.subplots()      
    ax.imshow(arr_train,aspect='auto',origin='lower')
    ax.imshow(arr_pred,aspect='auto',origin='lower',cmap=cmap,interpolation='none')
    plt.savefig('lstm_preds.pdf')



def data_generator(arr_tracks,n_epoch):
    '''
    Generator function yielding a batch each time it is called.
    '''
    for epoch in range(n_epoch):
        for i in range(arr_tracks.shape[0]):
            arr=arr_tracks[i,:,:]
            timeseries = timeseries_generation(arr, n_lag=n_timesteps, n_lead=1).values
            X=timeseries[:,:-n_features]
            y=timeseries[:,-n_features:]
            X = X.reshape((X.shape[0], n_timesteps, n_features))
            yield (X, y)




if __name__=='__main__':
    
    # Defining some global variables
    seed=42
    n_timesteps=256 
    n_features=128 
    n_batch=768
    n_epoch=500
    
    # The trainings data comprises 290 tracks and the test data 5
    arr_tracks_train, arr_tracks_test=train_test_split(split=5)
    
    # The dataset is used to generate timeseries data with size 
    # of each sample (n_timesteps,n_features)
    # For testing purposes only one track is used for training the model
    X_train, y_train=produce_timeseries(arr_tracks_train,tracks_indexes=[68])
    X_test, y_test=produce_timeseries(arr_tracks_test)
    
    
    save_train_test(X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test=load_train_test()
    
    # To get an idea of a good model architecture a hyperameter search is started
    # best_model, best_params=hyperparam_search(X_train,y_train)
    # model = load_model('models/best_model.h5')
    
    # To train the model on generated data train_model function is called
    model=train_model(X_train, y_train)
    # If trainining with data generator is enabled
    # model=train_model(arr_tracks_train)
    model.save('lstm_model.h5')

    # Extending of a given song is initiated by a given starting point primer
    # This test uses a point of the training set to confirm that the model converged
    primer=X_train[-n_timesteps]
    n_preds=n_timesteps
    
    # The model forecasts data starting with primer for n_preds timesteps
    y_pred=forecast(model, primer, n_preds)
    
    # In this test y_test is the last part of y_train
    y_test=y_train[-n_timesteps:,:]
    y_train=y_train[:-n_timesteps,:]
    
    # Before plotting the arrays they need to be padded to have the same dimensions
    arr_train_padded, arr_pred_padded=pad_train_test_pred_for_plots(y_train,y_test,y_pred)
    plot_preds(arr_train_padded, arr_pred_padded)
    
    # By calculating the mae we quantify the performance of the model
    mae=mean_absolute_error(y_test, y_pred)
    
