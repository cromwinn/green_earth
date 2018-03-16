import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.dates as mdates
import matplotlib.cbook as cbook

from matplotlib.ticker import StrMethodFormatter

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error

import os
import folium
from folium.plugins import HeatMapWithTime, HeatMap


def simp_temps(df, simp=True):
    ''' If simp, returns dataframe with two columns: Timestamp and Temperature.
        Timestamp is cleaned of NaT, temperature of -9999 values,
        and the dataframe is sorted by timestamp. Else returns the full dataset.'''

    df['timestamp'] = pd.to_datetime(df[['year','month','day']], errors='coerce')
    if simp:
        temp = df[['timestamp','temperature']]
    else:
        temp = df
    temp = temp.sort_values(by='timestamp')
    temp = temp[temp['temperature'] != -9999]
    out = []
    for i in range(len(temp)):
        if type(temp.iloc[i]['timestamp']) == pd._libs.tslib.NaTType:
            out.append(i)
    temp.drop(temp.index[out])
    return temp

def temp_timely_means(temp):
    ''' Takes the simple temps output and returns daily, monthly and yearly
    means, starting from first of the month and year.  '''

    temp_dmeans = temp.groupby(temp['timestamp']).mean()
    temp_dmeans = temp_dmeans.reset_index()

    temp_m = pd.DataFrame()
    temp_m['temperature'] = temp['temperature']
    temp_m['timestamp'] = temp['timestamp'].apply(lambda x: x.replace(day=1))
    temp_mmeans = pd.DataFrame()
    temp_mmeans = temp_m.groupby(temp_m['timestamp']).mean().reset_index()

    temp_y = pd.DataFrame()
    temp_y['temperature'] = temp['temperature']
    temp_y['timestamp'] = temp['timestamp'].apply(lambda x: x.replace(day=1).replace(month=1))
    temp_ymeans = pd.DataFrame()
    temp_ymeans = temp_y.groupby(temp_y['timestamp']).mean().reset_index()

    return temp_dmeans, temp_mmeans, temp_ymeans

def plot_all_temperatures(temp, temp_dmeans, temp_mmeans):
    ''' Plots all temperature datapoints, compares against daily and monthly
        averages. '''

    fig, ax = plt.subplots(24,1, figsize=(12,140))

    for i, y in enumerate(np.arange(1985,2009)):
        ax[i].plot_date(temp['timestamp'], temp['temperature'], alpha=.1, label='All Temperature Data')
        ax[i].plot(temp_dmeans['timestamp'], temp_dmeans['temperature'], color='orange', label='Daily Averages')
        ax[i].plot(temp_mmeans['timestamp'], temp_mmeans['temperature'], linewidth=3, color='cyan', label='Monthly Averages')
        ax[i].set_xlim(pd.Timestamp(f'{y}-01-01'), pd.Timestamp(f'{y+1}-01-01'))
        ax[i].grid(True)
        ax[i].set_title(y)
        ax[i].set_ylabel('Global Mean Ocean Temperature in Celsius')
        ax[i].set_xlabel('Month')
        ax[i].legend()
        ax[i].yaxis.set_major_formatter(StrMethodFormatter('{x}°C'))

    plt.show()

def summary_plot(temp_ymeans, temp_mmeans):
    ''' Plots monthly and yearly averages against each other '''

    fig, ax = plt.subplots(1,1, figsize=(12,8))

    ax.plot(temp_ymeans['timestamp'], temp_ymeans['temperature'], color='red', linewidth=4, label='Yearly temp means')
    ax.grid(True)
    ax.set_title('Global Yearly Average Ocean Temp')

    ax.plot(temp_mmeans['timestamp'], temp_mmeans['temperature'], color='cyan', alpha=0.3, label='Monthly temp means')
    ax.grid(True)
        ax.set_title('Global Monthly Average Ocean Temp')
    ax.set_ylabel('Global Mean Ocean Temperature in Celsius')
    ax.set_xlabel('Year')
    ax.legend()
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x}°'))

    plt.show()


def one_train_test(df):
    ''' Makes a Linear Regression model, plots predicted temps against DOC '''

    X = df.drop(['temperature', 'timestamp'], axis=1)
    y = df.temperature

    X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1 )

    model = LinearRegression()
    model.fit(X_train,y_train)
    X_test_predicted = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(12,8))

    ax.scatter(X_test['doc'], y_test, color='blue', alpha=.33, label='actual')
    ax.scatter(X_test['doc'], X_test_predicted, color='orange', alpha=.1, label='predicted')
    ax.set_xlim((min(X_test[X_test['doc'] > -100]['doc'])-10), 300)
    ax.legend()
    ax.grid(True)
    ax.set_title('Predicted Temperatures Against DOC compared with Actual')
    ax.set_ylabel('Predicted Mean Ocean Temperature in Celsius')
    ax.set_xlabel('Dissolved Organic Carbon in ppt (DOC)')
    ax.legend()
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x}°'))

    plt.show()

def KFolds_plot(df, n_splits=10):
    ''' Plots n_splits graphs of predictions based on KFold test validations '''

    X = df.drop(['temperature', 'timestamp'], axis=1)
    y = df.temperature

    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    fig, ax = plt.subplots(10,1,figsize=(12,80))


    for (i, (train_index,test_index)) in enumerate(kf.split(X)):

        model = LinearRegression()
        model.fit(X_train,y_train)
        X_test_predicted = model.predict(X_test)


        ax[i].scatter(X_test['doc'], y_test, color='blue', alpha=.33, label='actual')
        ax[i].scatter(X_test['doc'], X_test_predicted, color='orange', alpha=.1, label='predicted')
        ax[i].set_xlim((min(X_test[X_test['doc'] > -100]['doc'])-10), 300)
        ax[i].legend()
        ax[i].grid(True)
        ax[i].set_title('Predicted Temperatures Against DOC compared with Actual')
        ax[i].set_ylabel('Predicted Mean Ocean Temperature in Celsius')
        ax[i].set_xlabel('Dissolved Organic Carbon in ppt (DOC)')
        ax[i].legend()
        ax[i].yaxis.set_major_formatter(StrMethodFormatter('{x}°'))

    plt.show()

def base_year_isolator(df, year, plot=False, model=None):
    ''' Isolates year as test, runs regression, plots predicted against salinity if plot '''

    X = df.drop(['temperature', 'timestamp'], axis=1)
    y = df.temperature

    X_train = X[X['year'] != year]
    y_train = df[df['year'] != year]['temperature']
    X_test = X[X['year'] == year]
    y_test = df[df['year'] == year]['temperature']

    if model = None:
        model = LinearRegression()

    model.fit(X_train,y_train)
    y_test_predicted = model.predict(X_test)

    if plot:
        fig, ax = plt.subplots(figsize=(12,8))

        ax.scatter(X_test['salinity'], y_test, color='blue', alpha=.33, label='actual')
        ax.scatter(X_test['salinity'], y_test_predicted, color='orange', alpha=.1, label='predicted')
        ax.set_xlim(30,40)
        ax.legend()
        ax.grid(True)
        ax.set_title('Predicted Temperatures Against Salinity compared with Actual')
        ax.set_ylabel('Predicted Mean Ocean Temperature in Celsius')
        ax.set_xlabel('Salinity in ppt')
        ax.legend()
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x}°'))

        plt.show()

    return model, round(mean_squared_error(y_test, y_test_predicted), 3)

def many_year_isolator(df):
    ''' Produces a list of models and their MSEs, returns the best one. '''

    models = []
    for y in range(1973, 2013):
        try:
            model, mse = base_year_isolator(df, y)
        except ValueError:
            print("year not working is: " + str(y)
    m = np.array(models)
    m = m[m[:,1].argsort()]

    print("Best performing model MSE: " + str(m[0,1]))
    final_model = m[0,0]

    return final_model


def plot_rolling_avgs(df):
    ''' Plots monthly and yearly rolling average global oceanic
        temperatures over full period '''

    df['timestamp'] = pd.to_datetime(df[['year','month','day']], errors='coerce')
    station_temp_means = df[df['temperature'] != -9999][['station', 'timestamp', 'temperature']].groupby(['timestamp', 'station' ]).mean()

    monthly_rolling_avgs = pd.DataFrame(columns=['Rolling Temp', 'timestamp'])

    for i in range(15070):

        monthly_rolling_avgs = monthly_rolling_avgs.append( { 'Rolling Temp' :  station_temp_means[
            pd.Timestamp('1972-07-24') + pd.DateOffset(days=30+i):
            pd.Timestamp('1972-07-24') + pd.DateOffset(days=60+i) ].mean().values[0],
                'timestamp' :  pd.Timestamp('1972-07-24') + pd.DateOffset(days=30+i)
                                   },  ignore_index = True )

    yearly_rolling_avgs = pd.DataFrame(columns=['Rolling Temp', 'timestamp'])

    for i in range(15070):

        yearly_rolling_avgs = yearly_rolling_avgs.append( { 'Rolling Temp' :  station_temp_means[
            pd.Timestamp('1972-07-24') + pd.DateOffset(days=365+i):
            pd.Timestamp('1972-07-24') + pd.DateOffset(days=730+i) ].mean().values[0],
                'timestamp' :  pd.Timestamp('1972-07-24') + pd.DateOffset(days=365+i)
                                   },  ignore_index = True )

    fig, ax = plt.subplots(1,1, figsize=(12,8))

    ax.plot(monthly_rolling_avgs['timestamp'], monthly_rolling_avgs['Rolling Temp'], color='cyan', alpha=0.33, label='Monthly temp means')
    ax.plot(yearly_rolling_avgs['timestamp'], yearly_rolling_avgs['Rolling Temp'], color='red', linewidth=4, label='Yearly temp means')
    ax.grid(True)
    ax.set_title('Global Rolling Monthly/Yearly Average Ocean Temp')

    ax.set_ylabel('Global Mean Ocean Temperature in Celsius')
    ax.set_xlabel('Year')
    ax.legend()
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x}°'))

    plt.show()
