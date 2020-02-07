from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

register_matplotlib_converters()

#did not write this function my self got it off a web forum but eddited it a bit my self
def getDateRangeFromWeek(p_year,p_week): #returns the last day of the week (sunday as a date time var)
    firstdayofweek = datetime.datetime.strptime(f'{p_year}-W{int(p_week )- 1}-1', "%Y-W%W-%w").date()
    lastdayofweek = firstdayofweek + datetime.timedelta(days=6.9)
    return lastdayofweek

#wrote 80% of this my self
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg, best_result = float("inf"), None, pd.DataFrame
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:

                    model = ARIMA(dataset, order=order)
                    results = model.fit(disp=-1)

                    diff = df_log - df_log.shift()
                    diff = diff.drop(index=0)
                    predic = pd.Series.to_frame(results.fittedvalues)

                    mse = mean_squared_error(diff, predic)
                    if mse < best_score:
                        best_score, best_cfg, best_res = mse, order, predic
                    print('ARIMA%s MSE=%.10f' % (order, mse))

                except:
                    continue
    print('Best ARIMA%s MSE=%.10f' % (best_cfg, best_score))

    plt.plot(df_log - df_log.shift())
    plt.plot(best_res, color='red')
    plt.show()
    print("\n\n******\n this shows that the model has quite a good fit, but i cannot figure out how to get from this to predictions unfortualty\n\n******\n")

    return best_res

#did not write this my self
def get_stationarity(timeseries,type):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    title = str("Rolling Mean & Standard Deviation ("+str(type)+")")
    plt.title(title)
    plt.show()

    # Dickey–Fuller test:
    result = adfuller(timeseries)
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
    print("this shows that the data is not increasing over time")

#wrote this my slef
def data_input_clean(path):
    df = pd.read_csv(path, header=None, nrows=167) #imports csv

    df = df.T #transposes
    df.columns = ["year", "delivery amount"] #names colums
    df["delivery amount"] = df["delivery amount"].astype(int) #set data type
    df.insert(1, "week", np.nan) #inserts a new column
    df[["year", "week"]] = df[["year", "week"]].astype(str) #sets data type for the other 2 columns

    df["week"] = df["year"].str[4:] #splits the time variable into year and
    df["year"] = df["year"].str[:4] #week

    df[["year", "week"]] = df[["year", "week"]].astype(int) #sets as itigers so can be processed though the function


    for index, row in df.iterrows():#itterates though the rows inputting the numbers into the function
        df.at[index, "year"] = getDateRangeFromWeek(row["year"], row["week"])

    return df

#wrote 1/2 of this my self
def seasonal_analisys(df):
    # seasonal anaisys
    result = seasonal_decompose(df["delivery amount"], model="additive", period=1)
    result.plot()
    pyplot.show()
    ###this shows that there is no seasonailty###

    print(df)
    # looks from randomness in data
    window = 3  # smoothing
    df.insert(3, "ma", np.nan)
    df["ma"] = df.iloc[:, 2].rolling(window=window).mean()

    # plot auto corelation
    autocorrelation_plot(df.iloc[window - 1:, 3])
    pyplot.title("Auto Corelation plot")
    pyplot.show()

    return df

#did not write this my self
def EDA(df):
    print(
        "\n******\nlooking at ways to make the data set stationary\na high p value means that the data is not stationary(the ADF stat is less than the 5%CV)")

    print("\n\n******\nlog the data first")
    df_log = np.log(df)
    rolling_mean = df_log.rolling(window=12).mean()
    df_log_minus_mean = df_log - rolling_mean
    df_log_minus_mean.dropna(inplace=True)
    get_stationarity(df_log_minus_mean,"log the data")

    # exp decay
    print("\n\n******\nexp decay on data")
    rolling_mean_exp_decay = df_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
    df_log_exp_decay = df_log - rolling_mean_exp_decay
    df_log_exp_decay.dropna(inplace=True)
    get_stationarity(df_log_exp_decay,"exp on data")

    # shift
    print("\n\n******\nlog(1st) - log(2nd) etc...")
    df_log_shift = df_log - df_log.shift()
    df_log_shift.dropna(inplace=True)
    get_stationarity(df_log_shift,"log(1st) - log(2nd) etc...")
    print("\n\n******\n")

    return df_log_shift, df_log

#did not write my self
def fit_and_plot():
    p_values = range(0, 3)
    d_values = range(0, 3)
    q_values = range(0, 3)
    best_res = evaluate_models(df_log_shifted, p_values, d_values,q_values)  # find most accuate pyperamiters for ARIMA model

    plt.plot(df_log - df_log.shift())
    plt.plot(best_res, color='red')
    plt.show()
    print("\n\n******\n this shows that the model has quite a good fit, but i cannot figure out how to get from this to predictions unfortualty\n\n******\n")

df = data_input_clean("sample.csv")
df = seasonal_analisys(df)
df = df.drop(columns=["year","week","ma"]) #dopping unused data
df.columns = ["Count"] #tidying
df_log_shifted, df_log = EDA(df)
fit_and_plot()
