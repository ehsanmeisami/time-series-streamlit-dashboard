import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import pyplot
import pathlib
from pandas import DataFrame
from pandas import concat
from xgboost import XGBRegressor
from matplotlib import pyplot
import dateutil


st.write("""
# Demand Predicition App 
Forecasting 6 weeks ahead by 'Product Number ID' and 'Point of Sales'""")
st.write("---")

project_path = pathlib.Path(__file__).parent.absolute()

df = pd.read_csv(str(project_path) + '/demand_forecast_data.csv',index_col=0)
df['date'] = pd.to_datetime(df['date'])

prodName = sorted(list(df['ProductName_ID'].unique()))
stores = sorted(list((df['Point-of-Sale_ID'].unique())))

# selecting product number
select_productname = st.sidebar.selectbox("Select a Product Number ID", (prodName))
# selecting point of sales
select_pointofsale = st.sidebar.selectbox("Select a Point of Sales", (stores))
# selecting model
model_chosen = st.sidebar.selectbox("Choose a model",("SARIMA","XGBoost"))
# selecting the weeks to predict
#wks_to_predict = st.sidebar.slider("Weeks to predict",1,12,6)

st.write("Product Name Id chosen:", select_productname)
st.write("Store Id chosen:", select_pointofsale)
st.write("Model chosen:", model_chosen)

#df_selected_product = df.loc[(df['ProductName_ID'] == select_productname) & (df['Point-of-Sale_ID'] == select_pointofsale)]

# Call this function after pick the right(p,d,q) for SARIMA based on AIC               
def sarima_eva(y,order,seasonal_order,seasonal_period,pred_date,y_to_test):
    # fit the model 
    mod = sm.tsa.statespace.SARIMAX(y,
                                order=order,
                                seasonal_order=seasonal_order,
                                #enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    
    # The dynamic=False argument ensures that we produce one-step ahead forecasts, 
    # meaning that forecasts at each point are generated using the full history up to that point.
    pred = results.get_prediction(start=pd.to_datetime(pred_date), dynamic=False)
    pred_ci = pred.conf_int()
    pred_ci['lower Sell-out units'] = pred_ci['lower Sell-out units'].apply(lambda x : x if x > 0 else 0)

    y_forecasted = pred.predicted_mean
    
    y_forecasted = y_forecasted.to_frame()
    y_forecasted['predicted_mean'] = y_forecasted['predicted_mean'].apply(lambda x : x if x > 0 else 0)
    y_forecasted = y_forecasted.squeeze()
    
    return (results)


def forecast(model,predict_steps,y):
    
    pred_uc = model.get_forecast(steps=predict_steps)

    #SARIMAXResults.conf_int, can change alpha,the default alpha = .05 returns a 95% confidence interval.
    pred_ci = pred_uc.conf_int()    
    pred_ci['lower Sell-out units'] = pred_ci['lower Sell-out units'].apply(lambda x : x if x > 0 else 0)

    pred_uc.predicted_mean = pred_uc.predicted_mean.to_frame()
    pred_uc.predicted_mean['predicted_mean'] = pred_uc.predicted_mean['predicted_mean'].apply(lambda x : x if x > 0 else 0)
    pred_uc.predicted_mean = pred_uc.predicted_mean.squeeze()
    
    # Produce the forcasted tables 
    pm = pred_uc.predicted_mean.reset_index()
    pm.columns = ['Date','Predicted_Mean']
    pci = pred_ci.reset_index()
    pci.columns = ['Date','Lower Bound','Upper Bound']
    final_table = pm.join(pci.set_index('Date'), on='Date')
    
    return (final_table)


def run_sarima():

    df_selected_product = df.loc[(df['ProductName_ID'] == select_productname) & (df['Point-of-Sale_ID'] == select_pointofsale)]
    df_selected_product = df_selected_product.set_index('date')
    edf2 = df_selected_product['2016-01-01':'2019-11-03'].resample('W').agg({"Sell-out units":'sum'})
    y = edf2['Sell-out units']

    # last date in df
    x = int(len(y)*0.3)
    end = y.tail(x).index[0]

    #us the last 8 month to validate
    date_until = end - dateutil.relativedelta.relativedelta(months=8)

    y_to_train = y[:end] # dataset to train
    y_to_val = y[end:] # last X months for test  
    predict_date = len(y) - len(y[:end]) # the number of data points for the test set

    model = sarima_eva(y,(6, 1, 0),(6, 1, 0, 7),7,end,y_to_val)
    final_table = forecast(model,7,y)
    
    final_table.index = final_table.Date
    fig, ax = plt.subplots( figsize=(14, 7))
    x = final_table.index.strftime("%Y-%m-%d").tolist()
    y = final_table.Predicted_Mean
    pyplot.plot(x ,y , marker='o', linestyle='-', linewidth=0.5, label='Forecasted')
    pyplot.legend()
    st.pyplot(fig)




# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values

def run_xgboost():
    
    df_selected = df.loc[(df['ProductName_ID'] == select_productname) & (df['Point-of-Sale_ID'] == select_pointofsale)]
    df_selected = df_selected[['Sell-out units','date']].set_index('date')
    df_selected.index = pd.to_datetime(df_selected.index)
    start_date = df_selected.index.max()
    series = df_selected.copy()
    values = series.values
    # transform the time series data into supervised learning
    train = series_to_supervised(values, n_in=6)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror',
                        colsample_bytree = 0.5,
                        eta= 0.15,
                        learning_rate = 0.01,
                        max_depth = 3,
                        min_child_weight = 7,
                        subsample = 0.5,
                        n_estimators = 100
                        )
    model.fit(trainX, trainy)
    # construct an input for a new preduction
    row = values[-6:].flatten()
    
    # make a one-step prediction
    #yhat = model.predict(asarray([row]))
    
    dti = pd.date_range(start_date, periods=6, freq="W-SUN")
    dti_s = dti.to_series(index=row)
    dti_s = dti_s.reset_index()
    dti_s = dti_s.rename(columns={'index':'Sell-out units',0:'date'}).set_index('date')
    fig, ax = plt.subplots( figsize=(14, 7))
    x = dti_s.index.strftime("%Y-%m-%d").tolist()
    y = dti_s['Sell-out units']
    pyplot.plot(x ,y , marker='o', linestyle='-', linewidth=0.5, label='Forecasted')
    pyplot.legend()
    st.pyplot(fig)


# run the chosen model
if model_chosen == "XGBoost":
    #st.write("Model not available")
    run_xgboost()

elif model_chosen == "SARIMA":
    run_sarima()

else:
    st.write("This model is not availabe.")

