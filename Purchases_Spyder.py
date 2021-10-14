# -*- coding: utf-8 -*-

import pandas as pd
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6

#%% Year 2014
april2014 = pd.read_excel('purchasecardtransactionsapril2014.xls', sheet_name='APR 2014')
may2014 = pd.read_excel('purchasecardtransactionsmay2014.xls', sheet_name='Sheet1')
june2014 = pd.read_excel('purchasecardtransactionsjune2014.xls', sheet_name='Sheet1')
july2014 = pd.read_excel('purchasecardtransactionsjuly2014.xls',sheet_name='JULY 2014')
august2014 = pd.read_excel('purchasecardtransactionsaugust2014.xls',sheet_name='AUG 2014')
september2014 = pd.read_excel('purchasecardtransactionsseptember2014.xls',sheet_name='SEPT 2014')
october2014 = pd.read_excel('purchasecardtransactionsoctober2014.xls',sheet_name='OCT 2014')
november2014 = pd.read_excel('purchasecardsnov2014.xls',sheet_name='Sheet1')
december2014 = pd.read_excel('purchasecardsdec2014.xls',sheet_name='Sheet1')

frames2014 = [april2014,may2014,june2014,july2014,august2014,september2014,
              october2014,november2014,december2014]
year2014 = pd.concat(frames2014)
print(year2014)

#%% Year 2015
january2015 = pd.read_excel('purchaseccardtransactionsjanuary2015.xls',sheet_name='Sheet1')
january2015 = january2015.rename(columns={'Directorate ':'Directorate'})
february2015 = pd.read_excel('publishspendpurchasecardsfebruary2015.xls',sheet_name='Sheet1')
february2015 = february2015.rename(columns={'Unnamed: 10':'Directorate'})
march2015 = pd.read_excel('publishspendpurchasecardsmarch2015.xls',sheet_name='Sheet1')
may2015 = pd.read_excel('svlrdclr05homesharechexefinainmngeneralappublishspendmay2015.xls',sheet_name='Sheet1')
june2015 = pd.read_excel('publishspendjune2015alldirectorates.xls',sheet_name='Sheet1')
july2015 = pd.read_excel('itemisedtransactionsjuly2015publishspend.xls',sheet_name='Sheet1')
august2015 = pd.read_excel('itemisedtransactionsaugust2015publishspendalldirectorates.xls',sheet_name='Sheet1')
august2015 = august2015.rename(columns={'Directorate ':'Directorate'})
september2015 = pd.read_excel('publishspendseptember2015.xls',sheet_name='Sheet1')
october2015 = pd.read_excel('publishspendoctober2015.xls',sheet_name='Sheet1')
november2015 = pd.read_excel('publishspendnovember2015.xls',sheet_name='Sheet1')
december2015 = pd.read_excel('publishedspenddecember2015.xls',sheet_name='Sheet1')

frames2015 = [january2015,february2015,march2015,may2015,june2015,july2015,
              august2015,september2015,october2015,november2015,december2015]
year2015 = pd.concat(frames2015)
print(year2015)

#%% Year 2016
january2016 = pd.read_excel('publishspendjanuary2016.xls',sheet_name='Sheet1')
february2016 = pd.read_excel('publishspendfebruary2016.xls',sheet_name='Sheet1')
march2016 = pd.read_excel('publishspendmarch2016.xls',sheet_name='Sheet1')
april2016 =pd.read_excel('publish-spend-april-2016.xls',sheet_name='Sheet1')
may2016 = pd.read_excel('publish-spend-may-2016.xls',sheet_name='Sheet1')
june2016 = pd.read_excel('publish-spend-june-2016.xls',sheet_name='Sheet1')
july2016 = pd.read_excel('publish-spend-july-2016.xls',sheet_name='Sheet1')
august2016 = pd.read_excel('publish-spend-august-2016.xls',sheet_name='Sheet1')
september2016 = pd.read_excel('publish-spend-september-2016.xls',sheet_name='Sheet1')
september2016 = september2016.rename(columns={'Directorates':'Directorate'})
october2016 = pd.read_excel('publish-spend-october-2016.xls',sheet_name='Sheet1')
november2016 = pd.read_excel('publishing-spend-november-2016.xls',sheet_name='Sheet1')
december2016 = pd.read_excel('publish-spend-december-2016.xls',sheet_name='Sheet1')

frames2016 = [january2016,february2016,march2016,april2016,may2016,june2016,
              july2016,august2016,september2016,october2016,november2016,december2016]
year2016 = pd.concat(frames2016)
print(year2016)

#%% Year 2017
january2017 = pd.read_excel('cusersfinainmndesktoppublish-copy-january-2017.xls',sheet_name='Sheet1')
february2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-february-2017-all-directorates.xls',sheet_name='Sheet1')
march2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-march-2017.xls',sheet_name='Sheet1')
april2017 =pd.read_excel('cusersfinainmndesktoppublish-spend-april-2017.xls',sheet_name='Sheet1')
may2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-may-2017.xls',sheet_name='Sheet1')
june2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-june--2017.xls',sheet_name='Sheet1')
july2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-july-2017.xls',sheet_name='Sheet1')
august2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-august-2017-all-directorates.xls',sheet_name='Sheet1')
september2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-sept-2017.xls',sheet_name='Sheet1')
#september2017 = september2016.rename(columns={'Directorates':'Directorate'})
october2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-october-2017-all-directorates.xls',sheet_name='Sheet1')
november2017 = pd.read_excel('cusersfinainmndesktoppublish-spend-november-2017-all-directorates.xls',sheet_name='Sheet1')

frames2017 = [january2017,february2017,march2017,april2017,may2017,june2017,
              july2017,august2017,september2017,october2017,november2017]
year2017 = pd.concat(frames2017)
print(year2017)
list(year2017.columns)

#%% Year 2018
year2018 = pd.read_excel('cusersfinainmndesktoppublish-spend-january-2018.xls',sheet_name='Sheet1')
print(year2018)
list(year2018.columns)

#%% Observations and dropping
## We notice a difference between the number of columns:
    ##The year 2014 has 12 columns: the same 11 columns of the years 15 and 16
    ##plus a column named 'Billing CUR Code' which has only one value (GBP).
    ##This value is not pertinent to the analysis so it must be removed.
    ##After the removal all years will have the same number of columns 
    ##providing the complete dataset from the time window specified above.
    
year2014 = year2014.drop(columns='BILLING CUR CODE')
year2017 = year2017.drop(columns={'ORIGINAL CUR','BILLING GROSS AMT'
                                  ,'BILLING CUR CODE','TRANS TAX AMT'
                                  ,'BILLING CUR CODE.1','Directorates'})
year2018 = year2018.drop(columns={'ORIGINAL CUR','BILLING GROSS AMT'
                                  ,'BILLING CUR CODE'})
#Changing Year 2017 to date format
year2017['TRANS DATE'] = pd.to_datetime(year2017['TRANS DATE'])

    #Concatenating years:
data = pd.concat([year2014,year2015,year2016,year2017,year2018])
print(data)

#%% Exploring and cleansing the dataset
print(data.dtypes)
    
#Let's see the quantity of NULLS in every variable. 
print(data.isnull().sum())
print(data.isnull().sum() * 100/ len(data.index))
    #The quantity of NULLS is less than 5%, making possible 
    #the removal of nulls without impacting the mean of each variable.

#The NULL value for **TRANS DATE
trans_date_null = data[data['TRANS DATE'].isnull()]
print(trans_date_null.head()) #This row is completely in blanc so must be removed.
data = data[data['TRANS DATE'].notnull()]
print(data.isnull().sum())


#The unique values for **TRANS VAT DESC
list(data['TRANS VAT DESC'].unique())
    # ['VR', 'VZ', '6.65%', 'VL', 'VT', nan, 'VE', 'VS']
    # Due to the nature of the variable (object) '6.65%' nor nan values are allowed.
data = data[data['TRANS VAT DESC']!='6.65%']
data = data[data['TRANS VAT DESC'].notnull()]
print(data.isnull().sum())

#The unique values for **TRANS CAC CODE 1
    #This variable is also an object and NULL values must be errased.
data.groupby('TRANS CAC CODE 1')['TRANS DATE'].nunique()
data = data[data['TRANS CAC CODE 1'].notnull()]
    # The variable TRANS CAC DESC 1 and TRANS CAC CODE 1 had the same amount of
    # NULLS, this means these both variables got the same NULL rows.
    # The same situation happens with TRANS CAC CODE 2 and TRANS CAC DESC 2.
data = data[data['TRANS CAC CODE 2'].notnull()]
print(data.isnull().sum())

#Only TRANS CAC CODE 3 is left to evaluate. 
list(data['TRANS CAC CODE 3'].unique())
data = data[data['TRANS CAC CODE 3'].notnull()]
print(data.isnull().sum())

#Directory has NULLS and some repeated values
list(data['Directorate'].unique())
data = data[data['Directorate'].notnull()]
print(data.isnull().sum())
list(data['Directorate'].unique())
    #The directories will be the following: 'Adult & Communities'
    #,'Corporate Resources','CYP&F Schools','Development',
    #'Local Services','Corporate Procurement','Adult Social Care and Health'

adults_rep = {'Adult & Communities','Adults & Comms','Adults & Communities','Adults'}
auxiliar = data.replace(adults_rep,'Adults & Communities')
CYPF_Schools_rep = {'CYP&F','CYP&F  ','CYP&F SCHOOLS','CYP&F Schools','CYO&F','CYP&F '}
auxiliar1 = auxiliar.replace(CYPF_Schools_rep,'CYP&F Schools')
development_rep = {'DEVELOPMENT','Development'}
auxiliar2 = auxiliar1.replace(development_rep,'Development')
local_serv_rep = {'Local Services','Local services'}
auxiliar3 = auxiliar2.replace(local_serv_rep,'Local services')
list(auxiliar3['Directorate'].unique())
data = auxiliar3


#%% Variables description

# 1. TRANS DATE : date of transaction

# 2. TRANS VAT DESC: VAT transaction description
    #The VAT tax (Value-added tax) is payable by any taxable person making a 
    #taxable supply (‘the supplier’) of goods or services, unless it is 
    #payable by another person.
    #The VAT rates: ['VR':reduced rate, 'VZ':Zero rate, 'VL':Leisure, 
        #'VT':Transport,'VE':Education, 'VS':Standard rate]
        
# 3. ORIGINAL GROSS AMT: original gross amount
     #The income amount calculated for tax purposes to be paid.
     
# 4. MERCHANT NAME: A qué proveedor se le compró.
# 5. CARD NUMBER
# 6. TRANS CAC CODE 1: Client Account Credit Transaction code
# 7. TRANS CAC DESC 1: Concepto del gasto (viáticos,gasolina,etc)
# 8. TRANS CAC CODE 2
# 9. TRANS CAC DESC 2: Qué área de la municipalidad compró.
#10. TRANS CAC CODE 3
#11. DIRECTORATE: Bajo qué gerencia o directorio se encuentra el área.
        
#En virtud del Código de prácticas recomendadas para las 
#autoridades locales sobre transparencia de datos, se alienta 
#a los ayuntamientos a publicar todas las transacciones con 
#tarjetas de compra corporativas.

#Ya publicamos detalles de todos nuestros gastos relevantes 
#de más de £ 500 en nuestra página Pagos a proveedores, y 
#continuaremos haciéndolo. 

#%% Variables' behaviour

data.select_dtypes('object').nunique()

data['TRANS VAT DESC'].unique()
data['TRANS CAC CODE 3'].unique()
data['Directorate'].unique()

data.describe()  #The only numerical variable 'ORIGINAL GROSS AMT'

#%% Descriptive analysis

# Objective:
# 1.Examine the types of expenses and their amounts by area or directorate.
    # -Expense forecast or
    # -Weird transactions or anomalies by cluster


#%%  1. The distribution of amounts per directorate
df_directory = data.groupby('Directorate')['ORIGINAL GROSS AMT'].agg('sum').reset_index()
df_directory['Percentage'] = 100 * df_directory['ORIGINAL GROSS AMT'] / df_directory['ORIGINAL GROSS AMT'].sum()
df_directory.sort_values('Percentage', ascending=False)

#The directorate Corporate Resources shows almost 44% of the
#   expenses so it will be analyzed in detail. 
df_corporate_resources = data[data['Directorate']=='Corporate Resources']
df_corporate_resources['Year'] = pd.DatetimeIndex(df_corporate_resources['TRANS DATE']).year 
df_corporate_resources['Month'] = pd.DatetimeIndex(df_corporate_resources['TRANS DATE']).month

import matplotlib.pyplot as plt
x = df_corporate_resources.set_index('TRANS DATE').groupby(pd.Grouper(freq='M')).agg({'ORIGINAL GROSS AMT':'sum'})
x.reset_index(level=0, inplace=True)

plt.plot(x['TRANS DATE'],x['ORIGINAL GROSS AMT'],color='green',linewidth=2, markersize=12)
plt.gcf().autofmt_xdate()
plt.title('Corporate Resources: Gross Amount per month', fontsize=15)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Gross amount', fontsize=12)
plt.axvline(x=['2014-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2016-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2017-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-01-01'], color='red')
plt.axvline(x=['2016-01-01'], color='red') 
plt.axvline(x=['2017-01-01'], color='red') 
plt.axvline(x=['2018-01-01'], color='red') 
    
#The directorate CYP&F Schools shows almost 33% expenses, another strong candidate. 
df_schools = data[data['Directorate']=='CYP&F Schools']
df_schools['Year'] = pd.DatetimeIndex(df_schools['TRANS DATE']).year 
df_schools['Month'] = pd.DatetimeIndex(df_schools['TRANS DATE']).month

y = df_schools.set_index('TRANS DATE').groupby(pd.Grouper(freq='M')).agg({'ORIGINAL GROSS AMT':'sum'})
y.reset_index(level=0, inplace=True)
plt.plot(y['TRANS DATE'],y['ORIGINAL GROSS AMT'],color='orange',linewidth=2, markersize=12)
plt.gcf().autofmt_xdate()
plt.title('CYP&F Schools: Gross Amount per month', fontsize=15)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Gross amount', fontsize=12)
plt.axvline(x=['2014-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2016-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2017-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-01-01'], color='red')
plt.axvline(x=['2016-01-01'], color='red') 
plt.axvline(x=['2017-01-01'], color='red') 
plt.axvline(x=['2018-01-01'], color='red') 

#The directorate Local services shows almost 16% expenses. 
df_local = data[data['Directorate']=='Local services']
df_local['Year'] = pd.DatetimeIndex(df_local['TRANS DATE']).year 
df_local['Month'] = pd.DatetimeIndex(df_local['TRANS DATE']).month

z = df_local.set_index('TRANS DATE').groupby(pd.Grouper(freq='M')).agg({'ORIGINAL GROSS AMT':'sum'})
z.reset_index(level=0, inplace=True)
plt.plot(z['TRANS DATE'],z['ORIGINAL GROSS AMT'],color='magenta',linewidth=2, markersize=12)
plt.gcf().autofmt_xdate()
plt.title('Local services: Gross Amount per month', fontsize=15)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Gross amount', fontsize=12)
plt.axvline(x=['2014-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2016-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2017-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-01-01'], color='red')
plt.axvline(x=['2016-01-01'], color='red') 
plt.axvline(x=['2017-01-01'], color='red') 
plt.axvline(x=['2018-01-01'], color='red') 

#Summary table
plt.plot(x['TRANS DATE'],x['ORIGINAL GROSS AMT'],color='green',linewidth=2, markersize=12)
plt.plot(y['TRANS DATE'],y['ORIGINAL GROSS AMT'],color='orange',linewidth=2, markersize=12)
plt.plot(z['TRANS DATE'],z['ORIGINAL GROSS AMT'],color='magenta',linewidth=2, markersize=12)
plt.gcf().autofmt_xdate()
plt.title('Top 3 directorates: Gross Amount per month', fontsize=15)
plt.xlabel('Months', fontsize=12)
plt.ylabel('Gross amount', fontsize=12)
plt.axvline(x=['2014-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2016-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2017-07-01'], color='gray', linestyle='--')
plt.axvline(x=['2015-01-01'], color='red')
plt.axvline(x=['2016-01-01'], color='red') 
plt.axvline(x=['2017-01-01'], color='red') 
plt.axvline(x=['2018-01-01'], color='red') 

#Outliers' treatment
x.describe()
x.boxplot() #We have an outlier
plt.scatter(x['ORIGINAL GROSS AMT'],x['TRANS DATE'],c='g') #We have 3 outliers

x[x['ORIGINAL GROSS AMT']>500000]
x[x['ORIGINAL GROSS AMT']<0]
x['ORIGINAL GROSS AMT'] = x['ORIGINAL GROSS AMT'].replace([0,614.03,540198.56,666557.89,709169.61],x['ORIGINAL GROSS AMT'].median())
x['ORIGINAL GROSS AMT'].iloc[42] = x['ORIGINAL GROSS AMT'].median()

#%% PROBLEM STATEMENT:
# Build a model to forecast the expenses of the 
# directorate Corporate Resources for the next 2 years.

#%% FORECASTING FOR DIRECTORATE "Corporate Resources" 

#Time series analysis: rolling 12 months
#Calculating rolling estadistics
indexedDataset = x.set_index(['TRANS DATE'])
rolmean = indexedDataset.rolling(window=12).mean() #monthly level
rolstd = indexedDataset.rolling(window=12).std()
print(rolmean,rolstd) 
#The value on 2015-02-28 after rolling is the mean of the first
#12 values in the original data.

#let's see if the mean and standard deviation are constant.
orig = plt.plot(indexedDataset,color='blue',label='Original data')
mean = plt.plot(rolmean,color='red',label='Rolling mean')
std = plt.plot(rolstd,color='black',label='Rolling std')
plt.legend(loc='best')
plt.title('Rolling Mean & Rolling Standard Deviation')
plt.gcf().autofmt_xdate()
plt.show()
#From the graphic we have a glimpse that the mean and the std are 
#not constant so the data doesn't seem stationary. 

#Checking with stadistic tests (Ho: timeseries is not stationary)
from statsmodels.tsa.stattools import adfuller
    # Dickey-Fuller Test
print('Results of Dickey-Fuller Test')
result = adfuller(z['ORIGINAL GROSS AMT'], autolag='AIC')
dfoutput = pd.Series(result[0:4],index=['Test statistic','pvalue','#Lags used','Number of obs used'])
for key,value in result[4].items():
    dfoutput['Critical Value (%s)'%key]=value
print(dfoutput)
if result[1] > 0.05: print('Series is not Stationary')
else: print('Series is Stationary')
    #We have a pvalue of  0.390077
    

#Estimating the trend
import numpy as np
    #Numbers on the y axis have changed because we've 
    #changed the scale through a logaritmic function. 
    #We see the trend also has changed
indexedDataset_logScale = np.log(indexedDataset)
plt.plot(indexedDataset_logScale)
plt.gcf().autofmt_xdate()

#Calculating the moving average with the same window but taking the 
#log series.
movingAverage = indexedDataset_logScale.rolling(window=12).mean()
movingSTD = indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color='red')
plt.gcf().autofmt_xdate()
    #The mean is still not stationary but it's quite better than the 
    #previous one so we can say the data is not stationary again.

datasetLogScaleMinusMovingAverage = indexedDataset_logScale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)
    #Then we remove the NaN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

#-------
def test_stationarity(timeseries):
    #Determining rolling stadistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    #Plot rolling stadistics:
    orig = plt.plot(timeseries,color='blue',label='Original data')
    mean = plt.plot(movingAverage ,color='red',label='Rolling mean')
    std = plt.plot(movingSTD ,color='black',label='Rolling std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.gcf().autofmt_xdate()
    plt.show()
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test')
    result = adfuller(timeseries['ORIGINAL GROSS AMT'], autolag='AIC')
    dfoutput = pd.Series(result[0:4],index=['Test statistic','pvalue','#Lags used','Number of obs used'])
    for key,value in result[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    if result[1] > 0.05 : print('Series is not Stationary')
    else: print('Series is Stationary')
test_stationarity(datasetLogScaleMinusMovingAverage)
    #The p-value is 5.252e-08 <0.05           -> now stationary
    #and the Critical value > Test statistic  -> now stationary
#-------
    #The graphic shows no such trend in the mean or the std
    #and both look much better than its predecesor. 

#Time to see the trend inside the time series
exponentialDecayWeightedAverage = indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage,color='red') 
     #As the timeseries progresses the average is also
     #progressing towards the lower side meaning the trend
     #goes downward and keeps decreasing with respect to the time.
     
#another transformation
indexedDataset_logScale.dropna(inplace=True)
datasetLogScaleMinusMovingExponentialDecayAverage = indexedDataset_logScale-exponentialDecayWeightedAverage
datasetLogScaleMinusMovingExponentialDecayAverage.dropna(inplace=True)
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)  
    #This was just in case and there are no variations 

#%% ARIMA MODEL (p,d,q)
    #(Autoregressive Integrated Moving Average Model)
#ARIMA model has 3 models in it: 
    # AR model or auto regressive
    # MA model for moving average
    # I model for the integration
#About the parameters:
    # p: The number of lag observations included in the model, also called the lag order.
    # d: The number of times that the raw observations are differenced, also called the degree of differencing.
    # q: The size of the moving average window, also called the order of moving average.

datasetLogDiffShifting = indexedDataset_logScale - indexedDataset_logScale.shift()
plt.plot(datasetLogDiffShifting)
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)
    #Now we see the mean and std are quite flat and so much better than before.
    #Stationarity = constant mean, constant variance & autocovariance that doesn't depend on time
    #Officially can say the time series are stationary now.
    
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logScale,period=2)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logScale,label='Original data')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonal')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label='Residual')
plt.legend(loc='best')
plt.tight_layout()

decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
#The function Seasonal decompose changed the output to series instead of dataframe
#We proceed to change the type.
serie = pd.Series.to_frame(decomposedLogData)
serie.columns = ['ORIGINAL GROSS AMT']
test_stationarity(serie)
    #From the graph output, visually we can say it's not stationary
    #that's why the moving average parameter must be in place
    #to smoothen the graph and predict whay's next

#ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(datasetLogDiffShifting,nlags=20)
lag_pacf = pacf(datasetLogDiffShifting,nlags=20,method='ols') #Ordinary least square method

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Autocorrelation Function (Qvalue)')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function (Pvalue)')
plt.tight_layout()
    #Both the value of P and Q are between the interval [0,2.5]

from statsmodels.tsa.arima_model import ARIMA
#AR Model
model = ARIMA(indexedDataset_logScale,order=(2,1,2)) #p,d,q 
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS: %.4F'% sum((results_AR.fittedvalues-datasetLogDiffShifting['ORIGINAL GROSS AMT'])**2))
print('Plotting AR Model')
    #We want the lowest RSS value but instead have 18.3441

#MA Model
model = ARIMA(indexedDataset_logScale,order=(2,1,2))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues,color='red')
plt.title('RSS: %.4F'% sum((results_MA.fittedvalues-datasetLogDiffShifting['ORIGINAL GROSS AMT'])**2))
print('Plotting MA Model')

model = ARIMA(indexedDataset_logScale,order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS: %.4F'% sum((results_ARIMA.fittedvalues-datasetLogDiffShifting['ORIGINAL GROSS AMT'])**2))
    #Here the ARIMA model brings an RSS of 18.3441

#Fitting on the time series converting the fitted values into series format
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff.head())

#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)

#Having the predictions run for the fitted values
predictions_ARIMA_log = indexedDataset_logScale
predictions_ARIMA_log = pd.Series(indexedDataset_logScale['ORIGINAL GROSS AMT'],index=indexedDataset_logScale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum)
predictions_ARIMA_log.head()
#Bringing the data in the original format.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)
plt.title('ARIMA Model (orange line)')

#Time to do predictions
indexedDataset_logScale.count()

results_ARIMA.plot_predict(1, 46+12*2) #I wanna predict 2 years (12*2 months)
#r = results_ARIMA.forecasts(steps=12*2)

#%% CONCLUSION
#   According to the graph above, the expenses of the 
#   "Corporate Resources" directory would be decreasing 
#   for the next 2 years.










