import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates



dati_product_A=pd.read_excel('/home/davide/Desktop/script spyder/Grad_Data_set.xlsx', 
                             sheet_name='product_A',
                             index_col='date')
dati_product_B=pd.read_excel('/home/davide/Desktop/script spyder/Grad_Data_set.xlsx', 
                             sheet_name='product_B',
                             index_col='date')



Non_matchingDatesA = [x for x in dati_product_A.index if x not in dati_product_B.index]

Non_matchingDatesB = [x for x in dati_product_B.index if x not in dati_product_A.index]

non_matching_dates=Non_matchingDatesA+Non_matchingDatesB

dati_aggiustatiA=dati_product_A.drop(Non_matchingDatesA)
dati_aggiustatiB=dati_product_B.drop(Non_matchingDatesB)


dati_aggiustatiA['rv_20']=""
dati_aggiustatiB['rv_20']=""


#1
period=20
dati_aggiustatiA['rv_20']=dati_aggiustatiA['Return'].rolling(window=20).std()*np.sqrt(period)
dati_aggiustatiB['rv_20']=dati_aggiustatiB['Return'].rolling(window=20).std()*np.sqrt(period)


#2 graph implied & real for both
#Here I convert dataframes to numpy objects and also I convert the dataframe index to an array 
#with type datetime
date_n=pd.to_datetime(dati_aggiustatiA.index)
datiA=dati_aggiustatiA.values
datiB=dati_aggiustatiB.values

fig, (ax1,ax2) =plt.subplots(2,1,dpi=300, sharex=True)

formatter = mdates.DateFormatter('%Y-%m')
ax1.xaxis.set_minor_formatter(formatter)
ax1.xaxis.set_major_formatter(formatter)
ax1.tick_params(axis='x', rotation=45, labelsize=6, direction='in')
ax2.xaxis.set_minor_formatter(formatter)
ax2.xaxis.set_major_formatter(formatter)
ax2.tick_params(axis='x', rotation=45, labelsize=6, direction='in')


ax1.plot(date_n[20:], datiA[20:,0], markersize=3, color='lightblue', linewidth=1)
ax1.plot(date_n[20:], datiA[20:,2], color='darkslategrey', markersize=3, linewidth=1)
ax1.set_title('Implied vs Realized product A')
ax1.legend(['Implied','Realized'], loc='best', fontsize='small')

ax2.plot(date_n[20:], datiB[20:,0], markersize=3, color='lightblue', linewidth=1)
ax2.plot(date_n[20:], datiB[20:,2], color='darkslategrey', markersize=3, linewidth=1)
ax2.set_title('Implied vs Realized product B')
ax2.legend(['Implied','Realized'], loc='best', fontsize='small')

#3 rolling correlation

# corr_coeficient=datiA[0].roll

# np.corrcoef(x, y)

# correlazione = dati_aggiustatiA['vols_20'].corr(dati_aggiustatiB['vols_20'])

rolling_r=dati_aggiustatiA['vols_20'].rolling(20).corr(dati_aggiustatiB['vols_20']) #with nans
fig,ax3=plt.subplots(1,1,dpi=300)
formatter = mdates.DateFormatter('%Y-%m')
ax3.xaxis.set_minor_formatter(formatter)
ax3.xaxis.set_major_formatter(formatter)
ax3.tick_params(axis='x', rotation=45, labelsize=10, direction='in')

ax3.plot(date_n[20:],rolling_r[20:], color='red', linewidth=2,
         linestyle=':' 
         )
ax3.set_title('Correlation IV A vs IV B over time (20 days window)')
