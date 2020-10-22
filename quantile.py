import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
data=pd.read_excel('/home/davide/Downloads/Telegram Desktop/gammat.xlsx')

#Return Close-to-Close SP500
data['daily return']= data['price'].pct_change() # period return
data['daily return'][0]=0
data['cum return']=data['daily return'].cumsum()


#definisci in quanti quantili
q=10

#Gamma and change gamma quantiles and concatenate columns to data

data['gamma_change']=data['gex'].pct_change()
data['gamma_change'][0]=0
data['return_t+1']=data['price'].pct_change()
data['return_t+1'][0]=0
#shiftiamo la colonna price change di una riga su
data['return_t+1'] = data['return_t+1'].shift(-1)
#eliminiamo l'ultima riga col nan
data.drop(data.shape[0] -1, inplace=True)
data['quantile_gamma']= pd.qcut(data['gex'],q = q, labels = False)
data['quantile_gamma_change']= pd.qcut(data['gamma_change'],q = q, labels = False)

numeri_quantili=data['quantile_gamma'].unique()
numeri_quantili.sort()


lista_vuota = []
                                                                #return t+1 e gamma quantile
for quantile in numeri_quantili:
    colonna = np.array(data.loc[data['quantile_gamma']==quantile]['return_t+1'])
    
    lista_vuota.append(colonna)

lista_nomi=['quantile: '+str(quantile)  for quantile in numeri_quantili]

df=pd.DataFrame(lista_vuota).transpose()     
df.columns=lista_nomi
#fill nans = 0
df.fillna(value=0,inplace=True)
summary = df.describe(include='all')

#create a subplot without frame
ax = plt.subplot(111, frame_on=False)
#remove axis
ax.xaxis.set_visible(False) 
ax.yaxis.set_visible(False) 
table(ax,summary,loc='center',colWidths=[0.25]*len(summary.columns))
#save the plot as a png file
plt.savefig('summary_plot.png')



x_mean=summary.loc['mean']
x_std=summary.loc['std']*(1/252)
fig2,ax2=plt.subplots(1,1, dpi=300)
ax2.patch.set_facecolor('#ababab')
ax2.patch.set_alpha(0.3)
ax2.set_axisbelow(True)
ax2.yaxis.grid(color='gray', linestyle='dashed')
ax2.set_ylabel('return t+1')
plt.plot(summary.columns,x_mean,linewidth=1)
plt.plot(summary.columns, x_std,linewidth=1)
last_quantile=data['quantile_gamma'].iloc[-1]

plt.vlines(x=last_quantile, ymin=min(summary.loc['mean']) ,ymax=max(summary.loc['std']), colors='purple', ls='--', lw=0.5)
fig2.autofmt_xdate()
ax2.set_title("Return close-to-close at t+1 of SP500 on GEX quantiles")
plt.legend(['Mean','Stdev','Last GEX value'])
plt.show()

