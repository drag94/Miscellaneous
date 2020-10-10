import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

import matplotlib.dates as mdates
# =============================================================================
# 
# 
# =============================================================================

dati=pd.read_csv('C:/Users/39339/Desktop/vixcurrent.csv')



dati['spread']=dati['VIX Close']-dati['Vxn Close']
spread=dati['spread']
media=dati['spread'].mean()
diff_media_ultimo=spread.values[-1]-media
mediana=dati['spread'].median(axis=0)

# retta zero al grafico dello spread
y=np.zeros(len(dati.index))

ultimovix= round(dati['VIX Close'].values[-1],2)
ultimovxn= round(dati['Vxn Close'].values[-1],2)
ultimospread=round(spread.iloc[-1],2)
lastdate= str(date_n.iloc[-1])
# =============================================================================
# # PRENDO LA COLONNA DATE E LO CONVERTO in numpy datetime64 POI
# # converto il dataframe in numpy per fare i grafici

# =============================================================================
date_n=pd.to_datetime(dati['Date'])
dati=dati.values


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, dpi=300)

# qui formatto le date e i tick pure
formatter = mdates.DateFormatter('%Y-%m')
ax1.xaxis.set_minor_formatter(formatter)
ax1.xaxis.set_major_formatter(formatter)
ax1.tick_params(axis='x', rotation=45, labelsize=4, direction='in')
ax3.xaxis.set_minor_formatter(formatter)
ax3.xaxis.set_major_formatter(formatter)
ax3.tick_params(axis='x', rotation=45, labelsize=4, direction='in')

ax1.plot(date_n[4000:], dati[4000:,1], markersize=3, color='lightblue', linewidth=1)
ax1.plot(date_n[4000:], dati[4000:,2], color='darkslategrey', markersize=3, linewidth=1)
ax1.set_title('VIX-VXN')
ax1.legend(['VIX ' + str(ultimovix), 'VXN '+ str(ultimovxn)], loc='best', fontsize='small')


ax3.plot(date_n[4000:],dati[4000:,3], color='lightblue')
ax3.plot(date_n[4000:], dati[4000:,3]*0, linestyle='--')
ax3.legend(['Spread VIX-VXN ' + str(ultimospread)], loc='best', fontsize='x-small')

ax4.hist(spread[4000:], bins=30, orientation='horizontal')

ax2.axes.xaxis.set_visible(False)
ax2.axes.yaxis.set_visible(False)
ax2.axis('off')
ax2.text(0.0,0.75,'$SUMMARY$ \n'
        'Last: '+str(round((dati[-1,3]),2)), horizontalalignment='left')
ax2.text(0.0,0.65,'Mean: '+str(round(media,2)), horizontalalignment='left')

ax2.text(0.0,0.55,'Diff From Mean: '+str(round(diff_media_ultimo,2)))

ax2.text(0.0,0.45,'Median: '+str(round((mediana),2)))
ax2.text(0.0,0.35,'Diff From Med.: '+str(round((dati[-1,3]- mediana),2)))
ax2.text(0.0,0.25,'Stdev: '+str(round(statistics.stdev(dati[4000:,3]),2)))
ax2.text(0.0,0.15, 'Percentile at 5%: '+str(round(np.percentile(dati[4000:,3],95),2)))   

# NUOVO ATTRIBUTO NEL SUMMARY
ax2.text(0.0,0.05, 'Last date: '+lastdate)              
         















