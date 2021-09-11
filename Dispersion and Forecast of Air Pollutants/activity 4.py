import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#filter August 2 data only
data = pd.read_csv('Meteo 224 Lab Activity 01-Data.csv')
#dates = data['Datum/Zeit'].str[:5].unique()
data = data[data['Datum/Zeit'].str[:10] == '02.08.2015']

#get hours
hours = data['Datum/Zeit'].str[-5:].unique()

#get important parameters 
temp = data['GRIMM-392 Temp.'].astype(np.float64)
hum = data['GRIMM-392 Humidity'].astype(np.float64)
conc = data[' PM2.5'].astype(np.float64)
speed = data['GRIMM-392 W.Speed'].astype(np.float64)
direction = data['GRIMM-392 W.Direct'].astype(np.float64)

# =============================================================================
# fig,ax = plt.subplots()
# ax2=ax.twinx()
# 
# #plot temperature and humidity
# ax.plot(np.arange(24),temp,color='red',label='temperature')
# ax2.plot(np.arange(24),hum,color='blue',label='humidity')
# 
# #plot wind speed and concentrations
# # =============================================================================
# # ax.plot(np.arange(24),direction,color='red')
# # ax2.plot(np.arange(24),speed,color='blue')
# # =============================================================================
# 
# ax.set_ylabel('Temperature ($^oC$)',fontsize=14,color='red')
# ax2.set_ylabel('Relative Humidity',fontsize=14,color='blue')
# 
# ax.spines['left'].set_color('red')
# ax2.spines['right'].set_color('blue')
#  
# ax.tick_params(axis='y', colors='red')
# ax2.tick_params(axis='y', colors='blue')
# 
# plt.xticks(np.arange(0,24,3),hours[::3],fontsize=12)
# plt.xlim(0,23)
# #plt.savefig('hum_temp.png',dpi=300,bbox_inches='tight')
# =============================================================================


distance = [57,57,58,60,63,68,73,80,89,98,106,111,113,
            110,103,96,89,83,78,73,70,66,64,61]
fig,ax = plt.subplots()
ax2=ax.twinx()

#plot distance and wind speed
ax.plot(np.arange(24),distance,color='red',label='temperature')
ax2.plot(np.arange(24),direction,color='blue',label='humidity')

ax.set_ylabel('Distance from source ($km$)',fontsize=14,color='red')
ax2.set_ylabel('Wind Direction ($^o$)',fontsize=14,color='blue')

ax.spines['left'].set_color('red')
ax2.spines['right'].set_color('blue')
 
ax.tick_params(axis='y', colors='red')
ax2.tick_params(axis='y', colors='blue')

plt.xticks(np.arange(0,24,3),hours[::3],fontsize=12)
plt.xlim(0,23)
#plt.savefig('dist_direct.png',dpi=300,bbox_inches='tight')

















