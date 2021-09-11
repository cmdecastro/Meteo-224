import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data = pd.read_csv('Meteo 224 Lab Activity 01-Data.csv')
#dates = data['Datum/Zeit'].str[:5].unique()
dates = data['Datum/Zeit'].str[:10].unique()
dates = dates[(dates!='Datum/Zeit') & (dates!='Datum')]
hours = data['Datum/Zeit'].str[-5:].unique()
hours = hours[hours!='/Zeit']
hours = np.append(hours[-1],hours[:-1])

daily_ave = []

days_index = np.arange(0,len(dates)+1).reshape((7,7))
hourly_ave = np.zeros((7,24))
conc = data[' PM2.5']
conc = np.array(conc[conc!=' PM2.5'].astype(np.float64))
days_label = ['Saturday','Sunday','Monday','Tuesday','Wednesday',
              'Thursday','Friday']

temp_hourly = np.zeros((7,24))
hum_hourly = np.zeros((7,24))
speed_hourly = np.zeros((7,24))

# =============================================================================
# fig,ax = plt.subplots()
# ax2=ax.twinx()
# =============================================================================

## Get hourly and daily average PM2.5 conc, temperature, and humidity for each day of the week
# =============================================================================
# for i in range(7):
#     for j in range(24):
#         index = days_index[:,i]
#         if i == 6:
#             index = index[0:-1]
#         h = hours[j]
#         d = dates[index]
#         
#         days = data[data['Datum/Zeit'].isin(d+str(' ')+h)]
#         conc_ave = days[' PM2.5']
#         temp_ave = days['GRIMM-392 Temp.']
#         hum_ave = days['GRIMM-392 Humidity']
#         speed_ave = days['GRIMM-392 W.Speed']
#         
#         conc_ave = (conc_ave[conc_ave!=' PM2.5'].astype(np.float64)).mean()
#         temp_ave = (temp_ave[temp_ave!='GRIMM-392 Temp.'].astype(np.float64)).mean()
#         hum_ave = (hum_ave[hum_ave!='GRIMM-392 Humidity'].astype(np.float64)).mean()
#         speed_ave = (speed_ave[speed_ave!='GRIMM-392 W.Speed'].astype(np.float64)).mean()
#         
#         hourly_ave[i][j] = conc_ave
#         temp_hourly[i][j] = temp_ave
#         hum_hourly[i][j] = hum_ave
#         speed_hourly[i][j] = speed_ave
# =============================================================================
        
    #ax.plot(np.arange(24),hourly_ave[i],alpha=0.5)
    #ax2.plot(np.arange(24),temp_hourly[i],alpha=0.5)
    
    #ax2.plot(np.arange(24),hum_hourly[i],alpha=0.5)
    #ax2.plot(np.arange(24),speed_hourly[i],alpha=0.5)

#ax.plot(np.arange(24),np.mean(hourly_ave,axis=0),color='red',linewidth=2.5)
#ax2.plot(np.arange(24),np.mean(temp_hourly,axis=0),color='blue',linewidth=2.5)

#ax2.plot(np.arange(24),np.mean(hum_hourly,axis=0),color='blue',linewidth=2.5#)
#ax2.plot(np.arange(24),np.mean(speed_hourly,axis=0),color='blue',linewidth=2.5)


# =============================================================================
# plt.xticks(np.arange(0,24,3),hours[::3],fontsize=10)
# plt.xlim(0,23)
# plt.ylabel('PM2.5 concentration ($\mu g /m^3$)',fontsize=12)
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# =============================================================================

# =============================================================================
# plt.xticks(np.arange(0,24,3),hours[::3],fontsize=10)
# plt.xlim(0,23)
# 
# ax.set_ylabel('PM2.5 concentration ($\mu g /m^3$)',fontsize=12,color='red')
# #ax2.set_ylabel('Temperature ($^oC$)',fontsize=13,color='blue')
# 
# ax2.set_ylabel('Realtive Humidity',fontsize=13,color='blue')
# #ax2.set_ylabel('Wind Speed ($m/s$)',fontsize=13,color='blue')
# 
# ax2.spines['left'].set_color('red')
# ax2.spines['right'].set_color('blue')
# 
# ax.tick_params(axis='y', colors='red')
# ax2.tick_params(axis='y', colors='blue')
# =============================================================================

#plt.legend(bbox_to_anchor=(1.1, 1.05), loc='upper left')
#plt.savefig('hum_conc_ave.png',dpi=300,bbox_inches='tight')

## Get total average concentrations for each day of the week
# =============================================================================
# plt.bar(np.arange(7),np.mean(hourly_ave,axis=1),color='purple')
# plt.xticks(np.arange(7),days_label,fontsize=11,rotation=45)
# plt.ylim(22,34)
# plt.ylabel('PM2.5 concentration ($\mu g /m^3$)',fontsize=12)
# =============================================================================
#plt.savefig('week_ave.png',dpi=300,bbox_inches='tight')


## Get total average concentrations for weekdays and weekends
# =============================================================================
# daily_mean = np.mean(hourly_ave,axis=1)
# weekends = np.mean(daily_mean[:2])
# weekdays = np.mean(daily_mean[2:])
# 
# plt.bar(np.arange(2),[weekends,weekdays],color='slateblue')
# plt.xticks(np.arange(2),['weekends','weekdays'],fontsize=13)
# plt.ylim(20,32)
# plt.ylabel('PM2.5 concentration ($\mu g /m^3$)',fontsize=12)
# =============================================================================
#plt.savefig('weekends.png',dpi=300,bbox_inches='tight')

## Get mean and standard dev of PM2.5 concentrations for weekends and weekends
# =============================================================================
# conc = []
# length = []
# 
# for i in range(7):
#     index = days_index[:,i]
#     if i == 6:
#         index = index[0:-1]
#     d = dates[index]
#     
#     days = data[data['Datum/Zeit'].str[:10].isin(d)]
#     conc_ave = days[' PM2.5'].astype(np.float64)
#     
#     conc.append(conc_ave.values)
#     length.append(len(conc_ave))
#     
# conc = np.array(conc)
# w_ends = np.concatenate(conc[:2])
# w_days = np.concatenate(conc[2:])
# 
# w_ends = w_ends[~np.isnan(w_ends)]
# w_days = w_days[~np.isnan(w_days)]
# =============================================================================

## Get daily average concentrations from August 1 to September 17, 2015
# =============================================================================
# for d in dates:
#     day = data[data['Datum/Zeit'].str.contains(d)]
#     day_conc = np.array(day[" PM2.5"].astype(np.float64))
#     daily_ave.append(np.nanmean(day_conc))
# 
# thresh_index = np.where(np.array(daily_ave)>35)
# 
# #plt.title('Daily Ave. PM2.5 Concentrations')
# plt.plot(np.arange(1,len(dates)+1),daily_ave,color='purple',marker='o',label='daily ave.') 
# plt.xticks(np.arange(1,len(dates)+1,5),dates[0::5],rotation=45,fontsize=8)
# plt.ylabel('PM2.5 concentration ($\mu g /m^3$)',fontsize=12) 
# plt.hlines(35,xmin=0,xmax=len(dates)+1,linestyle='--',color='black',linewidth=2,label='NAAQG')
# plt.hlines(25,xmin=0,xmax=len(dates)+1,linestyle='dotted',color='black',linewidth=3,label='WHO')
# plt.xlim(0,len(dates)+1)
# plt.legend(loc='lower right')
# =============================================================================
#plt.savefig('daily_ave.png',dpi=300,bbox_inches='tight')


# =============================================================================
# ## Get average concentrations every 13 hours from August 1 to September 17, 2015
# conc = data[' PM2.5, ug/m3']
# conc = np.array(conc[conc!=' PM2.5'].astype(np.float64))
# 
# plt.title('13-hour Ave. PM2.5 Concentrations')
# plt.plot(np.arange(len(conc)/13),np.nanmean(conc.reshape(-1, 13),axis=1),color='purple',marker='o')
# plt.xlabel('13-hours')
# plt.ylabel('PM2.5 concentration ($\mu g /m^3$)') 
# =============================================================================

## Get hourly concentrations with 24 hour running average
conc = data[' PM2.5']
conc = conc[conc!=' PM2.5'].astype(np.float64)
moving_ave = conc.rolling(24).apply(np.nanmean)

# plt.title('13-hour Ave. PM2.5 Concentrations')
# =============================================================================
# plt.plot(np.arange(1,len(conc)+1),conc,color='purple',alpha=0.4,label='hourly')
# plt.plot(np.arange(1,len(conc)+1),moving_ave,color='blue',label='moving ave.')
# plt.hlines(35,xmin=0,xmax=len(conc)+1,linestyle='--',color='black',linewidth=2,label='NAAQG')
# plt.hlines(25,xmin=0,xmax=len(conc)+1,linestyle='dotted',color='black',linewidth=3,label='WHO')
# plt.xticks(np.arange(1,len(conc)+1,115),dates[0::5],rotation=45,fontsize=8)
# plt.ylabel('PM2.5 concentration ($\mu g /m^3$)',fontsize=12)
# plt.xlim(0,len(conc)+1)
# plt.legend(loc='best')
# =============================================================================
#plt.savefig('running_ave.png',dpi=300,bbox_inches='tight') 

## Get windrose plot
# =============================================================================
# conc = data[' PM2.5, ug/m3']
# conc = conc[conc!=' PM2.5'].astype(np.float64)
# 
# direct = data['GRIMM-392 W.Direct']
# direct = direct[direct!='GRIMM-392 W.Direct'].astype(np.float64)
# 
# speed = data['GRIMM-392 W.Speed']
# speed = speed[speed!='GRIMM-392 W.Speed'].astype(np.float64)
# 
# direct_max = max(direct.dropna())
# direct_min = min(direct.dropna())
# direct_range = np.linspace(direct_min,direct_max,36)
# direct_mean = np.convolve(direct_range, np.ones(2), 'valid') / 2
# 
# direct_bins = pd.cut(direct,bins=35,labels=direct_mean)
# 
# speed_max = max(speed.dropna())
# speed_min = min(speed.dropna())
# speed_range = np.linspace(speed_min,speed_max,11)
# speed_mean = np.convolve(speed_range, np.ones(2), 'valid') / 2
# 
# speed_bins = pd.cut(speed,bins=10,labels=speed_mean)
# 
# conc_array = np.zeros((len(speed_mean),len(direct_mean)))
# 
# for d in range(len(direct_mean)):
#     for s in range(len(speed_mean)):
#         d_index = direct_bins[direct_bins==direct_mean[d]].index
#         s_index = speed_bins[speed_bins==speed_mean[s]].index
#         indices = np.intersect1d(d_index,s_index) 
#         mean_conc = conc.loc[indices].mean()
#         
#         conc_array[s][d] = mean_conc
#     
# Direct, Speed = np.meshgrid(direct_mean*np.pi/180, speed_mean)
# 
# fig, ax = plt.subplots(subplot_kw={"projection":"polar"})
# 
# im = ax.pcolormesh(Direct, Speed, conc_array, cmap='Oranges')
# plt.colorbar(im,pad=0.075)
# #plt.savefig('windrose.png',dpi=300,bbox_inches='tight')
# 
# plt.show()
# =============================================================================

## Calculate and plot daily AQI 
# =============================================================================
# aqi = np.array([129,98,111,100,91,84,69,57,59,60,88,119,129,71,97,76,84,94,96,
#                 72,58,57,40,63,73,90,100,126,113,79,78,100,81,104,81,84,72,71,87,
#                 59,65,121,77,90,79,124,99,92])
# 
# fig1 = plt.figure(figsize=(8,6))
# ax1 = fig1.add_subplot(111)
# ax1.plot(np.arange(1,len(dates)+1),aqi,color='black',marker='o') 
# plt.text(len(dates)+2,20,"Good",fontsize=13)
# plt.text(len(dates)+2,70,"Moderate",fontsize=13)
# plt.text(len(dates)+2,125,"Unhealthy for",fontsize=13)
# plt.text(len(dates)+2,110,"Sensitive Groups",fontsize=13)
# plt.text(len(dates)+2,170,"Unhealthy",fontsize=13)
# plt.text(len(dates)+2,240,"Very Unhealthy",fontsize=13)
# plt.text(len(dates)+2,340,"Hazardous",fontsize=13)
# plt.text(len(dates)+2,440,"Very Hazardous",fontsize=13)
# 
# ax1.add_patch(patches.Rectangle((0, 0), len(dates), 50, color='green',alpha=0.7))
# ax1.add_patch(patches.Rectangle((0, 50), len(dates), 50, color='yellow',alpha=0.7))
# ax1.add_patch(patches.Rectangle((0, 100), len(dates), 50, color='orange',alpha=0.7))
# ax1.add_patch(patches.Rectangle((0, 150), len(dates), 50, color='red',alpha=0.7))
# ax1.add_patch(patches.Rectangle((0, 200), len(dates), 100, color='purple',alpha=0.7))
# ax1.add_patch(patches.Rectangle((0, 300), len(dates), 100, color='brown',alpha=0.7))
# ax1.add_patch(patches.Rectangle((0, 400), len(dates), 100, color='maroon',alpha=0.7))
# 
# ax1.set_xlim(0,len(dates))
# ax1.set_ylim(0,500)
# plt.xticks(np.arange(1,len(dates)+1,5),dates[0::5],rotation=45,fontsize=10)
# plt.ylabel('AQI',fontsize=14)
# =============================================================================
#plt.savefig('aqi.png',dpi=300,bbox_inches='tight')

# =============================================================================
# x = 2*((7.29E-5))*np.sin(45*np.pi/180)#((7.29E-5)**2)*(6.37E6)/(2*9.81)
# y = (200**2)/(2*25)
# =============================================================================

# =============================================================================
# x = np.cbrt(6.67E-11*5.97E24/((7.29E-5)**2))
# y = 6.37E6
# =============================================================================

# =============================================================================
# x = 287*250/9.81
# y = np.log(500/1000)
# =============================================================================
