import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#LCL and CCL in hPa
# =============================================================================
# LCL = np.array([900,925,925,880,920,890,925,925,905,910,925,900,925,900])
# CCL = np.array([830,925,925,845,920,840,925,925,905,900,925,840,925,840])
# =============================================================================

#LCL and CCL in km
LCL = np.array([0.95,0.73,0.73,1.13,0.77,1.04,0.73,0.73,0.91,0.86,0.73,0.95,0.73,0.95])
CCL = np.array([1.6,0.73,0.73,1.46,0.77,1.5,0.73,0.73,0.91,0.95,0.73,1.5,0.73,1.5])

dates = ['05/10','05/11','05/12','05/13',
         '05/14','05/15','05/16']
x = np.arange(14)

#radiative forcing with and without aerosols
data = pd.read_csv('radiative_forcings.csv')
data1 = pd.read_csv('rad_forcing_without.csv')

#diffuse up with and without aerosols
data2 = pd.read_csv('diffuse_up_with.csv')
data3 = pd.read_csv('diffuse_up_without.csv')

wavelengths = data.columns[1:]
lambda_ = np.array([255,305,355,405,455,505,555,605,655,705])

bg = plt.imread('visible-spectrum.png')

x_axis = np.arange(len(wavelengths))


#plot diffuse up with and without aerosols
plt.figure(figsize=(8,6))

plt.imshow(bg,alpha=0.5,extent=[267,810,0,600])

plt.plot(np.linspace(0,800,10),lambda_*data.loc[0][1:].astype(float),label=dates[0],marker='o',linewidth=3)
plt.plot(np.linspace(0,800,10),lambda_*data.loc[1][1:].astype(float),label='05/11-05/16',marker='o',linewidth=3)

plt.plot(np.linspace(0,800,10),lambda_*data1.loc[0][1:].astype(float),label=dates[0],marker='o',linewidth=3,linestyle='--')
plt.plot(np.linspace(0,800,10),lambda_*data1.loc[1][1:].astype(float),label='05/11-05/16',marker='o',linewidth=3,linestyle='--')

plt.xticks(np.linspace(0,800,10),wavelengths,rotation=45,fontsize=14)
plt.xlabel('wavelengths (nm)',fontsize=18)
plt.xlim(xmax=810)

plt.yticks(fontsize=14)
plt.ylabel('net radiative forcing ($W/m^2$)',fontsize=18)
plt.ylim(0,600)


plt.legend(loc='best',fontsize=14)
#plt.savefig('rad_forcing.png',dpi=300,bbox_inches='tight')

plt.show()

#plot radiative forcing
# =============================================================================
# plt.figure(figsize=(8,6))
# 
# plt.imshow(bg,alpha=0.5,extent=[3,9.2,0,6])
# 
# for i in range(len(data)):
#     plt.plot(wavelengths,data.loc[i][1:].astype(float)*5,label=dates[i],marker='o',linewidth=3)
# 
# plt.xticks(rotation=45,fontsize=12)
# plt.xlabel('wavelengths (nm)',fontsize=13)
# plt.xlim(xmax=9.2)
# 
# plt.yticks(np.linspace(0,6,4),np.round(np.linspace(0,1.2,4),1),fontsize=12)
# plt.ylabel('radiative forcing ($W/m^2\ nm^{-1} $)',fontsize=13)
# plt.ylim(0,6)
# 
# 
# plt.legend(loc='best')
# #plt.savefig('rad_forcing.png',dpi=300,bbox_inches='tight')
# 
# plt.show()
# =============================================================================

#plot cloud baase height and cloud top height for each day
# =============================================================================
# plt.plot(x,LCL,label='cloud base',color='green',marker='o')
# plt.plot(x,CCL,label='cloud top',color='purple',marker='o')
# 
# plt.ylabel('height (km)',fontsize=13)
# plt.ylim(0.7,1.7)
# 
# #plt.ylabel('height (hPa)',fontsize=13)
# #plt.ylim(950,800)
# 
# plt.yticks(fontsize=12)
# plt.xticks(np.arange(0,14,2),dates,fontsize=12)
# 
# plt.legend(loc='best')
# 
# #plt.savefig('cloud_base_top_km.png',dpi=600,bbox_inches='tight')
# plt.show()
# 
# =============================================================================
