import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#We put all the lines in a list
data = []
with open("Lab Activity 2.txt") as fp:
    lines = fp.read()
    data = lines.split('\n')

df_data= []
for item in data:
    df_data.append(item.split()) #I cant see if 1 space or 2 separate the values

#df_data should be something like [[row1col1,row1col2,row1col3],[row2col1,row2col2,row3col3]]

#List to dataframe

df = pd.DataFrame(df_data)
df = df.drop(columns=list(range(11,16)))

h= pd.to_numeric(df[1], errors='coerce').dropna()
t = pd.to_numeric(df[2], errors='coerce').dropna()


index = h[h<1000].index

daily_index = []
counter = 0
checker = 0

labels = ['May 10','May 11','May 12','May 13','May 14','May 15','May 16']
colors = ['red','orange','brown','green','blue','indigo','violet']

temp_series_600 = []
temp_series_700 = []
temp_series_800 = []
temp_series_900 = []
temp_series_1000 = []

for i in index:
    if i == 102:
        continue
    elif i == 5 or i - daily_index[-1] == 1:
        daily_index.append(i)
    else:        
        height = h.loc[daily_index[1:]]
        temp = t.loc[daily_index[1:]]
        
        rounded_height = round(height,-2)
        print(rounded_height)
        
        height_index1 = rounded_height[rounded_height==600].index
        height_index2 = rounded_height[rounded_height==800].index
        height_index3 = rounded_height[rounded_height==900].index
        height_index4 = rounded_height[rounded_height==700].index
        height_index5 = rounded_height[rounded_height==1000].index
        
        temp_mean1 = np.mean(temp.loc[height_index1])
        temp_mean2 = np.mean(temp.loc[height_index2])
        temp_mean3 = np.mean(temp.loc[height_index3])
        temp_mean4 = np.mean(temp.loc[height_index4])
        temp_mean5 = np.mean(temp.loc[height_index5])
        
        temp_series_600.append(temp_mean1)
        temp_series_800.append(temp_mean2)
        temp_series_900.append(temp_mean3)
        temp_series_700.append(temp_mean4)
        temp_series_1000.append(temp_mean5)
        
# =============================================================================
#         if counter % 2 != 0:
#             plt.plot(temp,height,label=labels[checker],color=colors[checker])
#             print(round(height,-2)) 
#             checker += 1
# =============================================================================
        
        counter += 1

        daily_index = [i]

plt.plot(range(len(temp_series_600)),temp_series_600,label='600 m',color='red',marker='o')
plt.plot(range(len(temp_series_800)),temp_series_800,label='800 m',color='blue',marker='o')
plt.plot(range(len(temp_series_900)),temp_series_900,label='900 m',color='green',marker='o')
plt.plot(range(len(temp_series_700)),temp_series_700,label='700 m',color='purple',marker='o')
plt.plot(range(len(temp_series_1000)),temp_series_1000,label='1000 m',color='orange',marker='o')
plt.xticks(np.arange(0,14,2),labels,fontsize=12,rotation=45)
plt.ylabel('temperature ($^oC$)',fontsize=13)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left',fontsize=10)
# plt.savefig('time_series.png',dpi=300,bbox_inches='tight')

# =============================================================================
# plt.ylim(614,982)
# plt.xlabel('Temperature ($^oC$)',fontsize=12)
# plt.ylabel('altitude (m)',fontsize=12)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.legend()
# =============================================================================
#plt.savefig('evening.png',dpi=300,bbox_inches='tight')

plt.show()