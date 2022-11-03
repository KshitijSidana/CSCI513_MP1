from calendar import day_abbr
import matplotlib.pyplot as plt
import csv
  
x = []
y = []
data = []
  
with open('logs/episode-00000.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        data.append(row)

for row in data[1:]:
    x.append(row[0])
    y.append(float(row[1]))
  
plt.plot(x, y, color = 'g', linestyle = 'dashed',
         marker = 'o',label = "Weather Data")
  
plt.xticks(rotation = 25)
plt.xlabel('Dates')
plt.ylabel('Temperature(°C)')
plt.title('Weather Report', fontsize = 20)
plt.grid()
plt.legend()
plt.show()