import matplotlib.pyplot as plt
import math
import numpy as np
from datetime import datetime, timedelta
import random



## Font values #######
SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 16

plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=MEDIUM_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
###########################

temp = [20.7, 20.58, 19.95, 20.11, 20.28]
humidity = [36.16, 36.06, 36.50, 38.00, 38.65]
time_stamp = [17.00, 17.45, 18.30, 19.15, 20.00]
time_stamp_ticks_val = [17, 17.75, 18.5, 19.25, 20]
time_stamp_ticks = ["17:00", "17:45", "18:30", "19:15", "20:00"]

temp_rev = [random.uniform(*(15.4,16.5)) for _ in range(5)]
humidity_rev = [random.uniform(*(38.2,39.6)) for _ in range(5)]
    
    
    
def plot_temperature_humidity(temperatures, humidity, time):
    plt.style.use("bmh")

    # Calculate average temperature and humidity
    average_temperature = np.mean(temperatures)
    average_humidity = np.mean(humidity)
    temperature_variance = np.var(temperatures)
    humidity_variance = np.var(humidity)
    
    print("avg_temp: ", average_temperature,"\t variance_temp: ", temperature_variance, "\n avg_hmudity: ", average_humidity, "\t Variance: humidity: ", humidity_variance)
    

    # Create the plot with increased figure size
    fig, ax1 = plt.subplots(figsize=(7, 6))

    # Plot temperature
    ax1.plot(time, temperatures, '--', color='blue', linewidth=0.4, marker='o', markersize=10, mec='blue', mew=1, label='Temperature', mfc='none')

    # Plot humidity on a secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(time, humidity, '--', color='green', linewidth=0.4, marker='o', markersize=10, mec='green', mew=1, label='Humidity', mfc='none')

    ax1.set_xticks(time_stamp_ticks_val)
    ax1.set_xticklabels(time_stamp_ticks)
    
    # Plot average temperature and humidity
    ax1.plot(time[2], average_temperature, 'o', markersize=10, color="blue", mec='black', label="Average Temperature")
    ax2.plot(time[2], average_humidity, 'o', markersize=10, color="green", mec='black', label="Average Humidity")

    # Set axis labels
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)', color='blue')
    ax2.set_ylabel('Humidity (%)', color='green')

    # Set grid
    ax1.grid(True)

    # Add legend
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Set title
    plt.title('Temperature and Humidity Variation')

    # Adjust y-limits for legend
    ax1.set_ylim(bottom=min(temperatures) - 0.5, top=max(temperatures) + 1)
    ax2.set_ylim(bottom=min(humidity) - 0.5, top=max(humidity) + 1)

    # Display the plot
    plt.tight_layout()
    plt.show()


average_temperature = np.mean([16.37, 15.8, 15.84, 15.63, 15.81])
average_humidity = np.mean([38.59, 38.64, 38.42, 39.52, 38.38])
temperature_variance = np.var([16.37, 15.8, 15.84, 15.63, 15.81])
humidity_variance = np.var([38.59, 38.64, 38.42, 39.52, 38.38])

#print("avg_temp: ", average_temperature,"\t variance_temp: ", temperature_variance, "\n avg_hmudity: ", average_humidity, "\t Variance: humidity: ", humidity_variance)
X = 4
Y = 3
Z = 2.5

V = X * Y * Z
S = 2 * X * Y + 2 * X * Z + 2 * Y * Z

for i in range(1,10):
    val = 0.16 * V / (S*(1-(i/10)**2))
    print("i: ", i, "\tval: ",val)



