import matplotlib.pyplot as plt
import math

def gaussian_pulse(timestep, peak_time, sigma):
    amplitude = 1.0  # Amplitude of the Gaussian pulse
    mean = peak_time  # Mean (center) of the Gaussian pulse

    # Calculate the value of the Gaussian pulse at the given timestep
    value = amplitude * math.exp(-((timestep - mean) ** 2) / (2 * sigma ** 2))

    return value

# Parameters for the Gaussian pulse
peak_time = 10  # Center of the pulse
sigma = 2  # Width of the pulse

# Array to store the pulse values
pulse_values = []

# Iterate over timesteps and calculate Gaussian pulse values
for timestep in range(20):
    if timestep > 1 and timestep < 25:
        value = gaussian_pulse(timestep, peak_time, sigma)
        pulse_values.append(value)
    else: pulse_values.append(0)

# Plot the Gaussian pulse
plt.plot(pulse_values)
plt.xlabel('Timestep')
plt.ylabel('Value')
plt.title('Gaussian Pulse')
plt.show()