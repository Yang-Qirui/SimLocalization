import numpy as np
import matplotlib.pyplot as plt

'''Do not ensure connectivity'''

# Number of sensors
n_sensors = 50

# Generate random sensor locations in a 100x100 area
sensor_locations = np.random.rand(n_sensors, 2) * 100

def calculate_rssi(d, rssi_at_d0=-30, d0=1, n=2.5, sigma=0):
    """Calculate RSSI using the log-distance path-loss model."""
    rssi = rssi_at_d0 - 10 * n * np.log10(d / d0)
    noise = np.random.normal(0, sigma)
    return rssi + noise

# Set communication range
comm_range = 20

rssi_matrix = np.zeros((n_sensors, n_sensors))

for i in range(n_sensors):
    for j in range(n_sensors):
        if i != j:
            distance = np.linalg.norm(sensor_locations[i] - sensor_locations[j])
            if distance <= comm_range:
                rssi_matrix[i][j] = calculate_rssi(distance)

plt.scatter(sensor_locations[:, 0], sensor_locations[:, 1], c='blue')
for i in range(n_sensors):
    for j in range(n_sensors):
        if rssi_matrix[i][j] != 0:
            plt.plot([sensor_locations[i][0], sensor_locations[j][0]], 
                     [sensor_locations[i][1], sensor_locations[j][1]], 'r-')

plt.savefig("./data/network_distribute.png")
np.save('./data/rssi.npy', rssi_matrix)
np.save('./data/groundtruth_pos.npy', sensor_locations)
