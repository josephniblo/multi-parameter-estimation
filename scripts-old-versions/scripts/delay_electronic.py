import os, sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.environ["TTAG"])
from ttag import *


if getfreebuffer() == 0:
   	buf = TTBuffer(0)
else:
   	buf = TTBuffer(getfreebuffer() - 1)

if buf.getrunners() == 0:
	buf.start()


# Options
detector_names = [12, 4, 2, 10]


def estimate_delays_from_buffer(buf, scan_range, step_size, channels, integration_time=1):
    cc = []
    scan_points = np.arange(scan_range[0],scan_range[1],step_size)

    for x in scan_points:
        cc.append(buf.multicoincidences(integration_time,step_size / 2,channels,[0,x]))

    
    delay = scan_points[np.argmax(cc)]

    if np.average(cc) == 0:
        cc_significance = 0
    else:
        cc_significance = np.max(cc) / np.average(cc)

    print(f"Max coincidences: {np.max(cc)}, Delay: {delay*1e9:.2f} ns, Significance: {cc_significance:.2f}")

    #plot   
    plt.plot(scan_points, cc, label='Data')
    plt.xlabel("Delay [s]")
    plt.ylabel("Coincidences")
    plt.title("Delay scan")
    plt.grid()
    plt.legend()
    plt.savefig(f"delay_scan_{channels}.png")
    plt.close()
	
    return scan_points, cc, delay, cc_significance


def get_max_spanning_tree(pairwise_delays):
    G = nx.Graph()

    # Add edges with significance as weight
    for det_i, det_j, fine_delay, significance in pairwise_delays:
        G.add_edge(det_i, det_j, weight=significance, fine_delay=fine_delay)

    # Compute the maximum spanning tree
    mst = nx.maximum_spanning_tree(G, weight='weight')

    return list(mst.edges(data=True))


# Get all pairwise coarse delays
# Initial coarse range for delay scan: +/- 1000 ns
scan_range = [-1000e-9, 1000e-9]
coarse_step_size = buf.resolution*100

detector_indices = [name - 1 for name in detector_names]

pairwise_delays = []

for i, det_i in enumerate(detector_indices):
    for j, det_j in enumerate(detector_indices[i+1:]):
        print("Estimating delay for detectors", det_i, det_j)
        # coarse
        channels = [det_i, det_j]
        xs, cc, coarse_delay, significance = estimate_delays_from_buffer(buf, scan_range, coarse_step_size, channels, 1)

        # fine range for delay scan: +/- 10 ns
        scan_range = [coarse_delay - coarse_step_size, coarse_delay + coarse_step_size]

        step_size = buf.resolution*5
        xs, cc, fine_delay, significance = estimate_delays_from_buffer(buf, scan_range, step_size, channels, 5)
        pairwise_delays.append([det_i, det_j, fine_delay, significance])


# Get the maximum spanning tree
mst_edges = get_max_spanning_tree(pairwise_delays)

delays = {}

print("Maximum Spanning Tree Edges:")
for edge in mst_edges:
    det_i, det_j, data = edge
    fine_delay = data['fine_delay']
    significance = data['weight']

    print(f"Detectors {det_i+1} and {det_j+1}: Fine delay = {fine_delay*1e9:.2f} ns, Significance = {significance:.2f}")


mst_graph = nx.Graph()
for i, j, data in mst_edges:
    mst_graph.add_edge(i, j, delay=data['fine_delay'])

# Compute signed delays from reference using BFS
reference = detector_indices[0]
delays = {reference: 0.0}
visited = set([reference])
queue = [reference]

while queue:
    current = queue.pop(0)
    for neighbor in mst_graph.neighbors(current):
        if neighbor not in visited:
            edge_data = mst_graph[current][neighbor]
            delay = edge_data['delay']
            # Direction: from current to neighbor
            if (current, neighbor) in mst_graph.edges:
                delays[neighbor] = delays[current] + delay
            else:
                delays[neighbor] = delays[current] - delay
            visited.add(neighbor)
            queue.append(neighbor)

# Print delays
print(f"\nRelative delays from reference detector {reference}")
for det in sorted(delays):
    print(f"Detector {det+1}: Delay = {delays[det]*1e9:.2f} ns")

# plt.plot(xs,cc)
# plt.xlabel("Delay [s]")
# plt.ylabel("Coincidences")
# plt.title("Delay scan")
# plt.grid()
# plt.show()
# plt.savefig("delay_scan.png")
# plt.close()
# print("Max coincidences", max(cc))

# print("Delay [ns]", delay*1e9)

