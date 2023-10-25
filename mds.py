import random
from matplotlib import pyplot as plt
import numpy as np
import argparse
import heapq
from sklearn.manifold import MDS

# Convert RSSI to Distance using LDPL model
def rssi_to_distance(rssi, r0=-30, n=2.5, d0=1):
    return d0 * 10**(np.abs(r0 - rssi) / (10*n))

# Dijkstra's algorithm to find shortest path
def dijkstra(distance_matrix, start):
    n = len(distance_matrix)
    shortest_path = [float('inf')] * n
    shortest_path[start] = 0
    predecessors = [-1] * n  # Store the path
    unvisited_nodes = [(0, start)]
    
    while unvisited_nodes:
        current_distance, current_node = heapq.heappop(unvisited_nodes)
        for neighbor, distance in enumerate(distance_matrix[current_node]):
            distance_through_current = current_distance + distance
            if distance_through_current < shortest_path[neighbor]:
                shortest_path[neighbor] = distance_through_current
                predecessors[neighbor] = current_node  # Update the path
                heapq.heappush(unvisited_nodes, (distance_through_current, neighbor))
                
    return shortest_path, predecessors

def retrieve_path(predecessors, start, end):
    path = []
    while end != start:
        path.append(end)
        end = predecessors[end]
    path.append(start)
    return path[::-1]  # Reverse the path

def smacof_incomplete_rssi(rssi_matrix, dim=2, max_iter=3000):
    # Step 1: Convert RSSI to distance
    distance_matrix = rssi_to_distance(rssi_matrix)
    
    # Identify missing distances (where RSSI was zero)
    missing_mask = (rssi_matrix == 0)
    
    print(rssi_matrix)
    # Set missing distances to a large value (or any initialization scheme)
    max_known_distance = np.max(distance_matrix[~missing_mask])
    distance_matrix[missing_mask] = max_known_distance
    
    
    # Step 3: Apply SMACOF using sklearn's implementation, with weights
    mds = MDS(n_components=dim, metric=True, dissimilarity='precomputed', max_iter=max_iter)
    coords = mds.fit_transform(distance_matrix)
    
    return coords

def run(args):
    rssi_mat = np.load("./data/rssi.npy")
    np.set_printoptions(threshold=np.inf)
    # print(rssi_mat)
    # assert 0
    groundtruth_pos = np.load("./data/groundtruth_pos.npy")
    start_anchor = random.randint(0, rssi_mat.shape[0] - 1)
    start_pos = groundtruth_pos[start_anchor]
    dist_mat = np.vectorize(rssi_to_distance)(rssi_mat)
    mask_rssi = rssi_mat == 0
    dist_mat[mask_rssi] = np.inf
    shortest_path, predecessors = dijkstra(dist_mat, start_anchor)
    '''find ending anchors'''
    successors = [0] * len(predecessors)
    for pred in predecessors:
        successors[pred] += 1
    end_points = [i for i, x in enumerate(successors) if x == 0]
    if len(end_points) / len(successors) <= 0.2:
        end_anchors = end_points
    else:
        end_anchors = random.sample(end_points, (int)(0.2 * len(successors)))
    print("Anchors:", start_anchor, end_anchors)
    pos_valid = [True if (i in end_anchors or i == start_anchor) else False for i in range(len(successors))]
    
    while len(end_anchors) >= 2:
        sampled_end_anchors = random.sample(end_anchors, 2)
        end_anchor_0 = sampled_end_anchors[0]
        end_anchor_1 = sampled_end_anchors[1]

        path_0 = retrieve_path(predecessors, start_anchor, end_anchor_0)
        path_1 = retrieve_path(predecessors, start_anchor, end_anchor_1)
        all_points = list(set(path_0 + path_1))
        print(path_0, path_1)
        print(all_points)
        area_dist_mat = dist_mat[np.ix_(all_points, all_points)]
        area_rssi_mat = rssi_mat[np.ix_(all_points, all_points)]

        '''Iterative MDS'''
        coords = smacof_incomplete_rssi(area_rssi_mat)

        pred_end_anchor_0 = coords[all_points.index(end_anchor_0)]
        pred_end_anchor_1 = coords[all_points.index(end_anchor_1)] 
        pred_start_anchor = coords[all_points.index(start_anchor)]
        print(pred_end_anchor_0, pred_end_anchor_1, pred_start_anchor)
        preds = np.array([pred_end_anchor_0, pred_end_anchor_1, pred_start_anchor])
        truth = np.array([groundtruth_pos[end_anchor_0], groundtruth_pos[end_anchor_1], start_pos])
        plt.scatter([x[0] for x in truth], [x[1] for x in truth], label="truth")
        pred_mat = np.vstack([pred_end_anchor_0 - pred_start_anchor, pred_end_anchor_1 - pred_start_anchor])
        inverse = np.linalg.inv(pred_mat)
        t_ = np.matmul(np.vstack([groundtruth_pos[end_anchor_0] - start_pos, groundtruth_pos[end_anchor_1] - start_pos]), inverse)
        est_end_anchor_0 = np.matmul(t_, (pred_end_anchor_0 - pred_start_anchor)) + start_pos
        est_end_anchor_1 = np.matmul(t_, (pred_end_anchor_1 - pred_start_anchor)) + start_pos
        preds = np.array([est_end_anchor_0, est_end_anchor_1, pred_start_anchor])
        plt.scatter([x[0] for x in preds], [x[1] for x in preds], label="pred")
        for i, pos in enumerate(preds):
            plt.text(pos[0], pos[1], str(i), fontsize=12, ha='right', va='bottom')
        for i, pos in enumerate(truth):
            plt.text(pos[0], pos[1], str(i), fontsize=12, ha='right', va='bottom')
        plt.legend()
        plt.show()
        assert 0
        end_anchors.remove(end_anchor_0)
        end_anchors.remove(end_anchor_1)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-anc", help="the number of archors", default=10, type=int)
    run(arg_parser.parse_args())

if __name__ == "__main__":
    main()