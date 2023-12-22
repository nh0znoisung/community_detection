import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tqdm import tqdm
import os


class SNMF():
    def __init__(
        self,
        adj_matrix: np.ndarray,
        adj_list: np.ndarray = None,
        dimensions: int = 32,
        iterations: int = 600,
        seed: int = 42
    ):
        self.adj_matrix = adj_matrix
        self.adj_list = adj_list
        self.num_node = adj_matrix.shape[0]
        self.num_edge = adj_list.shape[0]
        self.dimensions = dimensions
        self.iterations = iterations
        self.seed = seed

    # def __del__(self):
    #     np.random.seedseq([]) # Reset the RNG


    def modularity(self, adj_matrix, communities: np.ndarray):
        assert adj_matrix.shape[0] == communities.shape[0]
        assert len(communities.shape) == 1
        assert adj_matrix.shape[0] == adj_matrix.shape[1]
        n = adj_matrix.shape[0]
        m = np.sum(adj_matrix) / 2  # number of edges
        k = np.sum(adj_matrix, axis=1) # degree of each node

        Q = 0
        for i in range(n):
            for j in range(n):
                sig = 1 if communities[i] == communities[j] else 0
                Q += (adj_matrix[i][j] - (k[i] * k[j] / (2*m))) * sig / (2*m)

        return Q
    
    def _set_seed(self):
        np.random.seed(self.seed)

    def update(self, G,X):
        numerator = G @ X # Top
        denominator = 2 * X @ X.T @ X  # Bottom
        newX = X * (1/2 + (numerator / denominator))
        return newX

    def get_loss(self, G,X):
        tar = G - X @ X.T
        loss = np.linalg.norm(tar) ** 2
        return loss


    def fit(self, dim: int = 32, clear_stats: bool = False):
        self._set_seed()

        X = np.random.rand(self.num_node, dim) # Start
        losses = []

        os.makedirs(f'results/snmf/runs_{dim}', exist_ok=True)
        print(f"Directory results/snmf/runs_{dim} created successfully!")

        # Iteration
        start_time = time.time()
        for i in tqdm(range(self.iterations)):
            # print(f'>> Iter {i}')
            X = self.update(self.adj_matrix,X)
            loss = self.get_loss(self.adj_matrix,X)

            if i > 0 and losses[-1] - loss < 1e-8:
                print(f'>> Early stopping at iteration {i}')
                break

            losses.append(loss)
        elapsed_time = time.time() - start_time
        
        print('Done iteration!!!')

        # Loss-epochs
        losses_df = pd.DataFrame({"epoch": range(len(losses)), "loss": losses})
        losses_df.to_csv(f'results/snmf/runs_{dim}/losses.csv', index=False)
        print("Done saved losses CSV")
        
        # Save images 
        plt.plot(losses)
        plt.xlabel("Number of epochs")  # Adjust as needed
        plt.ylabel("Loss")
        plt.title(f"Loss of SNMF over time with {dim} communities")
        plt.savefig(f'results/snmf/runs_{dim}/losses.png')
        print('Done saved losses image')

        # Save X
        np.savetxt(f'results/snmf/runs_{dim}/X.txt', X)
        print('Done saved X')

        # Save probabilities
        normalizedX = X / np.sum(X, axis=1, keepdims=True)
        np.savetxt(f'results/snmf/runs_{dim}/prob.txt', normalizedX, fmt='%.2f')
        print('Done saved Probabilities')


        # Community
        communities = np.argmax(X, axis=1)
        np.savetxt(f'results/snmf/runs_{dim}/communities.txt', communities, fmt='%d')
        print('Done saved Community')

        
        # Log lại


        # num_communities, modularity, epoch_stop, elapsed_time
        stats_path = f"results/snmf/stats.csv"

        if clear_stats:
            os.remove(stats_path)

        # Check if the file exists
        if not os.path.exists(stats_path):
            stats_df = pd.DataFrame(columns=["num_communities", "modularity", "losses", "epoch_stop", "elapsed_time"])
        else:
            stats_df = pd.read_csv(stats_path)

        new_row = {
            "num_communities": dim,
            "modularity": self.modularity(self.adj_matrix, communities),
            "losses": losses[-1],
            "epoch_stop": i,
            "elapsed_time": elapsed_time
        }
        new_df = pd.DataFrame([new_row])
        stats_df = pd.concat([stats_df, new_df], ignore_index=True)

        stats_df.to_csv(stats_path, index=False)
        print('Done save general stats')

    def fit_all(self, low: int, high: int):
        for comm in range(low, high + 1):
            print(f'>>> Community {comm} >>>')
            self.fit(comm)


with open("comm_output.txt", "r") as file:
    content = file.read()

list_of_lists = []
for block in content.split("[")[1:]:  # Split by "[" and discard the first empty string
    inner_list = [int(num) for num in block.split("]")[0].split(",")]  # Remove trailing "]" and split by commas
    list_of_lists.append(inner_list)



communities = np.zeros((4039,)) 

for idx, value in enumerate(list_of_lists):
    for v in value:
        communities[v] = idx

print(communities)

def load_adj_matrix(filename: str):
    """
     file:  dimacs10-football/out.dimacs10-football" or "facebook_combined.txt"
     Undirected graph
    """
    with open(filename, "r") as file:
        adj_list = np.array([tuple(map(int, line.split())) for line in file])

    max_node = max(max(edge) for edge in adj_list) + 1
    adj_matrix = np.zeros((max_node, max_node))

    for node1, node2 in adj_list:
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1

    print('>> Max node:', max_node)
    print('>> Adj_matrix: \n', adj_matrix)
    return adj_matrix, adj_list

G, adj_list = load_adj_matrix("facebook_combined.txt")

snmf = SNMF(G, adj_list)
print(snmf.modularity(G,communities))
