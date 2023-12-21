import csv
import numpy as np

data_dir = "./data/"

class HyperGraph():
    N = 0
    M = 0
    Z = 0
    E = []
    A = np.zeros(M, dtype=int)
    X = np.zeros(N, dtype=int)

    def __init__(self, N, M, Z):
        self.N = N
        self.M = M
        self.E = [set() for _ in range(0, self.M)]
        self.A = np.zeros(self.M, dtype=int)
        self.Z = Z
        self.X = np.zeros(N, dtype=int)

def read_empirical_hypergraph_data(data_name, print_info=True):
    if data_name in {"workplace", "hospital"}:
        G = read_nicolo_hypergraph_data(data_name, print_info)
    elif data_name in {"contact-primary-school", "contact-high-school"}:
        G = read_benson_hypergraph_data(data_name, print_info)
    else:
        print("ERROR: given data set is not defined.")
        exit()

    return G

def read_benson_hypergraph_data(data_name, print_info=True):
    N, E, A = 0, [], []
    f_path = data_dir + data_name + "/hyperedges-" + data_name + ".txt"
    with open(f_path, 'r') as f:
        for line in f:
            e = set([int(i)-1 for i in list(str(line[:-1]).split(","))])
            if len(e) < 2:
                continue
            if e not in E:
                E.append(e)
                A.append(1)
            else:
                m = E.index(e)
                A[m] += 1
            for i in e:
                if i+1 > N:
                    N = i+1

    M = len(E)

    f_path = data_dir + data_name + "/label-names-" + data_name + ".txt"
    with open(f_path, 'r') as f:
        Z = len(f.readlines())

    G = HyperGraph(N, M, Z)
    G.E = list(E)
    G.A = np.array(A)

    f_path = data_dir + data_name + "/node-labels-" + data_name + ".txt"
    valid_labels = set()
    with open(f_path, 'r') as f:
        i = 0
        for line in f:
            z = int(line[:-1]) - 1
            G.X[i] = z
            i += 1
            valid_labels.add(z)

    if set(valid_labels) != set(range(0, Z)):
        print("Error: node labels are not valid.")
        print(set(valid_labels), set(range(0, Z)))
        exit()

    sum_degree = sum([len(G.E[m]) for m in range(0, G.M)])
    D = max([len(G.E[m]) for m in range(0, G.M)])

    if print_info:
        print("Number of nodes: " + str(G.N))
        print("Number of hyperedges: " + str(G.M))
        print("Average degree of the node: " + str(float(sum_degree) / G.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / G.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Number of different node lables: " + str(Z) + "\n")

    return G

def read_nicolo_hypergraph_data(data_name, print_info=True):
    ori_data_name = {
        "workplace": "workspace1",
        "hospital": "hospital",
        "gene-disease": "curated_gene_disease_associations",
    }

    Z = {
        "workplace": 5,
        "hospital": 4,
        "gene-disease": 25,
    }

    if data_name not in ori_data_name:
        print("Error: given data set is not found.")
        exit()

    f_path = data_dir + data_name + "/" + ori_data_name[data_name] + ".npz"
    npzfile = np.load(f_path, allow_pickle=True)
    A, B, hyperedges = npzfile['A'], npzfile['B'], npzfile['hyperedges']

    if len([e for e in hyperedges if len(e) < 2]) > 0:
        print("Error: hyperedges with |e| < 2 are included.")
        exit()

    N, M = int(B.shape[0]), len(hyperedges)

    G = HyperGraph(N, M, Z[data_name])
    G.E = [set(e) for e in hyperedges]
    G.A = np.array(A)

    f_path = data_dir + data_name + "/" + ori_data_name[data_name] + "_meta.csv"
    valid_labels = set()
    with open(f_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            i, z = int(row[0]), int(row[3])
            G.X[i] = z
            valid_labels.add(z)

    if set(valid_labels) != set(range(0, Z[data_name])):
        print("Error: node labels are not valid.")
        print(set(valid_labels), set(range(0, Z[data_name])))
        exit()

    sum_degree = sum([len(G.E[m]) for m in range(0, G.M)])
    D = max([len(G.E[m]) for m in range(0, G.M)])

    if print_info:
        print("Number of nodes: " + str(G.N))
        print("Number of hyperedges: " + str(G.M))
        print("Average degree of the node: " + str(float(sum_degree) / G.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / G.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Number of different node lables: " + str(Z[data_name]) + "\n")

    return G
