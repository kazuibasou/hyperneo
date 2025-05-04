import math
import random
import numpy as np
import networkx as nx
from collections import Counter
from scipy.special import binom, comb
import itertools
from collections import deque
from collections import defaultdict
import csv
import pickle
from itertools import combinations

#data_dir = "/Users/kazuki/lab/hyperneo/data/"
data_dir = "/Users/knakajima/research/hyperneo/data/"
#data_dir = "/Users/testuser/Desktop/hyperneo/data/"

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
        self.Z = Z
        self.E = [set() for _ in range(0, self.M)]
        self.A = np.zeros(self.M, dtype=int)
        self.X = np.zeros(N, dtype=int)

    def is_connected(self):
        vlist = defaultdict(list)
        elist = defaultdict(list)
        for m in range(0, self.M):
            for i in self.E[m]:
                vlist[m].append(i)
                elist[i].append(m)

        searched = {i: 0 for i in range(0, self.N)}
        nodes = set()
        v = 0

        Q = deque()
        searched[v] = 1
        Q.append(v)
        while len(Q) > 0:
            v = Q.popleft()
            nodes.add(v)
            for m in elist[v]:
                for w in vlist[m]:
                    if searched[w] == 0:
                        searched[w] = 1
                        Q.append(w)

        return self.N == sum(list(searched.values()))

    def calc_pairwise_shortest_path_length(self):
        vlist = defaultdict(list)
        elist = defaultdict(list)
        for m in range(0, self.M):
            for i in self.E[m]:
                vlist[m].append(i)
                elist[i].append(m)

        d = np.full((self.N, self.N), -1)

        for s in range(0, self.N):
            Q = deque()
            d[s][s] = 0
            Q.append(s)

            while len(Q) > 0:
                v = Q.popleft()
                for m in elist[v]:
                    for w in vlist[m]:
                        if d[s][w] < 0:
                            d[s][w] = d[s][v] + 1
                            Q.append(w)

        return d

def strict_modularity(G: HyperGraph, K, comm_label):
    comm_nodes = {c_i: set() for c_i in range(0, K)}
    for i in range(0, G.N):
        c_i = comm_label[i]
        comm_nodes[c_i].add(i)

    node_degree = np.zeros(G.N)
    num_hyperedges = np.zeros(G.N + 1)
    contributing_score = np.zeros(K)
    sum_of_node_degrees = np.zeros(K)
    D = 2
    for m in range(0, G.M):
        e = G.E[m]
        for i in e:
            node_degree[i] += G.A[m]
            c_i = comm_label[i]
            sum_of_node_degrees[c_i] += G.A[m]

        s = len(e)
        num_hyperedges[s] += G.A[m]

        if s > D:
            D = s

        for c_i in range(0, K):
            if set(e) <= set(comm_nodes[c_i]):
                contributing_score[c_i] += G.A[m]

    x = np.sum(contributing_score)
    y = np.sum([num_hyperedges[d] * np.sum([pow(float(sum_of_node_degrees[c_i]) / np.sum(sum_of_node_degrees), d) for c_i in range(0, K)]) for d in range(2, D+1)])
    h_mod = float(x - y) / np.sum(G.A)

    return h_mod

def read_empirical_hypergraph_data(data_name, print_info=True):
    if data_name in {"workplace", "hospital", "gene-disease", "house-committees", "senate-committees"}:
        G = read_nicolo_hypergraph_data(data_name, print_info)
    elif data_name in {"contact-primary-school", "contact-high-school"}:
        G = read_benson_hypergraph_data(data_name, print_info)
    else:
        print("ERROR: given data set is not defined.")
        exit()

    return G

def read_benson_hypergraph_data(data_name, print_info=True):
    N, E, A = 0, [], []
    f_path = data_dir + "emp/" + data_name + "/hyperedges-" + data_name + ".txt"
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

    f_path = data_dir + "emp/" + data_name + "/label-names-" + data_name + ".txt"
    with open(f_path, 'r') as f:
        Z = len(f.readlines())

    G = HyperGraph(N, M, Z)
    G.E = list(E)
    G.A = np.array(A)

    f_path = data_dir + "emp/" + data_name + "/node-labels-" + data_name + ".txt"
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

    count_by_attribute = {}
    for i in range(0, G.N):
        x = G.X[i]
        count_by_attribute[x] = count_by_attribute.get(x, 0) + 1

    if print_info:
        sum_degree = sum([len(G.E[m]) for m in range(0, G.M)])
        D = max([len(G.E[m]) for m in range(0, G.M)])
        count_by_size = {}
        for m in range(0, M):
            s = len(G.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Number of nodes: " + str(G.N))
        print("Number of hyperedges: " + str(G.M))
        print("Sum of hyperedge weights: " + str(np.sum(G.A)))
        print("Average degree of the node: " + str(float(sum_degree) / G.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / G.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(count_by_size))
        print("Number of distinct node lables: " + str(G.Z))
        print("Number of nodes by attribute: " + str(count_by_attribute))
        print("Connected hypergraph: " + str(G.is_connected()))
        print()

    return G

def read_nicolo_hypergraph_data(data_name, print_info=True):
    ori_data_name = {
        "workplace": "workspace1",
        "hospital": "hospital",
        "gene-disease": "curated_gene_disease_associations",
        "house-committees": "house-committees",
        "senate-committees": "senate-committees",
    }

    Z = {
        "workplace": 5,
        "hospital": 4,
        "gene-disease": 25,
        "house-committees": 2,
        "senate-committees": 2,
    }

    if data_name not in ori_data_name:
        print("Error: given data set is not found.")
        exit()

    f_path = data_dir + "emp/" + data_name + "/" + ori_data_name[data_name] + ".npz"
    npzfile = np.load(f_path, allow_pickle=True)
    A, B, hyperedges = npzfile['A'], npzfile['B'], npzfile['hyperedges']

    if len([e for e in hyperedges if len(e) < 2]) > 0:
        print("Error: hyperedges with |e| < 2 are included.")
        exit()

    N, M = int(B.shape[0]), len(hyperedges)

    G = HyperGraph(N, M, Z[data_name])
    G.E = [set(e) for e in hyperedges]
    G.A = np.array(A)

    f_path = data_dir + "emp/" + data_name + "/" + ori_data_name[data_name] + "_meta.csv"
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

    count_by_attribute = {}
    for i in range(0, G.N):
        x = G.X[i]
        count_by_attribute[x] = count_by_attribute.get(x, 0) + 1

    if print_info:
        sum_degree = sum([len(G.E[m]) for m in range(0, G.M)])
        D = max([len(G.E[m]) for m in range(0, G.M)])
        count_by_size = {}
        for m in range(0, M):
            s = len(G.E[m])
            count_by_size[s] = count_by_size.get(s, 0) + 1

        print("Number of nodes: " + str(G.N))
        print("Number of hyperedges: " + str(G.M))
        print("Sum of hyperedge weights: " + str(np.sum(G.A)))
        print("Average degree of the node: " + str(float(sum_degree) / G.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / G.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(count_by_size))
        print("Number of distinct node lables: " + str(G.Z))
        print("Number of nodes by attribute: " + str(count_by_attribute))
        print("Connected hypergraph: " + str(G.is_connected()))
        print()

    return G

def read_synthetic_hypergraph(p_u, w_out, w_in, D, M_N_ratio, p_beta, N, sample_no, print_info=False):
    dir_path = data_dir + "syn_hypergraphs/samples/p_u_" + p_u + "_w_in_" \
               + w_in + "_w_out_" + w_out + "_D_" + str(D) + "_M_N_ratio_" + str(M_N_ratio) + "_p_beta_" + p_beta + "_N_" + str(N)
    f_path = dir_path + "/sample_" + str(sample_no) + ".pkl"

    print(f_path)

    with open(f_path, mode="rb") as f:
        G_ = pickle.load(f)

    A, E = G_.get_repr()

    M = len(E)
    K = 2
    Z = 2
    G = HyperGraph(N, M, Z)
    G.N = N
    G.M = M
    G.Z = Z
    G.E = [set(e) for e in E]
    G.A = A
    count_by_size = {}
    for m in range(0, M):
        s = len(E[m])
        count_by_size[s] = count_by_size.get(s, 0) + 1
        G.A[m] = A[m]

    U = np.zeros((N, K))
    f_path = data_dir + "syn_hypergraphs/inputs/u_" + p_u + "_N_" + str(N) + ".txt"
    with open(f_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines)):
            U[i] = [float(u) for u in lines[i][:-1].split(" ")]

    Beta = np.zeros((K, Z))
    f_path = data_dir + "syn_hypergraphs/inputs/beta_" + p_beta + ".txt"
    with open(f_path, 'r') as f:
        lines = f.readlines()
        for k in range(0, len(lines)):
            Beta[k] = [float(p) for p in lines[k][:-1].split(" ")]

    G.X = np.zeros(N, dtype=int)
    for i in range(0, N):
        #ps = []
        #sum_uik = sum([U[i][k] for k in range(0, K)])
        #for z in range(0, Z):
        #    p = float(sum([U[i][k] * Beta[k][z] for k in range(0, K)])) / sum_uik
        #    ps.append(p)
        #z = int(random.choices(range(0, Z), weights=ps, k=1)[0])
        #G.X[i] = z

        #print(i, array_equal(np.array(U[i]), np.array([float(p_u), 1 - float(p_u)])), array_equal(np.array(U[i]), np.array([1 - float(p_u), float(p_u)])))
        G.X[i] = int(random.choices(range(0, Z), weights=U[i], k=1)[0])

    #print("read a synthetic hypergraph.")

    if print_info:
        sum_degree = sum([len(G.E[m]) for m in range(0, G.M)])
        D = max([len(G.E[m]) for m in range(0, G.M)])

        print("Dataset name: syn" + str(N))
        print("Number of nodes: " + str(G.N))
        print("Number of hyperedges: " + str(G.M))
        print("Average degree of the node: " + str(float(sum_degree) / G.N))
        print("Average size of the hyperedge: " + str(float(sum_degree) / G.M))
        print("Maximum size of the hyperedge: " + str(D))
        print("Hyperedge size distribution: " + str(count_by_size))
        print("Number of different node lables: " + str(Z) + "\n")

    return G

def generate_uniform_hsbm(N, k, K, alpha, beta, p_att):

    print(N, k, K, alpha, beta, p_att)

    community_sizes = []
    for i in range(0, K-1):
        community_sizes.append(int(N/K))
    community_sizes.append(N - sum(community_sizes))

    if p_att < 0 or p_att > 1:
        print("ERROR: p_att must be at least 0 and at most 1.0.")
        exit()

    #print(community_sizes)

    p = float(alpha*math.log(N)) / binom(N-1, k-1)
    q = float(beta*math.log(N)) / binom(N-1, k-1)

    n = 0
    community = np.zeros(N, dtype=int)
    for i in range(0, K):
        for j in range(n, n+community_sizes[i]):
            community[j] = i
        n += community_sizes[i]

    rng = np.random.default_rng()
    random_lst = rng.random(int(binom(N, k)))

    E, A = [], []
    ii = 0
    for e in itertools.combinations(sorted(range(0, N)), k):
        comm_labels = set([community[i] for i in e])
        p_ = random_lst[ii]
        if len(comm_labels) == 1:
            flag = p_ <= p
        else:
            flag = p_ <= q

        ii += 1

        if not flag:
            continue

        if e not in E:
            E.append(e)
            A.append(1)
        else:
            m = E.index(e)
            A[m] += 1

    node_index = {}
    E_, A_, community_ = [], [], []
    for e in E:
        e_ = set()
        for v in e:
            if v not in node_index:
                node_index[v] = len(node_index)
                community_.append(community[v])
            e_.add(node_index[v])

        if e_ not in E_:
            E_.append(e_)
            A_.append(1)
        else:
            m = E_.index(e_)
            A_[m] += 1

    N_ = len(node_index)
    M_ = len(E_)
    X_ = np.zeros(N_, dtype=int)
    node_attributes = set()
    for i in range(0, len(community_)):
        if random.random() <= p_att:
            X_[i] = community_[i]
        else:
            X_[i] = np.random.randint(0, K)
        node_attributes.add(X_[i])

    Z_ = 2
    G = HyperGraph(N_, M_, Z_)
    G.E = np.array(E_)
    G.A = np.array(A_)
    G.X = np.array(X_, dtype=int)

    print(G.N, G.M, G.Z)

    return G, community_

def generate_hymmsbm(N, D, K, p, w_in, w_out=0.01):

    print(N, D, K, p, w_in)

    community_sizes = []
    for i in range(0, K - 1):
        community_sizes.append(int(N / K))
    community_sizes.append(N - sum(community_sizes))

    if p < 0.5 or p > 1:
        print("ERROR: mu must be between 0.5 and 1.0.")
        exit()

    U_ = [
        [p, 1-p],
        [1-p, p],
    ]

    n = 0
    U = np.zeros((N, K), dtype=float)
    for i in range(0, K):
        for j in range(n, n + community_sizes[i]):
            U[j] = U_[i]
        n += community_sizes[i]

    W = np.full((K, K), w_out, dtype=float)
    for i in range(0, K):
        W[i][i] = w_in

    E, A = [], []
    for k in range(2, D+1):
        for e in itertools.combinations(sorted(range(0, N)), k):
            s = len(e)
            kappa_e = (float(s*(s-1))/2) * binom(N-2, s-2)
            lambda_e = float(sum([(U[i] * W * U[j].T).sum() for (i, j) in list(combinations(sorted(e), 2))]))
            A_e = np.random.poisson(lam=float(lambda_e)/kappa_e, size=1)[0]

            if A_e > 0:
                E.append(e)
                A.append(A_e)

    node_index = {}
    E_, A_, X_, community_ = [], [], [], []
    for e in E:
        e_ = set()
        for v in e:
            if v not in node_index:
                node_index[v] = len(node_index)
                X_.append(np.random.choice(a=range(0, K), size=1, p=U[v])[0])
                if np.all(U[v] == U_[0]):
                    community_.append(0)
                else:
                    community_.append(1)
            e_.add(node_index[v])

        if e_ not in E_:
            E_.append(e_)
            A_.append(1)
        else:
            m = E_.index(e_)
            A_[m] += 1

    N_ = len(node_index)
    M_ = len(E_)
    Z_ = 2
    G = HyperGraph(N_, M_, Z_)
    G.E = np.array(E_)
    G.A = np.array(A_)
    G.X = np.array(X_, dtype=int)

    U_ = np.zeros((G.N, K), dtype=float)
    for v in node_index:
        i = node_index[v]
        U_[i] = U[v]

    print(G.N, G.M, G.Z)

    return G, community_