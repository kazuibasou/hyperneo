import numpy as np
import hyperneo
import hypergraph
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
import plotly.express as px
from scipy.special import binom
import umap
from itertools import combinations
import warnings
import json

warnings.simplefilter('ignore')

fontsize = 30
#fp = FontProperties(fname=r'/Users/kazuki/PycharmProjects/lab/IPAfont00303/ipagp.ttf',size=fontsize)
plt.rcParams["font.size"] = fontsize
#plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['xtick.direction'] = 'in' #x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in' #y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 2.5 #x軸主目盛り線の線幅
plt.rcParams['xtick.minor.width'] = 1 #x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 2.5 #y軸主目盛り線の線幅
plt.rcParams['ytick.minor.width'] = 1 #y軸主目盛り線の線幅
plt.rcParams['axes.linewidth'] = 2.5 #軸の線幅edge linewidth。囲みの太さ
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['xtick.minor.size'] = 12.5
plt.rcParams['ytick.major.size'] = 15
plt.rcParams['ytick.minor.size'] = 12.5
plt.rcParams['hatch.linewidth'] = 0.3

data_dir = "./data/"

def read_settings():
    
    settings = json.load(open("settings.json", 'r'))

    return settings

def compute_best_comm_order(U_ref, U, K, eps=1e-10):
    N1, N2 = U_ref.shape[0], U.shape[0]
    K1, K2 = U_ref.shape[1], U.shape[1]
    if N1 != N2 or K1 != K2:
        print("Error: given matrices do not have the same shape.")
        print(N1, N2)
        exit()

    community_order_lst = list(itertools.permutations(range(0, K), r=K))
    highest_sim, best_comm_order = float("-inf"), []
    for comm_order in community_order_lst:
        num, den = 0, 0
        for i in range(0, N1):
            U_i = np.array([U[i][k] for k in list(comm_order)])
            num += float(np.dot(U_ref[i], U_i)) / max((np.linalg.norm(U_ref[i]) * np.linalg.norm(U_i)), eps)
            den += 1
        cs = float(num) / den
        if cs > highest_sim:
            highest_sim = cs
            best_comm_order = comm_order

    return highest_sim, best_comm_order

def inferred_membership_and_affinity_matrices(G, data_name, settings, U, W):

    K = int(U.shape[1])
    label_order = settings[data_name]["label_order"]

    node_lst_by_label = {x: [] for x in range(0, G.Z)}
    for i in range(0, G.N):
        x = int(G.X[i])
        node_lst_by_label[x].append(i)

    node_lst = []
    for x in label_order:
        node_lst += node_lst_by_label[x]
        #print(len(node_lst_by_label[x]))

    U_sum = U.sum(axis=1)
    for i in range(0, G.N):
        U[i] /= U_sum[i]

    node_propensity = np.zeros((G.Z, K))
    for i in range(0, G.N):
        z = int(G.X[i])
        node_propensity[z] += U[i]

    community_order = []
    for i in range(0, min(G.Z, K)):
        z = label_order[i]
        k_lst = sorted([(k, node_propensity[z][k]) for k in range(0, K) if k not in community_order], reverse=True, key=lambda x: x[1])
        k_ = int(k_lst[0][0])
        community_order.append(k_)

    membership_matrix = np.zeros((G.N, K))
    for j in range(0, len(node_lst)):
        i = node_lst[j]
        membership_matrix[j] = [U[i][k] for k in community_order]

    yticks = []
    ylabels = []
    ypoints = []
    for i in range(0, len(label_order)):
        cumulative_count = sum([len(node_lst_by_label[label_order[j]]) for j in range(0, i)])
        ypoints.append(cumulative_count)

    for i in range(0, G.N + 1):
        if i in ypoints:
            yticks.append(i)
            ylabels.append("")

    fontsize = 30
    plt.rcParams["font.size"] = fontsize
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 0
    plt.rcParams['xtick.minor.width'] = 0
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 0

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(membership_matrix, vmax=1.0, vmin=0.0, cmap='Reds', square=True, cbar_kws={"shrink": .8})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax.set_aspect(membership_matrix.shape[1] / membership_matrix.shape[0])
    ax.set_xticks(list(range(0, K)))
    ax.set_xticklabels(["" for _ in range(0, K)])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    for i in range(0, K):
        ax.axvline(x=i, linewidth=1, color="black")
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig("./figs/" + str(data_name) + "_inferred_membership_matrix.png")

    ordered_W = np.zeros((K, K))
    for i in range(0, len(community_order)):
        k1 = community_order[i]
        for j in range(0, len(community_order)):
            k2 = community_order[j]
            ordered_W[i][j] = W[k1][k2]

    ordered_W /= np.max(W)

    fontsize = 20
    plt.rcParams["font.size"] = fontsize
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 0
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 0

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(ordered_W, vmax=1.0, vmin=0.0, cmap='Reds',
                     #annot=True,
                     #fmt='.2f',
                     square=True, cbar_kws={"shrink": .8}, linewidths=0.5)
    ax.set_xticks(list(range(0, K)))
    ax.set_xticklabels(["" for _ in range(0, K)])
    ax.set_yticks(list(range(0, K)))
    ax.set_yticklabels(["" for _ in range(0, K)])
    for i in range(0, K):
        ax.axvline(x=i, linewidth=1, color="black")
        ax.axhline(y=i, linewidth=1, color="black")
    #cbar_kws = {"shrink": .82}
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig("./figs/" + str(data_name) + "_inferred_affinity_matrix.png")

    return

def node_layout(G, data_name, settings, U, W, metric):
    K = int(U.shape[1])
    label_name = settings[data_name]["label_name"]
    label_order = settings[data_name]["label_order"]
    random_state = settings["random_state"]

    node_lst_by_label = {x: [] for x in range(0, G.Z)}
    num_nodes_by_label_ = {x: 0 for x in range(0, G.Z)}
    for i in range(0, G.N):
        x = int(G.X[i])
        node_lst_by_label[x].append(i)
        num_nodes_by_label_[x] += 1

    node_lst = []
    for x in label_order:
        node_lst += node_lst_by_label[x]
        #print(len(node_lst_by_label[x]))

    node_index = {}
    for i in range(0, len(node_lst)):
        v = node_lst[i]
        node_index[v] = i

    color_lst = [
        "royalblue",
        "orange",
        "green",
        "red",
        "purple",
        "brown",
        "violet",
        "dimgray",
        "lightgray",
        "olive",
        "darkturquoise",
    ]

    learned_mat = np.zeros((G.N, G.N), dtype=float)
    E = []
    node_degree = {i: 0 for i in range(0, G.N)}
    for m in range(0, G.M):
        s = len(G.E[m])
        norm = (float(s*(s-1))/2) * binom(G.N-2, s-2)
        exp = float(sum([(U[i] * W * U[j].T).sum() for (i, j) in list(combinations(G.E[m], 2))])) / norm
        for (i, j) in list(combinations(G.E[m], 2)):
            learned_mat[i][j] += exp
            learned_mat[j][i] += exp
            if (i, j) not in E:
                E.append((i, j))
        for i in G.E[m]:
            node_degree[i] += 1

    average_node_degree = float(np.average(list(node_degree.values())))
    #print(random_state, average_node_degree, metric)

    mapper = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=int(average_node_degree), min_dist=0.1, metric=metric)
    embedded_mat = mapper.fit_transform(learned_mat)

    node_lst_by_label = {x: [] for x in range(0, G.Z)}
    for i in range(0, G.N):
        x = int(G.X[i])
        node_lst_by_label[x].append(i)

    x, y, z, ids = [], [], [], []
    for label in label_order:
        for i in node_lst_by_label[label]:
            ids.append(i)
            x.append(float(embedded_mat[i][0]))
            y.append(float(embedded_mat[i][1]))
            z.append(str(label_name[int(G.X[i])]))

    label_name_lst = label_name
    c_map = {str(label_name_lst[i]): str(color_lst[i]) for i in range(0, len(label_order))}

    df = pd.DataFrame(
        data={
            'Id': ids,
            'UMAP1': x,
            'UMAP2': y,
            # 'Dim 3': z,
            'color': z,
        }
    )

    fig = px.scatter(df, x='UMAP1', y='UMAP2', hover_name="Id",
                     hover_data=['Id'],
                     color='color',
                     #symbol=symbol,
                     color_discrete_map= c_map,
                     )

    fig.update_layout(
        font=dict(
            family="Arial",
            size=20,
            color="black"
        ),
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="white",
        # legend={'traceorder': 'normal'},
        showlegend=False,
    )

    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
    fig.update_xaxes(gridcolor='lightgray', showline=True, linecolor='black', linewidth=2, zeroline=True, zerolinecolor='lightgray', zerolinewidth=1)
    fig.update_yaxes(gridcolor='lightgray', showline=True, linecolor='black', linewidth=2, zeroline=True, zerolinecolor='lightgray', zerolinewidth=1)

    fig.write_html("./figs/" + str(data_name) + "_node_layout_umap_" + str(metric) + ".html")
    fig.write_image("./figs/" + str(data_name) + "_node_layout_umap_" + str(metric) + ".pdf")

    return

if __name__ == '__main__':

    data_name = "workplace"
    #data_name = "hospital"
    #data_name = "contact-high-school"
    #data_name = "contact-primary-school"

    print("Data: " + data_name + " hypergraph.")

    if data_name in {"workplace", "hospital"}:
        G = hypergraph.read_nicolo_hypergraph_data(data_name, True)
    elif data_name in {"contact-primary-school", "contact-high-school"}:
        G = hypergraph.read_benson_hypergraph_data(data_name, True)
    else:
        print("ERROR: given data set is not defined.")
        exit()

    setting = {}

    # Inference
    K = int(param[0])
    gamma = float(param[1])
    model = hyperneo.HyperNEO(G, K, gamma, random_state=random_state)
    best_loglik, (U, W, Beta) = model.fit()

    # Visualize inference results
    visualize_inferred_membership_and_affinity_matrices(G, data_name, label_order, U, W)
    visualize_node_layout_using_hyperneo(G, data_name, label_name, label_order, U, W, metric="euclidean")
    visualize_node_layout_using_hyperneo(G, data_name, label_name, label_order, U, W, metric="cosine")