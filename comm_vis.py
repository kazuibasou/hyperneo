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
plt.rcParams["font.size"] = fontsize
plt.rcParams['xtick.direction'] = 'in' 
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams['xtick.major.width'] = 2.5 
plt.rcParams['xtick.minor.width'] = 1 
plt.rcParams['ytick.major.width'] = 2.5 
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2.5 
plt.rcParams['xtick.major.size'] = 15
plt.rcParams['xtick.minor.size'] = 12.5
plt.rcParams['ytick.major.size'] = 15
plt.rcParams['ytick.minor.size'] = 12.5
plt.rcParams['hatch.linewidth'] = 0.3

data_dir = "./data/"

def read_settings():
    
    settings = json.load(open("settings.json", 'r'))

    return settings

def inferred_membership_matrix(G, data_name, settings, U, W):

    N, K = int(U.shape[0]), int(U.shape[1])
    label_name = settings[data_name]["label_name"]
    label_order = settings[data_name]["label_order"]
    community_order = settings[data_name]["community_order"]

    node_lst_by_label = {x: [] for x in range(0, G.Z)}
    for i in range(0, G.N):
        x = int(G.X[i])
        node_lst_by_label[x].append(i)

    node_lst = []
    for x in label_order:
        node_lst += node_lst_by_label[x]
        #print(len(node_lst_by_label[x]))

    U_sum = U.sum(axis=1)
    normalized_U = np.zeros((N, K), dtype=float)
    for i in range(0, G.N):
        normalized_U[i] = U[i] / U_sum[i]

    membership_matrix = np.zeros((G.N, K))
    for j in range(0, len(node_lst)):
        i = node_lst[j]
        membership_matrix[j] = [normalized_U[i][k] for k in community_order]

    yticks = []
    ylabels = []
    ypoints = []
    for i in range(0, len(label_order)):
        cumulative_count = sum([len(node_lst_by_label[label_order[j]]) for j in range(0, i)])
        ypoints.append(cumulative_count)
    ypoints.append(G.N)

    for i in range(0, G.N + 1):
        if i in ypoints:
            yticks.append(i)
            ylabels.append("")

    x_labels = ["k = " + str(i) for i in range(1, K+1)]
    y_labels = [str(label_name[i]) for i in label_order]

    yticks_ = []
    for i in range(0, len(label_order)):
        yticks_.append(float(ypoints[i] + ypoints[i+1]) / 2)

    fontsize = 30
    plt.rcParams["font.size"] = fontsize
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 0
    plt.rcParams['xtick.minor.width'] = 0
    plt.rcParams['ytick.major.width'] = 0
    plt.rcParams['ytick.minor.width'] = 0

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(membership_matrix, 
                     vmax=1.0, vmin=0.0, cmap='Reds', square=True, cbar_kws={"shrink": .8},
                     xticklabels=x_labels, yticklabels=y_labels)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax.set_aspect(membership_matrix.shape[1] / membership_matrix.shape[0])
    ax.set_xticks([0.5 + i for i in range(0, K)])
    ax.set_xticklabels(["k = " + str(i) for i in range(1, K+1)])
    #ax.set_yticks(yticks)
    ax.set_yticks(yticks_)
    ax.set_yticklabels(y_labels)
    for i in range(0, K):
        ax.axvline(x=i, linewidth=1, color="black")
    for i in ypoints:
        ax.axhline(y=i, linewidth=1, color="black")
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig("./figs/" + str(data_name) + "_inferred_membership_matrix.png")

    return

def inferred_affinity_matrix(G, data_name, settings, U, W):

    N, K = int(U.shape[0]), int(U.shape[1])
    label_order = settings[data_name]["label_order"]
    community_order = settings[data_name]["community_order"]

    node_lst_by_label = {x: [] for x in range(0, G.Z)}
    for i in range(0, G.N):
        x = int(G.X[i])
        node_lst_by_label[x].append(i)

    node_lst = []
    for x in label_order:
        node_lst += node_lst_by_label[x]
        #print(len(node_lst_by_label[x]))

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
    plt.rcParams['xtick.major.width'] = 0
    plt.rcParams['xtick.minor.width'] = 0
    plt.rcParams['ytick.major.width'] = 0
    plt.rcParams['ytick.minor.width'] = 0

    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(ordered_W, vmax=1.0, vmin=0.0, cmap='Reds',
                     #annot=True,
                     #fmt='.2f',
                     square=True, cbar_kws={"shrink": .8}, linewidths=0.5)
    ax.set_xticks([0.5 + i for i in range(0, K)])
    ax.set_xticklabels(["k = " + str(i) for i in range(1, K+1)])
    ax.set_yticks([0.5 + i for i in range(0, K)])
    ax.set_yticklabels(["k = " + str(i) for i in range(1, K+1)])
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
    random_state = int(settings["random_state"])

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
            'Node ID': ids,
            'UMAP1': x,
            'UMAP2': y,
            'Attribute': z,
        }
    )

    fig = px.scatter(df, x='UMAP1', y='UMAP2', hover_name='Node ID',
                     hover_data=['Node ID'],
                     color='Attribute',
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
        legend={'traceorder': 'normal'},
        showlegend=False,
    )

    fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
    fig.update_xaxes(gridcolor='lightgray', showline=True, linecolor='black', linewidth=2, zeroline=True, zerolinecolor='lightgray', zerolinewidth=1)
    fig.update_yaxes(gridcolor='lightgray', showline=True, linecolor='black', linewidth=2, zeroline=True, zerolinecolor='lightgray', zerolinewidth=1)

    fig.write_html("./figs/" + str(data_name) + "_node_layout_umap_" + str(metric) + ".html")
    fig.write_image("./figs/" + str(data_name) + "_node_layout_umap_" + str(metric) + ".pdf")
    fig.show()

    return
