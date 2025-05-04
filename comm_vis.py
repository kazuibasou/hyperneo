import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from scipy.special import binom
import umap
from itertools import combinations
import json
from sklearn.manifold import TSNE
from matplotlib.colors import LogNorm
from plotly.validators.scatter.marker import SymbolValidator
from sklearn.decomposition import PCA
import trimap, pacmap
import scipy

#warnings.simplefilter('ignore')

fontsize = 30
plt.rcParams["font.size"] = fontsize
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
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

def plot_inferred_membership_matrix(G, data_name, U, label_name, label_order, community_order, model_name):

    N, K = int(U.shape[0]), int(U.shape[1])

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
        membership_matrix[j] = [U[i][k] for k in community_order]

    ypoints = []
    for i in range(0, len(label_order)):
        cumulative_count = sum([len(node_lst_by_label[label_order[j]]) for j in range(0, i)])
        ypoints.append(cumulative_count)
    ypoints.append(G.N)

    minor_yticks = []
    minor_ylabels = []
    for i in range(0, G.N + 1):
        if i in ypoints:
            minor_yticks.append(i)
            minor_ylabels.append("")

    x_labels = ["k = " + str(k) for k in community_order]
    major_ylabels = [str(label_name[i]) for i in label_order]

    major_yticks = []
    for i in range(0, len(label_order)):
        major_yticks.append(float(ypoints[i] + ypoints[i+1]) / 2)

    fontsize = 30
    plt.rcParams["font.size"] = fontsize
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 0
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 0
    plt.rcParams['ytick.minor.width'] = 1.5
    plt.rcParams['axes.linewidth'] = 3.5

    plt.figure(figsize=(12.5, 10))
    ax = sns.heatmap(membership_matrix,
                     #vmax=1.0,
                     vmin=0.0,
                     cmap='Reds', square=True, cbar_kws={"shrink": .8},
                     #norm=LogNorm()
                     )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax.set_aspect(membership_matrix.shape[1] / membership_matrix.shape[0])
    ax.set_xticks(ticks=[i for i in range(0, K+1)], labels=["" for _ in range(0, K+1)], minor=True)
    ax.set_xticks(ticks=[0.5 + i for i in range(0, K)], labels=["k = " + str(k) for k in community_order], minor=False)
    ax.set_yticks(ticks=minor_yticks, labels=minor_ylabels, minor=True)
    ax.set_yticks(ticks=major_yticks, labels=major_ylabels, minor=False)
    ax.tick_params(axis='both', length=1.5)
    for i in range(0, K):
        ax.axvline(x=i, linewidth=1, color="black")
    for i in ypoints:
        ax.axhline(y=i, linewidth=1, color="black")
    #plt.subplots_adjust(left=0.2, right=0.95, bottom=0.12, top=0.95)
    plt.tight_layout()
    plt.savefig("./figs/" + str(data_name) + "_" + str(model_name) + "_inferred_membership_matrix.png")
    plt.show()

    return

def plot_inferred_affinity_matrix(G, data_name, W, community_order, model_name):

    N, K = int(G.N), int(W.shape[0])

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
    ax.set_xticklabels(["k = " + str(k) for k in community_order])
    ax.set_yticks([0.5 + i for i in range(0, K)])
    ax.set_yticklabels(["k = " + str(k) for k in community_order])
    for i in range(0, K):
        ax.axvline(x=i, linewidth=1, color="black")
        ax.axhline(y=i, linewidth=1, color="black")
    #cbar_kws = {"shrink": .82}
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig("./figs/" + str(data_name) + "_" + str(model_name) + "_inferred_affinity_matrix.png")

    return

def plot_inferred_attribute_correlation_matrix(data_name, beta, label_name, label_order, community_order, model_name):

    K, Z = int(beta.shape[0]), int(beta.shape[1])

    ordered_beta = np.zeros((K, Z))
    for i in range(0, len(community_order)):
        k = community_order[i]
        for j in range(0, len(label_order)):
            z = label_order[j]
            ordered_beta[i][j] = beta[k][z]

    fontsize = 30
    plt.rcParams["font.size"] = fontsize
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['xtick.major.width'] = 0
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 0
    plt.rcParams['ytick.minor.width'] = 1.5
    plt.rcParams['axes.linewidth'] = 3.5

    plt.figure(figsize=(12.5, 10))
    ax = sns.heatmap(ordered_beta, vmax=1.0, vmin=0.0, cmap='Reds',
                     annot=True,
                     fmt='.2f',
                     square=True, cbar_kws={"shrink": .8},
                     linewidths=0.5,
                     annot_kws={"fontsize": 25},
                     )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax.set_aspect(ordered_beta.shape[1] / ordered_beta.shape[0])
    ax.set_yticks([0.5 + i for i in range(0, K)])
    ax.set_yticklabels(["k = " + str(k) for k in community_order], minor=False)
    ax.set_xticks([0.5 + i for i in range(0, Z)])
    ax.set_xticklabels([label_name[i] for i in label_order], minor=False)
    for i in range(0, K):
        ax.axvline(x=i, linewidth=1, color="black")
    for i in range(0, Z):
        ax.axhline(y=i, linewidth=1, color="black")
    #cbar_kws = {"shrink": .82}
    #plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.xticks(rotation=90)
    plt.yticks(rotation=360)
    plt.tight_layout()
    plt.savefig("./figs/" + str(data_name) + "_" + str(model_name) + "_inferred_attribute_correlation_matrix.png")

    return

def node_embedding(G, data_name, U, W, label_name, label_order, random_state, metric, model_name, dim_red,
                   fig_show=True, pca_dim=30):

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

    symbol_lst = [
        "o",
        "x",
        "+",
        "v",
        "<",
        ">",
        "^",
        "s",
        "*",
        "D",
    ]

    raw_symbols = SymbolValidator().values

    learned_mat = np.zeros((G.N, G.N), dtype=float)
    E = []
    node_degree = {i: 0 for i in range(0, G.N)}
    for m in range(0, G.M):
        s = len(G.E[m])
        norm = (float(s * (s - 1)) / 2) * binom(G.N - 2, s - 2)
        #norm = (float(s) / 2) * binom(G.N - 2, s - 2)
        #norm = float(s-1)
        #norm = float(s * (s - 1)) / 2
        exp = float(sum([(U[i] * W * U[j].T).sum() for (i, j) in list(combinations(G.E[m], 2))])) / norm
        for (i, j) in list(combinations(G.E[m], 2)):
            learned_mat[i][j] += exp
            learned_mat[j][i] += exp
            if (i, j) not in E:
                E.append((i, j))
        for i in G.E[m]:
            node_degree[i] += 1

    adj_mat = np.zeros((G.N, G.N), dtype=float)
    for m in range(0, G.M):
        for (i, j) in list(combinations(sorted(list(G.E[m])), 2)):
            adj_mat[i][j] += 1
            adj_mat[j][i] += 1

    average_num_neighbors = float(np.sum([np.count_nonzero(adj_mat[i]) for i in range(0, G.N)])) / G.N

    mapper = PCA(n_components=pca_dim, svd_solver='full')
    learned_mat = mapper.fit_transform(learned_mat)

    if dim_red == 't-SNE':
        mapper = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=int(average_num_neighbors),
                      max_iter=1000, random_state=random_state, metric=metric)
    elif dim_red == 'UMAP':
        mapper = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=int(average_num_neighbors),
                           min_dist=0.1, metric=metric, n_jobs=1)
    elif dim_red == 'triMAP':
        mapper = trimap.TRIMAP(n_dims=2, n_inliers=int(average_num_neighbors), distance=metric)
    elif dim_red == 'PaCMAP':
        mapper = pacmap.PaCMAP(n_components=2, n_neighbors=int(average_num_neighbors))
    else:
        print("ERROR: specified dimensionality reduction method is not supported.")
        exit()

    embedded_mat = mapper.fit_transform(learned_mat)

    node_lst_by_label = {x: [] for x in range(0, G.Z)}
    for i in range(0, G.N):
        x = int(G.X[i])
        node_lst_by_label[x].append(i)

    x, y, z, ids, s = [], [], [], [], []
    for label in label_order:
        for i in node_lst_by_label[label]:
            ids.append(i)
            x.append(float(embedded_mat[i][0]))
            y.append(float(embedded_mat[i][1]))
            z.append(str(label_name[int(G.X[i])]))
            s.append(raw_symbols[int(G.X[i])])

    label_name_lst = label_name
    c_map = {str(label_name_lst[i]): str(color_lst[i]) for i in range(0, len(label_order))}

    hori_axis = dim_red + "1"
    ver_axis = dim_red + "2"

    df = pd.DataFrame(
        data={
            'Node ID': ids,
            hori_axis: x,
            ver_axis: y,
            'Attribute': z,
            'symbol': s,
        }
    )

    fig = px.scatter(df, x=hori_axis, y=ver_axis, hover_name='Node ID',
                     hover_data=['Node ID'],
                     color='Attribute',
                     symbol='symbol',
                     color_discrete_map=c_map,
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

    fig.write_html("./figs/" + str(data_name) + "_" + str(model_name) + "_node_embedding_" + dim_red.lower() + "_" + str(metric) + ".html")
    fig.write_image("./figs/" + str(data_name) + "_" + str(model_name) + "_node_embedding_" + dim_red.lower() + "_" + str(metric) + ".pdf")
    if fig_show:
        fig.show(renderer="png")

    return embedded_mat
