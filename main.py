import hypergraph, hyperneo, comm_vis

if __name__ == '__main__':

    settings = comm_vis.read_settings()

    data_name = "workplace"
    #data_name = "hospital"
    #data_name = "contact-high-school"
    #data_name = "contact-primary-school"

    G = hypergraph.read_empirical_hypergraph_data(data_name, print_info=True)

    random_state = settings["random_state"]
    (K, gamma) = settings[data_name]["hyperparam"]

    model = hyperneo.HyperNEO(G, K, gamma, random_state=random_state)
    best_loglik, (U, W, Beta) = model.fit()

    label_name = settings[data_name]["label_name"]
    label_order = settings[data_name]["label_order"]
    community_order = settings[data_name]["community_order"]

    comm_vis.inferred_membership_matrix(G, data_name, U, label_name, label_order, community_order)

    comm_vis.inferred_affinity_matrix(G, data_name, W, community_order)

    comm_vis.node_layout(G, data_name, U, W, label_name, label_order, random_state, metric="euclidean", fig_show=False)

    comm_vis.node_layout(G, data_name, U, W, label_name, label_order, random_state, metric="cosine", fig_show=False)
