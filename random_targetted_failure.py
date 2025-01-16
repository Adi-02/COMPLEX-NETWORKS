import networkx as nx
import random
import matplotlib.pyplot as plt

def random_failures(graph_file, num_failures, step=5):
    G = nx.read_graphml(graph_file)
    nodes = list(G.nodes)
    random.shuffle(nodes)


    metrics = {"step": [], "largest_component_size": [], "avg_shortest_path": [],"clustering_coefficient": []}

    for i in range(0, num_failures, step):
        to_remove = nodes[i:i + step]
        G.remove_nodes_from(to_remove)
        
        largest_component = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        metrics["step"].append(i + step)

        metrics["largest_component_size"].append(len(largest_component))

        connected_lengths = []
        for component in nx.weakly_connected_components(G):
            subgraph = G.subgraph(component)
            lengths = dict(nx.shortest_path_length(subgraph))
            for src, targets in lengths.items():
                connected_lengths.extend(targets.values())

        if connected_lengths:
            metrics["avg_shortest_path"].append(sum(connected_lengths) / len(connected_lengths))
        else:
            metrics["avg_shortest_path"].append(float('inf'))
                
        metrics["clustering_coefficient"].append(nx.average_clustering(G.to_undirected()))

    return metrics

def targeted_failures(graph_file, num_failures, step=5, centrality_measure="betweenness"):
    G = nx.read_graphml(graph_file)

    if centrality_measure == "degree":
        centrality = nx.degree_centrality(G)
    elif centrality_measure == "betweenness":
        centrality = nx.betweenness_centrality(G, normalized=True, weight="weight")
    elif centrality_measure == "pagerank":
        centrality = nx.pagerank(G, alpha=0.85)
    else:
        raise ValueError("Invalid centrality measure.")
    
    nodes_sorted = sorted(centrality, key=centrality.get, reverse=True)

    metrics = {"step": [], "largest_component_size": [], "avg_shortest_path": [], "clustering_coefficient": []}

    for i in range(0, num_failures, step):
        to_remove = nodes_sorted[i:i + step]
        G.remove_nodes_from(to_remove)
        
        largest_component = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
        metrics["step"].append(i + step)

        metrics["largest_component_size"].append(len(largest_component))

        connected_lengths = []
        for component in nx.weakly_connected_components(G):
            subgraph = G.subgraph(component)
            lengths = dict(nx.shortest_path_length(subgraph))
            for src, targets in lengths.items():
                connected_lengths.extend(targets.values())

        if connected_lengths:
            metrics["avg_shortest_path"].append(sum(connected_lengths) / len(connected_lengths))
        else:
            metrics["avg_shortest_path"].append(float('inf'))
        
        metrics["clustering_coefficient"].append(nx.average_clustering(G.to_undirected()))

    return metrics

def plot_failure_results(random_metrics, targeted_metrics):
    plt.figure(figsize=(9, 6))
    plt.plot(random_metrics["step"], random_metrics["largest_component_size"], label="Random", color="blue")
    plt.plot(targeted_metrics["step"], targeted_metrics["largest_component_size"], label="Targeted", color="red")
    plt.title("Largest Component Size")
    plt.xlabel("Nodes Removed")
    plt.ylabel("Size")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(9, 6))
    plt.plot(random_metrics["step"], random_metrics["avg_shortest_path"], label="Random", color="blue")
    plt.plot(targeted_metrics["step"], targeted_metrics["avg_shortest_path"], label="Targeted", color="red")
    plt.title("Average Shortest Path Length")
    plt.xlabel("Nodes Removed")
    plt.ylabel("Path Length")
    plt.legend()
    plt.grid()

    plt.figure(figsize=(9, 6))
    plt.plot(random_metrics["step"], random_metrics["clustering_coefficient"], label="Random", color="blue")
    plt.plot(targeted_metrics["step"], targeted_metrics["clustering_coefficient"], label="Targeted", color="red")
    plt.title("Clustering Coefficient")
    plt.xlabel("Nodes Removed")
    plt.ylabel("Coefficient")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

