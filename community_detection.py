import networkx as nx
import community as community_louvain 
import folium


def convert_to_undirected(graph):
    undirected_graph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)
        if undirected_graph.has_edge(u, v):
            undirected_graph[u][v]['weight'] += weight
        else:
            u_attributes = graph.nodes[u]
            v_attributes = graph.nodes[v]
            if u not in undirected_graph.nodes:
                undirected_graph.add_node(u, label=u_attributes['label'], lat=u_attributes['lat'], long=u_attributes['long'])
            if v not in undirected_graph.nodes:
                undirected_graph.add_node(v, label=v_attributes['label'], lat=v_attributes['lat'], long=v_attributes['long'])
            undirected_graph.add_edge(u, v, weight=weight)
    return undirected_graph


def compute_modularity(undirected_graph):
    partition = community_louvain.best_partition(undirected_graph, weight='weight')
    num_communities = len(set(partition.values()))
    modularity_score = community_louvain.modularity(partition, undirected_graph, weight='weight')
    return modularity_score, partition, num_communities


def visualize_communities_on_map(graph, partition, map_center=(40.7128, -74.0060)):
    m = folium.Map(location=map_center, zoom_start=12)
    community_colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'black']

    for node in graph.nodes():
        community = partition.get(node, 0) % len(community_colors)
        attributes = graph.nodes[node]
        lat = float(attributes['lat'])
        long = float(attributes['long'])
        color = community_colors[community]
        folium.CircleMarker(
            location=(lat, long),
            radius=5,
            popup=f"{node} (Community {community})",
            color=color,
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    return m


def remove_top_central_nodes(graph, top_n=5):
    centrality = nx.degree_centrality(graph)
    
    top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_nodes = [node[0] for node in top_nodes]
    
    graph_removed = graph.copy()
    graph_removed.remove_nodes_from(top_nodes)
    
    print(f"Removed top {top_n} nodes: {top_nodes}")
    return graph_removed

def compute_community_sizes(partition):
    community_sizes = {}
    for node, community in partition.items():
        community_sizes[community] = community_sizes.get(community, 0) + 1
    return community_sizes



def run_community_detection():
    time_periods = ["morning", "afternoon", "evening", "night"]
    modularity_scores = []
    community_size = []
    num_nodes_to_rem = 5

    for period in time_periods:
        directed_graph = nx.read_graphml(f"output_files/{period}_trip_graph.graphml")
        undirected_graph = convert_to_undirected(directed_graph)

        modularity_score, partition, num_communities = compute_modularity(undirected_graph)

        map_visualization = visualize_communities_on_map(undirected_graph, partition)
        map_visualization.save(f"community_detection/community_map_{period}.html")

        filtered_graph = remove_top_central_nodes(undirected_graph, top_n=num_nodes_to_rem)

        modularity_score_filtered, partition, num_filtered_communities = compute_modularity(filtered_graph)
        modularity_scores.append((modularity_score, modularity_score_filtered))

        community_size.append((num_communities, num_filtered_communities))

        map_visualization = visualize_communities_on_map(filtered_graph, partition)
        map_visualization.save(f"community_detection/community_map_filtered_{period}.html")

    print(f"Modularity scores (unfiltered, filtered) : {modularity_scores}")
    print(f"Community Sizes (unfiltered, filtered) : {community_size}")

    return modularity_scores, community_size