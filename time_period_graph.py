import networkx as nx
import folium
import re

def get_average_clustering(graph):
    return nx.average_clustering(graph, weight='weight')

def get_top_edges_by_weight(graph, top_n=5):
    edges = graph.edges(data=True)
    sorted_edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
    return sorted_edges[:top_n]

def compute_betweenness_centrality(graph):
    return nx.betweenness_centrality(graph, weight='weight')

def compute_degree_centrality(graph):
    return nx.degree_centrality(graph)

def get_top_central_nodes(centrality, top_n=5):
    return sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]

def summarize_metrics(graph_name):
    graph = nx.read_graphml(graph_name)
    degree_centrality = compute_degree_centrality(graph)
    betweenness_centrality = compute_betweenness_centrality(graph)
    avg_clustering = get_average_clustering(graph)
    top_edges = get_top_edges_by_weight(graph)

    summary = {
        "Graph Name": graph_name,
        "Top Degree Central Nodes": get_top_central_nodes(degree_centrality),
        "Top Betweenness Central Nodes": get_top_central_nodes(betweenness_centrality),
        "Average Clustering Coefficient": avg_clustering,
        "Top Edges (Trips)": top_edges
    }

    time_period = re.split(r'[_,/]+',graph_name)[2]
    summary_text_file = f"output_files/{time_period}_summary_data.txt"
    return summary, summary_text_file


def visualize_top_nodes_from_summary(graph_name, summary, time_period, map_center=(40.7128, -74.0060)):
    graph = nx.read_graphml(graph_name)
    m = folium.Map(location=map_center, zoom_start=12)
    top_nodes = summary["Top Degree Central Nodes"]
    for node, centrality in top_nodes:
        attributes = graph.nodes[node]
        lat = float(attributes['lat'])
        long = float(attributes['long'])
        name = node
        folium.CircleMarker(
            location=(lat, long),
            radius=20, 
            popup=f"Node: {node}, Centrality: {centrality:.4f}, Period: {time_period}",
            color='blue' if time_period == "Morning" else 'red' if time_period == "Evening" else 'green' if time_period == "Afternoon" else 'purple',
            fill=True
        ).add_to(m)

    m.save(f"output_files/{time_period}_top_nodes.html")


def write_summary_to_file(summary_data, file_name):
    try:
        with open(file_name, 'w') as file:
            file.write("Summary Report\n")
            file.write("=" * 50 + "\n\n")
            
            for key, value in summary_data.items():
                file.write(f"{key}:\n")
                
                if isinstance(value, list):
                    for item in value:
                        file.write(f"  - {item}\n")
                else:
                    file.write(f"  {value}\n")
                
                file.write("\n")
        
        print(f"Summary successfully written to {file_name}")
    except Exception as e:
        print(f"An error occurred while writing the summary to the file: {e}")

