import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt 
from preprocessing import combine_csv, add_lat_long_to_csv, get_dataset_with_time_periods
from random_targetted_failure import random_failures, targeted_failures, plot_failure_results
from time_period_graph import summarize_metrics, visualize_top_nodes_from_summary, write_summary_to_file
from community_detection import run_community_detection
from collections import Counter
import numpy as np 

def construct_graphs_from_csv(file_path, chunksize=100000):
    trip_counts = {} 
    trip_distance = {}
    places_lat_long = {}

    for chunk in pd.read_csv(file_path, usecols=["PULocation", "DOLocation", "PULat", "PULong","DOLat","DOLong","trip_distance"], chunksize=chunksize):
        for _, row in chunk.iterrows():
            origin, destination = row["PULocation"], row["DOLocation"]
            origin_lat, origin_long, dest_lat, dest_long, dist =  row["PULat"], row["PULong"], row["DOLat"], row["DOLong"], row["trip_distance"]

            if (origin, destination) not in trip_distance:
                trip_distance[(origin, destination)] = [dist, 1]
                trip_counts[(origin, destination)] = 0
            else:
                prev_dist = trip_distance[(origin, destination)][0]
                prev_counter = trip_distance[(origin, destination)][1]
                trip_distance[(origin, destination)][0] = int((prev_dist + dist) / (prev_counter + 1))
                trip_distance[(origin, destination)][1] += 1
                trip_counts[(origin, destination)] += 1
            

            if origin not in places_lat_long:
                places_lat_long[origin] = (origin_lat, origin_long)
            
            if destination not in places_lat_long:
                places_lat_long[destination] = (dest_lat, dest_long)

    new_trip_distance = {}
    for i in trip_counts:
        new_trip_distance[i] = trip_distance[i][0]

    trip_distance_graph = construct_graph(places_lat_long, new_trip_distance, True)
    trip_frequency_graph = construct_graph(places_lat_long, trip_counts, False)

    trip_distance_graph_file_name = "output_files/trip_dist_graph.graphml"
    nx.write_graphml(trip_distance_graph, trip_distance_graph_file_name)
    trip_frequency_graph_file_name = "output_files/trip_freq_graph.graphml"
    nx.write_graphml(trip_frequency_graph, trip_frequency_graph_file_name)

    return trip_frequency_graph_file_name, trip_distance_graph_file_name


def construct_graph(places_lat_long, trip_counts, dist):
    if dist:
        graph = nx.Graph()
    else:
        graph = nx.DiGraph()
    for (origin, destination), count in trip_counts.items():
        if not dist:
            if count > 30:
                if origin not in graph.nodes:
                    graph.add_node(origin, label=origin, lat=places_lat_long[origin][0], long=places_lat_long[origin][1])
                if destination not in graph.nodes:
                    graph.add_node(destination, label=destination, lat=places_lat_long[destination][0], long=places_lat_long[destination][1])
                graph.add_edge(origin, destination, weight=count)
        else:
            if origin not in graph.nodes:
                    graph.add_node(origin, label=origin, lat=places_lat_long[origin][0], long=places_lat_long[origin][1])
            if destination not in graph.nodes:
                graph.add_node(destination, label=destination, lat=places_lat_long[destination][0], long=places_lat_long[destination][1])
            graph.add_edge(origin, destination, weight=count)
    return graph


def construct_time_period_graphs(df, time_of_day):
    places_lat_long = {}
    time_trip_counts = {}

    for _, row in df.iterrows():
        origin, destination, origin_lat, origin_long, dest_lat, dest_long = (
            row["PULocation"], row["DOLocation"], row["PULat"], row["PULong"], row["DOLat"], row["DOLong"]
        )
        
        if (origin, destination) not in time_trip_counts:
            time_trip_counts[(origin, destination)] = 0
        time_trip_counts[(origin, destination)] += 1
        
        if origin not in places_lat_long:
            places_lat_long[origin] = (origin_lat, origin_long)

        if destination not in places_lat_long:
            places_lat_long[destination] = (dest_lat, dest_long)

    time_trip_graph = construct_graph(places_lat_long, time_trip_counts, False)
    
    graph_file_name = "output_files/" + time_of_day + "_trip_graph.graphml"
    nx.write_graphml(time_trip_graph, graph_file_name)

    return graph_file_name

# SOurce: https://en.wikipedia.org/wiki/Haversine_formula
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon1 - lon2)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def rank_distribution_plot(graph_file, alpha=0.88):
    G = nx.read_graphml(graph_file)

    ranks = []
    probabilities = []
    node_positions = {node: (float(G.nodes[node]['lat']), float(G.nodes[node]['long'])) for node in G.nodes}
    
    for u in G.nodes:
        u_lat, u_long = node_positions[u]
        distances = []
        for v in G.nodes:
            if u != v:
                v_lat, v_long = node_positions[v]
                distance = haversine_distance(u_lat, u_long, v_lat, v_long)
                distances.append((v, distance))
        
        distances = sorted(distances, key=lambda x: x[1])
        
        for idx, (v, d_uv) in enumerate(distances):
            rank_uv = idx + 1 
            probability_uv = 1 / (rank_uv ** alpha)  
            ranks.append(rank_uv)
            probabilities.append(probability_uv)
    

    plt.figure(figsize=(8, 5))
    plt.scatter(ranks, probabilities, color='purple', alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Rank Distribution")
    plt.xlabel("Rank")
    plt.ylabel("PDF")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    file_name = combine_csv()
    new_file_name = add_lat_long_to_csv(file_name)
    trip_freq_graph, trip_dist_graph = construct_graphs_from_csv(new_file_name)
    rank_distribution_plot(trip_dist_graph)

    # Research Question 1 
    random_metrics = random_failures("output_files/trip_freq_graph.graphml", 190)
    targeted_metrices = targeted_failures("output_files/trip_freq_graph.graphml",190, centrality_measure="pagerank")
    plot_failure_results(random_metrics, targeted_metrices)

    # Research Question 2
    morning_df, evening_df, afternoon_df, night_df = get_dataset_with_time_periods("output_files/ny_taxi_10_2024_data_lat_long.csv")
    morning_graph_file = construct_time_period_graphs(morning_df, "morning")
    evening_graph_file = construct_time_period_graphs(evening_df, "evening")
    afternoon_graph_file = construct_time_period_graphs(afternoon_df, "afternoon")
    night_graph_file = construct_time_period_graphs(night_df, "night")  
    morning_data_summary = summarize_metrics("output_files/morning_trip_graph.graphml") 
    evening_data_summary = summarize_metrics("output_files/evening_trip_graph.graphml")
    afternoon_data_summary = summarize_metrics("output_files/afternoon_trip_graph.graphml")
    night_data_summary = summarize_metrics("output_files/night_trip_graph.graphml")
    write_summary_to_file(morning_data_summary, "output_files/morning_summary_data.txt")
    write_summary_to_file(evening_data_summary, "output_files/evening_summary_data.txt")
    write_summary_to_file(afternoon_data_summary, "output_files/afternoon_summary_data.txt")
    write_summary_to_file(night_data_summary, "output_files/night_summary_data.txt")

    morning_map = visualize_top_nodes_from_summary("output_files/morning_trip_graph.graphml", morning_data_summary, "Morning")
    evening_map = visualize_top_nodes_from_summary("output_files/evening_trip_graph.graphml", evening_data_summary, "Evening")
    afternoon_map = visualize_top_nodes_from_summary("output_files/afternoon_trip_graph.graphml", afternoon_data_summary, "Afternoon")
    night_map = visualize_top_nodes_from_summary("output_files/night_trip_graph.graphml", night_data_summary, "Night")
    morning_map.save("output_files/morning_top_nodes.html")
    afternoon_map.save("output_files/afternoon_top_nodes.html")
    evening_map.save("output_files/evening_top_nodes.html")
    night_map.save("output_files/night_top_nodes.html")

    run_community_detection()





