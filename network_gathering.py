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
    # Initialize dictionaries to store aggregated data
    trip_counts = {}  # To store number of trips between zones
    trip_distance = {}
    places_lat_long = {}

    # Process the CSV file in chunks
    for chunk in pd.read_csv(file_path, usecols=["PULocation", "DOLocation", "PULat", "PULong","DOLat","DOLong","trip_distance"], chunksize=chunksize):
        # Iterate through each row in the chunk
        for _, row in chunk.iterrows():
            origin, destination = row["PULocation"], row["DOLocation"]
            origin_lat, origin_long, dest_lat, dest_long, dist =  row["PULat"], row["PULong"], row["DOLat"], row["DOLong"], row["trip_distance"]

            # Update trip count dictionary
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

    # # Construct the first graph: Trip Frequency Graph
    # trip_frequency_graph = nx.Graph()
    # for (origin, destination), count in new_trip_counts.items():
    #     if origin not in trip_frequency_graph.nodes:
    #         trip_frequency_graph.add_node(origin, label=origin, lat=places_lat_long[origin][0], long=places_lat_long[origin][1])
    #     if destination not in trip_frequency_graph.nodes:
    #         trip_frequency_graph.add_node(destination, label=destination, lat=places_lat_long[destination][0], long=places_lat_long[destination][1])
    #     trip_frequency_graph.add_edge(origin, destination, weight=count)
    trip_distance_graph = construct_graph(places_lat_long, new_trip_distance, True)
    trip_frequency_graph = construct_graph(places_lat_long, trip_counts, False)

    # print("Testing trip frequency graph")
    # print(trip_frequency_graph.nodes["JFK Airport"])

    # Save the graphs in GraphML format for Gephi
    trip_distance_graph_file_name = "output_files/trip_dist_graph.graphml"
    nx.write_graphml(trip_distance_graph, trip_distance_graph_file_name)
    trip_frequency_graph_file_name = "output_files/trip_freq_graph.graphml"
    nx.write_graphml(trip_frequency_graph, trip_frequency_graph_file_name)

    return trip_frequency_graph_file_name, trip_distance_graph_file_name


# Helper function to construct a graph from trip counts
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
    
    # Initialize dictionaries to store trip counts for each time period
    time_trip_counts = {}

    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        origin, destination, origin_lat, origin_long, dest_lat, dest_long = (
            row["PULocation"], row["DOLocation"], row["PULat"], row["PULong"], row["DOLat"], row["DOLong"]
        )

        # Update trip count dictionaries based on the time period
        
        if (origin, destination) not in time_trip_counts:
            time_trip_counts[(origin, destination)] = 0
        time_trip_counts[(origin, destination)] += 1
        
        # Update places latitude and longitude dictionary
        if origin not in places_lat_long:
            places_lat_long[origin] = (origin_lat, origin_long)

        if destination not in places_lat_long:
            places_lat_long[destination] = (dest_lat, dest_long)

    # Construct graphs for each time period
    time_trip_graph = construct_graph(places_lat_long, time_trip_counts, False)
    
    # Save the graphs in GraphML format for Gephi
    graph_file_name = "output_files/" + time_of_day + "_trip_graph.graphml"
    nx.write_graphml(time_trip_graph, graph_file_name)

    return graph_file_name

# def degree_distribution_plot(graph_file):
#     # Read the directed graph
#     G = nx.read_graphml(graph_file)

#     # Calculate in-degrees and out-degrees
#     in_degrees = [G.in_degree(node) for node in G.nodes()]
#     out_degrees = [G.out_degree(node) for node in G.nodes()]

#     # Calculate degree distribution for in-degrees
#     in_degree_count = Counter(in_degrees)
#     total_nodes = G.number_of_nodes()
#     in_degree_probabilities = {degree: count / total_nodes for degree, count in in_degree_count.items()}

#     # Calculate degree distribution for out-degrees
#     out_degree_count = Counter(out_degrees)
#     out_degree_probabilities = {degree: count / total_nodes for degree, count in out_degree_count.items()}

#     # Plot in-degree distribution
#     plt.figure(figsize=(8, 5))
#     plt.scatter(in_degree_probabilities.keys(), in_degree_probabilities.values(), color='red', alpha=0.7, label='In-Degree')
#     plt.title("In-Degree Distribution")
#     plt.xlabel("Degree")
#     plt.ylabel("Probability")
#     plt.grid(True)
#     plt.legend()
#     plt.show()

#     # Plot out-degree distribution
#     plt.figure(figsize=(8, 5))
#     plt.scatter(out_degree_probabilities.keys(), out_degree_probabilities.values(), color='blue', alpha=0.7, label='Out-Degree')
#     plt.title("Out-Degree Distribution")
#     plt.xlabel("Degree")
#     plt.ylabel("Probability")
#     plt.grid(True)
#     plt.legend()
#     plt.show()


def rank_distribution_plot(graph_file, alpha=0.88):
    # Read the directed graph
    G = nx.read_graphml(graph_file)

    # Ensure nodes have 'lat' and 'long' attributes for spatial data
    if not all('lat' in G.nodes[node] and 'long' in G.nodes[node] for node in G.nodes):
        raise ValueError("Graph nodes must have 'lat' and 'long' attributes for spatial ranking.")

    # Function to calculate the distance between two nodes (Haversine formula)
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in kilometers
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon1 - lon2)
        a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Compute ranks and probabilities for all transitions
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
        
        # Sort distances
        distances = sorted(distances, key=lambda x: x[1])
        
        # Compute rank and probability for each transition u -> v
        for idx, (v, d_uv) in enumerate(distances):
            rank_uv = idx + 1  # Rank is the position in the sorted list (1-based index)
            probability_uv = 1 / (rank_uv ** alpha)  # Probability based on rank and alpha
            ranks.append(rank_uv)
            probabilities.append(probability_uv)
    
    # Plot rank-based probabilities
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
    #trip_dist_graph = construct_graphs_from_csv("output_files/ny_taxi_10_2024_data_lat_long.csv")
    #rank_distribution_plot("output_files/trip_dist_graph.graphml")
    #file_name = combine_csv()
    #new_file_name = add_lat_long_to_csv(file_name)
    # new_file_name = "output_files/ny_taxi_10_2024_data_lat_long.csv"
    # trip_freq_graph, trip_dist_graph = construct_graphs_from_csv(new_file_name)
    # rank_distribution_plot(trip_dist_graph)
    rank_distribution_plot("output_files/trip_dist_graph.graphml")

    # # R1 
    # random_metrics = random_failures("output_files/trip_freq_graph.graphml", 190)
    # targeted_metrices = targeted_failures("output_files/trip_freq_graph.graphml",190, centrality_measure="pagerank")
    # plot_failure_results(random_metrics, targeted_metrices)

    # # R2
    # morning_df, evening_df, afternoon_df, night_df = get_dataset_with_time_periods("output_files/ny_taxi_10_2024_data_lat_long.csv")
    # morning_graph_file = construct_time_period_graphs(morning_df, "morning")
    # evening_graph_file = construct_time_period_graphs(evening_df, "evening")
    # afternoon_graph_file = construct_time_period_graphs(afternoon_df, "afternoon")
    # night_graph_file = construct_time_period_graphs(night_df, "night")  
    # morning_data_summary = summarize_metrics("output_files/morning_trip_graph.graphml") 
    # evening_data_summary = summarize_metrics("output_files/evening_trip_graph.graphml")
    # afternoon_data_summary = summarize_metrics("output_files/afternoon_trip_graph.graphml")
    # night_data_summary = summarize_metrics("output_files/night_trip_graph.graphml")
    # write_summary_to_file(morning_data_summary, "output_files/morning_summary_data.txt")
    # write_summary_to_file(evening_data_summary, "output_files/evening_summary_data.txt")
    # write_summary_to_file(afternoon_data_summary, "output_files/afternoon_summary_data.txt")
    # write_summary_to_file(night_data_summary, "output_files/night_summary_data.txt")

    # morning_map = visualize_top_nodes_from_summary("output_files/morning_trip_graph.graphml", morning_data_summary, "Morning")
    # evening_map = visualize_top_nodes_from_summary("output_files/evening_trip_graph.graphml", evening_data_summary, "Evening")
    # afternoon_map = visualize_top_nodes_from_summary("output_files/afternoon_trip_graph.graphml", afternoon_data_summary, "Afternoon")
    # night_map = visualize_top_nodes_from_summary("output_files/night_trip_graph.graphml", night_data_summary, "Night")

    # # Save the maps
    # morning_map.save("output_files/morning_top_nodes.html")
    # afternoon_map.save("output_files/afternoon_top_nodes.html")
    # evening_map.save("output_files/evening_top_nodes.html")
    # night_map.save("output_files/night_top_nodes.html")

    # run_community_detection()





