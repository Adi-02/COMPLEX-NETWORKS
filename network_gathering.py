import networkx as nx
import matplotlib.pyplot as plt 
from preprocessing import combine_csv, add_lat_long_to_csv, get_dataset_with_time_periods
from random_targetted_failure import random_failures, targeted_failures, plot_failure_results
from time_period_graph import summarize_metrics, visualize_top_nodes_from_summary, write_summary_to_file
from community_detection import run_community_detection
from collections import Counter

def construct_graph(places_lat_long, trip_counts):
    graph = nx.DiGraph()
    for (origin, destination), count in trip_counts.items():
        if count > 30:
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

    time_trip_graph = construct_graph(places_lat_long, time_trip_counts)
    
    graph_file_name = "output_files/" + time_of_day + "_trip_graph.graphml"
    nx.write_graphml(time_trip_graph, graph_file_name)

    return graph_file_name

def degree_distribution_plot(graph_file):
    G = nx.read_graphml(graph_file)
    in_degrees = [G.in_degree(node) for node in G.nodes()]
    out_degrees = [G.out_degree(node) for node in G.nodes()]

    in_degree_count = Counter(in_degrees)
    total_nodes = G.number_of_nodes()
    in_degree_probabilities = {degree: count / total_nodes for degree, count in in_degree_count.items()}

    out_degree_count = Counter(out_degrees)
    out_degree_probabilities = {degree: count / total_nodes for degree, count in out_degree_count.items()}

    plt.figure(figsize=(8, 5))
    plt.scatter(in_degree_probabilities.keys(), in_degree_probabilities.values(), color='red', alpha=0.7, label='In-Degree')
    plt.title("In-Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(out_degree_probabilities.keys(), out_degree_probabilities.values(), color='blue', alpha=0.7, label='Out-Degree')
    plt.title("Out-Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    file_name = combine_csv()
    df, output_file_name = add_lat_long_to_csv(file_name)
    trip_freq_graph = construct_time_period_graphs(df, "freq")
    degree_distribution_plot(trip_freq_graph)

    # Research Question 1 
    random_metrics = random_failures(trip_freq_graph, 190)
    targeted_metrices = targeted_failures(trip_freq_graph,190, centrality_measure="pagerank")
    plot_failure_results(random_metrics, targeted_metrices)

    # Research Question 2
    morning_df, evening_df, afternoon_df, night_df = get_dataset_with_time_periods(output_file_name)
    morning_graph_file = construct_time_period_graphs(morning_df, "morning")
    evening_graph_file = construct_time_period_graphs(evening_df, "evening")
    afternoon_graph_file = construct_time_period_graphs(afternoon_df, "afternoon")
    night_graph_file = construct_time_period_graphs(night_df, "night")  
    morning_data_summary, morning_summary_f_name = summarize_metrics(morning_graph_file) 
    evening_data_summary, evening_summary_f_name = summarize_metrics(evening_graph_file)
    afternoon_data_summary, afternoon_summary_f_name = summarize_metrics(afternoon_graph_file)
    night_data_summary, night_summary_f_name = summarize_metrics(night_graph_file)
    write_summary_to_file(morning_data_summary, morning_summary_f_name)
    write_summary_to_file(evening_data_summary, evening_summary_f_name)
    write_summary_to_file(afternoon_data_summary, afternoon_summary_f_name)
    write_summary_to_file(night_data_summary, night_summary_f_name)

    morning_map = visualize_top_nodes_from_summary(morning_graph_file, morning_data_summary, "Morning")
    evening_map = visualize_top_nodes_from_summary(evening_graph_file, evening_data_summary, "Evening")
    afternoon_map = visualize_top_nodes_from_summary(afternoon_graph_file, afternoon_data_summary, "Afternoon")
    night_map = visualize_top_nodes_from_summary(night_graph_file, night_data_summary, "Night")

    run_community_detection()





