import pandas as pd 
import os

def convert_parquet_to_csv():
    parquet_file = "input_files/yellow_tripdata_2024-10.parquet"
    csv_file_name = "input_files/yellow_tripdata_2024-10.csv"

    df = pd.read_parquet(parquet_file)
    df.to_csv(csv_file_name, index=False)

    os.makedirs("output_files", exist_ok=True)
    os.makedirs("community_detection", exist_ok=True)

    print(f"Converted {parquet_file} to {csv_file_name}")

def combine_csv():    
    convert_parquet_to_csv()
    file = "input_files/yellow_tripdata_2024-10.csv"
    columns_to_read = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "PULocationID", "DOLocationID", "total_amount"]
    taxi_zone_lookup = pd.read_csv('input_files/taxi_zone_lookup.csv')
    data_chunks = []
    valid_start_date = pd.Timestamp("2024-10-01")
    valid_end_date = pd.Timestamp("2024-10-31")

    print(f"Processing file: {file}")
    for chunk in pd.read_csv(file, usecols=columns_to_read, chunksize=100000):
        chunk["tpep_pickup_datetime"] = pd.to_datetime(chunk["tpep_pickup_datetime"], errors="coerce")
        chunk["tpep_dropoff_datetime"] = pd.to_datetime(chunk["tpep_dropoff_datetime"], errors="coerce")

        chunk = chunk.merge(
            taxi_zone_lookup[['LocationID', 'Zone']],
            left_on='PULocationID',
            right_on='LocationID',
            how='left'
        ).rename(columns={'Zone': 'PULocation'})
        chunk = chunk.drop(columns=['LocationID', 'PULocationID'])

        chunk = chunk.merge(
            taxi_zone_lookup[['LocationID', 'Zone']],
            left_on='DOLocationID',
            right_on='LocationID',
            how='left'
        ).rename(columns={'Zone': 'DOLocation'})
        chunk = chunk.drop(columns=['LocationID', 'DOLocationID'])
        data_chunks.append(chunk)

    combined_data = pd.concat(data_chunks, ignore_index=True)
    combined_data = combined_data.sort_values(by="tpep_pickup_datetime")
    combined_data = combined_data[
        (combined_data["tpep_pickup_datetime"] >= valid_start_date) &
        (combined_data["tpep_pickup_datetime"] <= valid_end_date) &
        (combined_data["tpep_dropoff_datetime"] >= valid_start_date) &
        (combined_data["tpep_dropoff_datetime"] <= valid_end_date)
    ]

    combined_data = combined_data.dropna(subset=["PULocation", "DOLocation"])
    combined_data = combined_data[combined_data['trip_distance'] != 0.0]
    output_file = 'output_files/ny_taxi_10_2024_data.csv'
    combined_data.to_csv(output_file, index=False, chunksize=100000)

    print(f"CSV File {output_file} succefully created!")

    return output_file


def add_lat_long_to_csv(file_name):
    lat_long = pd.read_csv("input_files/lat_long.csv")
    lat_long["Place"] = lat_long["Place"].str.replace(", New York", "", regex=False)
    nyc_taxi_data = pd.read_csv(file_name)
    
    nyc_taxi_data = nyc_taxi_data.merge(lat_long.rename(columns={"Place": "PULocation", "Lat": "PULat", "Long": "PULong"}),
        on="PULocation",
        how="left"
    )
    
    nyc_taxi_data = nyc_taxi_data.merge(lat_long.rename(columns={"Place": "DOLocation", "Lat": "DOLat", "Long": "DOLong"}),
        on="DOLocation",
        how="left"
    )
    
    output_file_name = file_name[:-4] + "_lat_long.csv"
    nyc_taxi_data.to_csv(output_file_name, index=False)
    print(f"File with latitude and longitude added: {output_file_name}")
    return output_file_name

def classify_time_period(row):
    morning_rush_hours = list(range(4, 11))
    evening_rush_hours = list(range(16, 21)) 
    afternoon_hours = list(range(11, 16))
    pickup_hour = row['tpep_pickup_datetime'].hour
    dropoff_hour = row['tpep_dropoff_datetime'].hour

    if pickup_hour in morning_rush_hours or dropoff_hour in morning_rush_hours:
        return 'Morning'
    elif pickup_hour in evening_rush_hours or dropoff_hour in evening_rush_hours:
        return 'Evening'
    elif pickup_hour in afternoon_hours or dropoff_hour in afternoon_hours:
        return 'Afternoon'
    else:
        return 'Night'

def get_dataset_with_time_periods(dataset_name):
    morning_chunks = []
    evening_chunks = []
    afternoon_chunks = []
    night_chunks = []

    chunk_iter = pd.read_csv(dataset_name, chunksize=100000)

    for chunk in chunk_iter:
        chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'])
        chunk['tpep_dropoff_datetime'] = pd.to_datetime(chunk['tpep_dropoff_datetime'])
        chunk['Time_Period'] = chunk.apply(classify_time_period, axis=1)
        morning_chunks.append(chunk[chunk['Time_Period'] == 'Morning'])
        evening_chunks.append(chunk[chunk['Time_Period'] == 'Evening'])
        afternoon_chunks.append(chunk[chunk['Time_Period'] == 'Afternoon'])
        night_chunks.append(chunk[chunk['Time_Period'] == 'Night'])

    morning_df = pd.concat(morning_chunks, ignore_index=True)
    evening_df = pd.concat(evening_chunks, ignore_index=True)
    afternoon_df = pd.concat(afternoon_chunks, ignore_index=True)
    night_df = pd.concat(night_chunks, ignore_index=True)

    return morning_df, evening_df, afternoon_df, night_df