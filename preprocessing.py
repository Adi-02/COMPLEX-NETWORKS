import pandas as pd 

def combine_csv():    
    csv_files = ["input_files/yellow_tripdata_2024-10.csv"]

    # Specify the columns you want to load
    columns_to_read = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "PULocationID", "DOLocationID", "total_amount"]

    # Load the taxi zone lookup file
    taxi_zone_lookup = pd.read_csv('input_files/taxi_zone_lookup.csv')

    # Create an empty list to store chunks
    data_chunks = []

    # Define the valid date range
    valid_start_date = pd.Timestamp("2024-10-01")
    valid_end_date = pd.Timestamp("2024-10-31")

    # Process each file in chunks
    for file in csv_files:
        print(f"Processing file: {file}")
        # Read the file in chunks, selecting specific columns
        for chunk in pd.read_csv(file, usecols=columns_to_read, chunksize=100000):
            # Convert datetime columns to proper datetime objects
            chunk["tpep_pickup_datetime"] = pd.to_datetime(chunk["tpep_pickup_datetime"], errors="coerce")
            chunk["tpep_dropoff_datetime"] = pd.to_datetime(chunk["tpep_dropoff_datetime"], errors="coerce")

            # Merge the chunk with taxi_zone_lookup for PULocationID
            chunk = chunk.merge(
                taxi_zone_lookup[['LocationID', 'Zone']],
                left_on='PULocationID',
                right_on='LocationID',
                how='left'
            ).rename(columns={'Zone': 'PULocation'})
            chunk = chunk.drop(columns=['LocationID', 'PULocationID'])

            # Merge the chunk with taxi_zone_lookup for DOLocationID
            chunk = chunk.merge(
                taxi_zone_lookup[['LocationID', 'Zone']],
                left_on='DOLocationID',
                right_on='LocationID',
                how='left'
            ).rename(columns={'Zone': 'DOLocation'})
            chunk = chunk.drop(columns=['LocationID', 'DOLocationID'])

            # Append the processed chunk to the list
            data_chunks.append(chunk)

    # Combine all the chunks into a single DataFrame
    combined_data = pd.concat(data_chunks, ignore_index=True)

    # Sort the combined data by tpep_pickup_datetime
    combined_data = combined_data.sort_values(by="tpep_pickup_datetime")

    # Filter rows with valid pickup and dropoff datetime ranges
    combined_data = combined_data[
        (combined_data["tpep_pickup_datetime"] >= valid_start_date) &
        (combined_data["tpep_pickup_datetime"] <= valid_end_date) &
        (combined_data["tpep_dropoff_datetime"] >= valid_start_date) &
        (combined_data["tpep_dropoff_datetime"] <= valid_end_date)
    ]

    # Drop rows with invalid datetime values (NaT)
    combined_data = combined_data.dropna(subset=["PULocation", "DOLocation"])

    combined_data = combined_data[combined_data['trip_distance'] != 0.0]

    # Save the combined data to a new CSV file
    output_file = 'output_files/ny_taxi_10_2024_data.csv'
    combined_data.to_csv(output_file, index=False, chunksize=100000)

    print(f"Combined {len(csv_files)} files into '{output_file}' with zone names, valid dates, and sorted by pickup datetime.")

    return output_file


def add_lat_long_to_csv(file_name):
   # Load the latitude and longitude data
    lat_long = pd.read_csv("input_files/lat_long.csv")
    
    # Remove ", New York" from the Place column in lat_long
    lat_long["Place"] = lat_long["Place"].str.replace(", New York", "", regex=False)
    
    # Load the NYC taxi data
    nyc_taxi_data = pd.read_csv(file_name)
    
    # Merge latitude and longitude for PULocation
    nyc_taxi_data = nyc_taxi_data.merge(
        lat_long.rename(columns={"Place": "PULocation", "Lat": "PULat", "Long": "PULong"}),
        on="PULocation",
        how="left"
    )
    
    # Merge latitude and longitude for DOLocation
    nyc_taxi_data = nyc_taxi_data.merge(
        lat_long.rename(columns={"Place": "DOLocation", "Lat": "DOLat", "Long": "DOLong"}),
        on="DOLocation",
        how="left"
    )
    
    # Save the updated dataframe to a new CSV file
    output_file_name = file_name[:-4] + "_lat_long.csv"
    nyc_taxi_data.to_csv(output_file_name, index=False)
    print(f"File with latitude and longitude added: {output_file_name}")
    return output_file_name

def get_dataset_with_time_periods(dataset_name):
    # Define time periods
    morning_rush_hours = list(range(4, 11))  # 6 AM - 11 AM
    evening_rush_hours = list(range(16, 21))  # 4 PM - 9 PM
    afternoon_hours = list(range(11, 16))  # 11 AM - 4 PM

    # Initialize empty lists to collect processed chunks for each time period
    morning_chunks = []
    evening_chunks = []
    afternoon_chunks = []
    night_chunks = []

    # Read the dataset in chunks
    chunk_iter = pd.read_csv(dataset_name, chunksize=100000)

    for chunk in chunk_iter:
        # Convert pickup and dropoff datetime columns to datetime objects
        chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'])
        chunk['tpep_dropoff_datetime'] = pd.to_datetime(chunk['tpep_dropoff_datetime'])

        # Create a function to classify a trip by time period
        def classify_time_period(row):
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

        # Apply the classification to each row
        chunk['Time_Period'] = chunk.apply(classify_time_period, axis=1)

        # Split the chunk into separate time periods
        morning_chunks.append(chunk[chunk['Time_Period'] == 'Morning'])
        evening_chunks.append(chunk[chunk['Time_Period'] == 'Evening'])
        afternoon_chunks.append(chunk[chunk['Time_Period'] == 'Afternoon'])
        night_chunks.append(chunk[chunk['Time_Period'] == 'Night'])

    # Concatenate all processed chunks into separate DataFrames
    morning_df = pd.concat(morning_chunks, ignore_index=True)
    evening_df = pd.concat(evening_chunks, ignore_index=True)
    afternoon_df = pd.concat(afternoon_chunks, ignore_index=True)
    night_df = pd.concat(night_chunks, ignore_index=True)

    # Print a summary of each time period
    print("Morning Data:", morning_df.head())
    print("Evening Data:", evening_df.head())
    print("Afternoon Data:", afternoon_df.head())
    print("Night Data:", night_df.head())

    return morning_df, evening_df, afternoon_df, night_df
    #return morning_df, evening_df, night_df