import googlemaps
import csv

def google_map_lat_long_finder(nodes):
     # Initialize Google Maps client
    gmaps = googlemaps.Client(key="GOOGLE_MAPS_KEY_HERE")

    # open the file in the write mode
    f = open('lat_long.csv', 'w')
    writer = csv.writer(f)
    # write the header
    writer.writerow(["Place", "Lat", "Long"])
    for node in nodes:
        node = node + ", New York"
        geocode_result = gmaps.geocode(node)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            # create the csv writer
            row = [node, location['lat'], location['lng']]
            # write a row to the csv file
            writer.writerow(row)

        else:
            print(f"Could not find coordinates for {node}")

    # close the file
    f.close()
