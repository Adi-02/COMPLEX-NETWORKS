import googlemaps
import csv

def google_map_lat_long_finder(nodes):
    gmaps = googlemaps.Client(key="GOOGLE_MAPS_KEY_WAS_USED_HERE")

    f = open('lat_long.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(["Place", "Lat", "Long"])
    for node in nodes:
        node = node + ", New York"
        geocode_result = gmaps.geocode(node)
        if geocode_result:
            location = geocode_result[0]['geometry']['location']
            row = [node, location['lat'], location['lng']]
            writer.writerow(row)

        else:
            print(f"Could not find coordinates for {node}")

    f.close()
