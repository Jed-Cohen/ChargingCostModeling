import numpy as np
import pandas as pd


# This method is used to interpolate missing data from nearby zip codes
def zip_interpolate(col, zipcode, zip_map):
    upper = zipcode
    lower = upper
    upper_factor = 1
    lower_factor = 1
    while (upper not in zip_map or zip_map[upper][col] == 0) and upper_factor == 1:
        upper += 1
        if upper == 100000:
            upper_factor = 0
            upper = 98107
    while (lower not in zip_map or zip_map[lower][col] == 0) and lower_factor == 1:
        lower -= 1
        if lower == 599:
            lower_factor = 0
            lower = 98107
    return (zip_map[upper][col] * upper_factor + zip_map[lower][col] * lower_factor) / (upper_factor + lower_factor)


if __name__ == '__main__':
    land = pd.read_csv('Data/LandPrices.csv')
    electric = pd.read_csv('Data/ElectricityData.csv')
    econ = pd.read_csv('Data/EconData.csv')
    stations = pd.read_csv('Data/Station_Data.csv', low_memory=False)
    zipcodes = pd.read_csv('Data/Zipcodes.csv')
    sales_tax = pd.read_csv('Data/Sales Tax Data.csv').to_numpy()
    population = pd.read_csv('Data/Population.csv').to_numpy()
    # These have to be deleted to convert to numpy
    del electric['utility_name']
    del electric['state']
    del electric['service_type']
    del electric['ownership']
    del econ['country']
    ar = zipcodes.to_numpy()
    land_array = land.to_numpy()
    econ_data = econ.to_numpy()
    electric_data = electric.to_numpy()
    num_stats = 6  # parameter that can be changed based on how many columns you want per station
    columns = {'DCFC_$/hr': stations['DCFC $/Hr'],
               'DCFC_flat': stations['DCFC flat'], 'DCFC_$/kWh': stations['DCFC $/kWh'], 'State': stations['State']}
    station_data = pd.DataFrame(columns)
    # Mark stations that have pricing data
    station_data['is_na'] = station_data[station_data.columns].isnull().apply(lambda x: all(x), axis=1)
    station_data['ZIP'] = stations['ZIP']
    station_data['DCFC_count'] = stations['EV DC Fast Count']
    d = {}  # A dictionary that takes zip codes as keys and returns an array of metrics
    for row in ar:
        d[row[0]] = np.zeros(num_stats + 2)
        d[row[0]][0], d[row[0]][1] = row[1], row[2]
    DCFC_counter = 0
    DCFC_stations = np.zeros((len(station_data['ZIP']), 4))  # An array of stations with their zips and prices
    for ind in station_data.index:
        if int(station_data['ZIP'][ind]) in d:
            if not station_data['is_na'][ind]:
                if not pd.isnull(station_data['DCFC_count'][ind]):
                    DCFC_stations[DCFC_counter][0] = station_data['ZIP'][ind]
                    if not pd.isnull(station_data['DCFC_$/hr'][ind]):
                        DCFC_stations[DCFC_counter][1] = station_data['DCFC_$/hr'][ind]
                    if not pd.isnull(station_data['DCFC_flat'][ind]):
                        DCFC_stations[DCFC_counter][2] = station_data['DCFC_flat'][ind]
                    if not pd.isnull(station_data['DCFC_$/kWh'][ind]):
                        DCFC_stations[DCFC_counter][3] = station_data['DCFC_$/kWh'][ind]
                    DCFC_counter += 1
            # used for station count later
            d[int(station_data['ZIP'][ind])][6] += 1
    # adding other metric to the dictionary
    # adding land value
    for row in land_array:
        if row[0] not in d:
            d[row[0]] = np.zeros(num_stats + 3)
        d[row[0]][2] = row[2]
    # adding sum of all electricity rates and total number of rates to calculate average
    for row in electric_data:
        if row[0] not in d:
            d[row[0]] = np.zeros(num_stats + 3)
        d[row[0]][3] += row[2]
        d[row[0]][4] += 1
    econ_dict = {}  # a dictionary that takes lat/lon pair to economic activity
    for row in econ_data:
        econ_dict[(row[0], row[1])] = row[15]
    # Add economic activity to dictionary based on closest lat/lon
    for row in ar:
        if (round(row[1]), round(row[2])) in econ_dict:
            d[row[0]][5] = econ_dict[(round(row[1]), round(row[2]))]
    for row in sales_tax:
        if int(row[1]) in d:
            d[row[1]][7] = row[3]
    final_data = np.zeros((len(ar), num_stats))   # an array for zip codes with their metrics
    for i in range(len(ar)):
        cur_zip = ar[i][0]
        final_data[i][0] = cur_zip
        if d[ar[i][0]][2] != 0:
            final_data[i][2] = d[cur_zip][2]
        else:
            final_data[i][1] = zip_interpolate(2, cur_zip, d)
        if d[cur_zip][4] != 0:
            final_data[i][1] = d[cur_zip][3] / d[cur_zip][4]
        else:
            final_data[i][1] = zip_interpolate(3, cur_zip, d) / zip_interpolate(4, cur_zip, d)
        if d[cur_zip][5] != 0:
            final_data[i][3] = d[cur_zip][5]
        else:
            final_data[i][3] = zip_interpolate(5, cur_zip, d)
        final_data[i][5] = d[cur_zip][7]
        final_data[i][4] = d[cur_zip][6]
    edited_data = pd.DataFrame(final_data, columns=['ZIP', 'Electricity Price', 'Land Value',
                                                    'Economic Activity', 'Station Count', 'Sales Tax'])
    edited_data.to_csv('Data/USA_model_input.csv')  # This is how I get the data that I give to the complete model
    for row in final_data:
        d[row[0]] = row
    DCFC_data = np.zeros((len(DCFC_stations), num_stats + 1))  # Array for stations and prices and metrics for model
    for i in range(len(DCFC_data)):
        if DCFC_stations[i][0] != 0:
            # Price standardization (can be tweaked)
            DCFC_data[i][0] = DCFC_stations[i][2] + DCFC_stations[i][1] * .5 + DCFC_stations[i][3] * 30
            ref = d[int(DCFC_stations[i][0])]
            # translate over the statistics
            for j in range(num_stats - 1):
                DCFC_data[i][j + 1] = ref[j + 1]
            DCFC_data[i][num_stats] = int(DCFC_stations[i][0])
    counter = 0
    # remove any rows without complete data
    for row in DCFC_data:
        if 0 not in (row[0], row[1], row[2], row[3], row[4]):
            counter += 1
    DCFC_clean = np.zeros((counter, num_stats + 1))
    counter = 0
    for row in DCFC_data:
        if 0 not in (row[0], row[1], row[2], row[3], row[4]):
            DCFC_clean[counter] = row
            counter += 1
    counter = 0
    columns = ['Converted Cost', 'Electric Price', 'Land Value',
               'Economic Activity', 'Station Count', 'Sales Tax', 'ZIP']
    DCFC_df = pd.DataFrame(DCFC_clean, columns=columns)
    DCFC_df.to_csv('Data/USA_stations.csv')
