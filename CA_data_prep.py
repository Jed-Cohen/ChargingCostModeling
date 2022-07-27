import numpy as np
import pandas as pd

# This program does essentially the same thing as Data_Prep.py but for California only
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
    zip_county = pd.read_csv('Data/CA zip county.csv')
    registration = pd.read_csv('Data/ca_ev_registrations_public.csv', low_memory=False)
    Codes = pd.read_csv('Data/County Codes.csv')
    code_dict = {}
    population_dict = {}
    for row in Codes.index:
        code_dict[Codes['County'][row]] = Codes['Code'][row]
        population_dict[Codes['County'][row]] = Codes['Population'][row]
    county_count = {}
    for row in registration.index:
        if registration['County GEOID'][row] != 'Unknown':
            if int(registration['County GEOID'][row]) in county_count:
                county_count[int(registration['County GEOID'][row])] += (623891 / 2542443)
            else:
                county_count[int(registration['County GEOID'][row])] = (623891 / 2542443)
    county_zip_dict = {}
    for row in zip_county.index:
        county_zip_dict[int(zip_county['Zip Code'][row])] = zip_county['County'][row]
    zip_count_dict = {}
    for key in county_zip_dict.keys():
        if key in county_zip_dict:
            if county_zip_dict[key] in code_dict:
                if code_dict[county_zip_dict[key]] in county_count:
                    zip_count_dict[key] = \
                        (county_count[code_dict[county_zip_dict[key]]],
                         population_dict[county_zip_dict[key]])
    land = pd.read_csv('Data/LandPrices.csv')
    electric = pd.read_csv('Data/ElectricityData.csv')
    econ = pd.read_csv('Data/EconData.csv')
    stations = pd.read_csv('Data/Station_Data.csv', low_memory=False)
    zipcodes = pd.read_csv('Data/Zipcodes.csv')
    sales_tax = pd.read_csv('Data/Sales Tax Data.csv').to_numpy()
    population = pd.read_csv('Data/Population.csv').to_numpy()
    valid_zips = pd.read_csv('Data/CA Zip Code List.csv').to_numpy()
    del electric['utility_name']
    del electric['state']
    del electric['service_type']
    del electric['ownership']
    del econ['country']
    ar = zipcodes.to_numpy()
    land_array = land.to_numpy()
    econ_data = econ.to_numpy()
    electric_data = electric.to_numpy()
    num_stats = 7
    d = {}
    columns = {'DCFC_$/hr': stations['DCFC $/Hr'],
               'DCFC_flat': stations['DCFC flat'], 'DCFC_$/kWh': stations['DCFC $/kWh'], 'State': stations['State']}
    station_data = pd.DataFrame(columns)
    station_data['is_na'] = station_data[station_data.columns].isnull().apply(lambda x: all(x), axis=1)
    station_data['ZIP'] = stations['ZIP']
    station_data['DCFC_count'] = stations['EV DC Fast Count']
    station_data['State'] = stations['State']
    L2_stations = np.zeros((len(station_data['ZIP']), 4))
    DCFC_stations = np.zeros((len(station_data['ZIP']), 4))
    for row in ar:
        if row[0] in valid_zips:
            d[row[0]] = np.zeros(num_stats + 3)
            d[row[0]][0], d[row[0]][1] = row[1], row[2]
    L2_counter = 0
    DCFC_counter = 0
    for ind in station_data.index:
        if int(station_data['ZIP'][ind]) in d:
            if not station_data['is_na'][ind] and station_data['State'][ind] == 'CA':
                if not pd.isnull(station_data['DCFC_count'][ind]):
                    DCFC_stations[DCFC_counter][0] = station_data['ZIP'][ind]
                    if not pd.isnull(station_data['DCFC_$/hr'][ind]):
                        DCFC_stations[DCFC_counter][1] = station_data['DCFC_$/hr'][ind]
                    if not pd.isnull(station_data['DCFC_flat'][ind]):
                        DCFC_stations[DCFC_counter][2] = station_data['DCFC_flat'][ind]
                    if not pd.isnull(station_data['DCFC_$/kWh'][ind]):
                        DCFC_stations[DCFC_counter][3] = station_data['DCFC_$/kWh'][ind]
                    DCFC_counter += 1
            d[int(station_data['ZIP'][ind])][6] += 1
    for row in land_array:
        if row[0] in valid_zips:
            if row[0] not in d:
                d[row[0]] = np.zeros(num_stats + 3)
            d[row[0]][2] = row[2]
    for row in electric_data:
        if row[0] in valid_zips:
            if row[0] not in d:
                d[row[0]] = np.zeros(num_stats + 3)
            d[row[0]][3] += row[2]
            d[row[0]][4] += 1
    econ_dict = {}
    for row in econ_data:
        econ_dict[(row[0], row[1])] = row[15]
    for row in population:
        if int(row[0]) in d:
            d[row[0]][8] = row[1]
    for row in sales_tax:
        if int(row[1]) in d:
            d[row[1]][7] = row[3]
    for row in ar:
        if (round(row[1]), round(row[2])) in econ_dict and row[0] in valid_zips:
            d[row[0]][5] = econ_dict[(round(row[1]), round(row[2]))]
    counter = 0
    for i in valid_zips:
        if i[0] in d:
            counter += 1
    temp = np.zeros(counter)
    counter = 0
    for i in valid_zips:
        if i[0] in d:
            temp[counter] = i
            counter += 1
    valid_zips = temp
    final_data = np.zeros((counter, num_stats))
    for i in range(len(valid_zips)):
        cur_zip = valid_zips[i]
        final_data[i][0] = cur_zip
        if d[cur_zip][2] != 0:
            final_data[i][2] = d[cur_zip][2]
        else:
            final_data[i][2] = zip_interpolate(2, cur_zip, d)
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
        if int(cur_zip) in zip_count_dict:
            final_data[i][6] = zip_count_dict[cur_zip][0] / zip_count_dict[cur_zip][1]
    edited_data = pd.DataFrame(final_data, columns=['ZIP', 'Electric Price', 'Land Value',
                                                    'Economic Activity', 'Station Count', 'Sales Tax', 'EV density'])
    edited_data.to_csv('Data/CA_model_input.csv')
    for row in final_data:
        d[row[0]] = row
    DCFC_data = np.zeros((len(DCFC_stations), num_stats + 1))
    for i in range(len(DCFC_data)):
        if DCFC_stations[i][0] != 0:
            DCFC_data[i][0] = DCFC_stations[i][2] + DCFC_stations[i][1] * .5 + DCFC_stations[i][3] * 20
            ref = d[int(DCFC_stations[i][0])]
            for j in range(num_stats - 1):
                DCFC_data[i][j + 1] = ref[j + 1]
            DCFC_data[i][num_stats] = int(DCFC_stations[i][0])
    counter = 0
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
               'Economic Activity', 'Station Count', 'Sales Tax', 'EV density', 'ZIP']
    DCFC_df = pd.DataFrame(DCFC_clean, columns=columns)
    DCFC_df.to_csv('Data/CA_stations.csv')
