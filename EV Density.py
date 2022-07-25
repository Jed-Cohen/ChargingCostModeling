import numpy as np
import pandas as pd


def correlation(column1, column2, data):
    return data[column1].corr(data[column2])


def check_zip(number, states):
    for state in states:
        if number in state['id']:
            return True
    return False


def zip_interpolate(col, cur_zip, zip_map):
    upper = cur_zip
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


def standardize(data):
    return (data - data.mean()) / data.std()


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def is_valid_zip(zipcode):
    try:
        zipcode = float(zipcode)
    except:
        return False
    if zipcode > 600:
        return True
    return False


def num_evs():
    registration_dict = {'CO': pd.read_csv('Data/Registration Data/co_ev_registrations_public.csv'),
                         'MN': pd.read_csv('Data/Registration Data/mn_ev_registrations_public.csv', low_memory=False),
                         'NJ': pd.read_csv('Data/Registration Data/nj_ev_registrations_public.csv', low_memory=False),
                         'NY': pd.read_csv('Data/Registration Data/ny_ev_registrations_public.csv'),
                         'OR': pd.read_csv('Data/Registration Data/or_ev_registrations_public.csv', low_memory=False),
                         'TX': pd.read_csv('Data/Registration Data/tx_ev_registrations_public.csv')}
                         # 'WA': pd.read_csv('Data/Registration Data/wa_ev_registrations_public.csv', low_memory=False)}
                         #'WI': pd.read_csv('Data/Registration Data/wi_ev_registrations_public.csv').
                         #'CT': pd.read_csv('Data/Registration Data/ct_ev_registrations_public.csv'),
                         # 'MI': pd.read_csv('Data/Registration Data/mi_ev_registrations_public.csv'),
    # 'VT': pd.read_csv('Data/Registration Data/vt_ev_registrations_public.csv'),}
    ev_sums = {}
    for key in registration_dict:
        state = registration_dict[key]
        for index in state.index:
            if is_valid_zip(state['ZIP Code'][index]):
                if state['ZIP Code'][index] not in ev_sums:
                    ev_sums[state['ZIP Code'][index]] = 0
                ev_sums[state['ZIP Code'][index]] += 1
    return ev_sums


if __name__ == '__main__':
    WA = pd.read_csv('Data/Zip Codes/WA zip codes.csv')
    OR = pd.read_csv('Data/Zip Codes/OR Zip Codes.csv')
    TX = pd.read_csv('Data/Zip Codes/TX Zip Codes.csv')
    NY = pd.read_csv('Data/Zip Codes/NY Zip Codes.csv')
    NJ = pd.read_csv('Data/Zip Codes/NJ Zip Codes.csv')
    MN = pd.read_csv('Data/Zip Codes/MN Zip Codes.csv')
    list = (WA, OR, TX, NY, NJ, MN)
    valid_states = ("CO", "MN", "NJ", "NY", "OR", "TX", "VT", "MI", "CT", "WI", "WA")
    land = pd.read_csv('Data/LandPrices.csv')
    electric = pd.read_csv('Data/ElectricityData.csv')
    econ = pd.read_csv('Data/EconData.csv')
    stations = pd.read_csv('Data/StationData - Copy.csv', low_memory=False)
    zipcodes = pd.read_csv('Data/Zipcodes.csv')
    sales_tax = pd.read_csv('Data/Sales Tax Data.csv').to_numpy()
    population = pd.read_csv('Data/Population.csv').to_numpy()
    del electric['utility_name']
    del electric['state']
    del electric['service_type']
    del electric['ownership']
    del econ['country']
    ar = zipcodes.to_numpy()
    land_array = land.to_numpy()
    econ_data = econ.to_numpy()
    electric_data = electric.to_numpy()
    num_stats = 9
    d = {}
    columns = {'L2_$/hr': stations['L2 $/hr'], 'L2_flat': stations['L2 flat'],
               'L2_$/kWh': stations['L2 $/kWh'], 'DCFC_$/hr': stations['DCFC $/Hr'],
               'DCFC_flat': stations['DCFC flat'], 'DCFC_$/kWh': stations['DCFC $/kWh'], 'State': stations['State']}
    station_data = pd.DataFrame(columns)
    station_data['is_na'] = station_data[station_data.columns].isnull().apply(lambda x: all(x), axis=1)
    station_data['ZIP'] = stations['ZIP']
    station_data['L2_count'] = stations['EV Level2 EVSE Num']
    station_data['DCFC_count'] = stations['EV DC Fast Count']
    station_data['State'] = stations['State']
    L2_stations = np.zeros((len(station_data['ZIP']), 4))
    DCFC_stations = np.zeros((len(station_data['ZIP']), 4))
    for row in ar:
        d[row[0]] = np.zeros(num_stats + 3)
        d[row[0]][0], d[row[0]][1] = row[1], row[2]
    L2_counter = 0
    DCFC_counter = 0
    for ind in station_data.index:
        if int(station_data['ZIP'][ind]) in d:
            if not station_data['is_na'][ind] and station_data['State'][ind] in valid_states:
                if not pd.isnull(station_data['L2_count'][ind]):
                    L2_stations[L2_counter][0] = station_data['ZIP'][ind]
                    if not pd.isnull(station_data['L2_$/hr'][ind]):
                        L2_stations[L2_counter][1] = station_data['L2_$/hr'][ind]
                    if not pd.isnull(station_data['L2_flat'][ind]):
                        L2_stations[L2_counter][2] = station_data['L2_flat'][ind]
                    if not pd.isnull(station_data['L2_$/kWh'][ind]):
                        L2_stations[L2_counter][3] = station_data['L2_$/kWh'][ind]
                    L2_counter += 1
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
    registration = pd.read_csv('Data/registration.csv')
    for i in registration.index:
        if registration['ZIP'][i] not in d:
            d[registration['ZIP'][i]] = np.zeros(num_stats + 3)
        d[registration['ZIP'][i]][9] = registration['Num EVs'][i]
    for row in land_array:
        if row[0] in d:
            d[row[0]][2] = row[2]
    for row in electric_data:
        if row[0] in d:
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
        if (round(row[1]), round(row[2])) in econ_dict:
            d[row[0]][5] = econ_dict[(round(row[1]), round(row[2]))]
    final_data = np.zeros((len(ar), num_stats))
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
        if d[cur_zip][8] != 0:
            final_data[i][6] = d[cur_zip][9] / d[cur_zip][8]
            final_data[i][7] = d[cur_zip][8]
            final_data[i][8] = d[cur_zip][9]
        else:
            final_data[i][6] = 0
    edited_data = pd.DataFrame(final_data, columns=['ZIP', 'Electricity Price', 'Land Value',
                                                    'Economic Activity', 'Station Count', 'Density', 'Sales Tax',
                                                    'Population', 'Num Evs'])
    land_electricity_corr = correlation('Land Value', 'Electricity Price', edited_data)
    land_econ_corr = correlation('Land Value', 'Economic Activity', edited_data)
    land_stations_corr = correlation('Land Value', 'Station Count', edited_data)
    electricity_econ_corr = correlation('Electricity Price', 'Economic Activity', edited_data)
    electricity_stations_corr = correlation('Electricity Price', 'Station Count', edited_data)
    econ_stations_corr = correlation('Station Count', 'Economic Activity', edited_data)
    print('Correlation')
    print('land value - electricity cost:', land_electricity_corr)
    print('land value - economic activity:', land_econ_corr)
    print('land value - number of stations:', land_stations_corr)
    print('electricity cost - economic activity:', electricity_econ_corr)
    print('electricity cost - number of stations:', electricity_stations_corr)
    print('economic activity - number of stations:', econ_stations_corr)
    L2_data = np.zeros((len(L2_stations), num_stats + 1))
    for row in final_data:
        d[row[0]] = row
    for i in range(len(L2_data)):
        if L2_stations[i][0] != 0:
            ref = d[int(L2_stations[i][0])]
            L2_data[i][0] = L2_stations[i][2] + L2_stations[i][1] * 3 + L2_stations[i][3] * 20
            for j in range(num_stats - 1):
                L2_data[i][j + 1] = ref[j + 1]
            L2_data[i][num_stats] = int(L2_stations[i][0])
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
        if 0 not in (row[0], row[1], row[2], row[3], row[4], row[6]):
            counter += 1
    DCFC_clean = np.zeros((counter, num_stats + 1))
    counter = 0
    for row in DCFC_data:
        if 0 not in (row[0], row[1], row[2], row[3], row[4], row[6]):
            DCFC_clean[counter] = row
            counter += 1
    counter = 0
    for row in L2_data:
        if 0 not in (row[0], row[1], row[2], row[3], row[4], row[6]):
            counter += 1
    L2_clean = np.zeros((counter, num_stats + 1))
    counter = 0
    for row in L2_data:
        if 0 not in (row[0], row[1], row[2], row[3], row[4], row[6]):
            L2_clean[counter] = row
            counter += 1
    columns = ['Converted Cost', 'Electric Price', 'Land Value',
               'Economic Activity', 'Station Count', 'Sales Tax', 'Density', 'Population', 'Num EVs', 'ZIP']
    DCFC_df = pd.DataFrame(DCFC_clean, columns=columns)
    L2_df = pd.DataFrame(L2_clean, columns=columns)
    DCFC_standardized = pd.DataFrame(columns=columns)
    L2_standardized = pd.DataFrame(columns=columns)
    DCFC_normalized = pd.DataFrame(columns=columns)
    L2_normalized = pd.DataFrame(columns=columns)
    for column in columns:
        if column != 'ZIP':
            DCFC_standardized[column] = standardize(DCFC_df[column])
            L2_standardized[column] = standardize(L2_df[column])
            DCFC_normalized[column] = normalize(DCFC_df[column])
            L2_normalized[column] = normalize(L2_df[column])
        else:
            DCFC_normalized[column] = DCFC_df[column]
            DCFC_standardized[column] = DCFC_df[column]
            L2_standardized[column] = L2_df[column]
            L2_normalized[column] = L2_df[column]
    DCFC_df.to_csv('Data/DCFC density data.csv')
    DCFC_normalized.to_csv('Data/DCFC normalized density.csv')
    DCFC_standardized.to_csv('Data/DCFC standardized density.csv')
    L2_normalized.to_csv('Data/L2 normalized density.csv')
    L2_standardized.to_csv('Data/L2 standardized density.csv')
    print('Correlation- level 2')
    print('Charging Cost - Electricity Cost:', correlation('Electric Price', 'Converted Cost', L2_df))
    print('Charging Cost - Land Value:', correlation('Land Value', 'Converted Cost', L2_df))
    print('Charging Cost - Economic Activity:', correlation('Economic Activity', 'Converted Cost', L2_df))
    print('Charging Cost - Station Count:', correlation('Station Count', 'Converted Cost', L2_df))
    print('DCFC')
    print('Charging Cost - Electricity Cost:', correlation('Electric Price', 'Converted Cost', DCFC_df))
    print('Charging Cost - Land Value:', correlation('Land Value', 'Converted Cost', DCFC_df))
    print('Charging Cost - Economic Activity:', correlation('Economic Activity', 'Converted Cost', DCFC_df))
    print('Charging Cost - Station Count:', correlation('Station Count', 'Converted Cost', DCFC_df))
