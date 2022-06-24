import numpy as np
import pandas as pd


def correlation(column1, column2, data):
    return data[column1].corr(data[column2])


if __name__ == '__main__':
    land = pd.read_csv('Data/LandPrices.csv')
    electric = pd.read_csv('Data/ElectricityData.csv')
    econ = pd.read_csv('Data/EconData.csv')
    stations = pd.read_csv('Data/StationData.csv', low_memory=False)
    zipcodes = pd.read_csv('Data/Zipcodes.csv')
    del electric['utility_name']
    del electric['state']
    del electric['service_type']
    del electric['ownership']
    del econ['country']
    ar = zipcodes.to_numpy()
    land_array = land.to_numpy()
    econ_data = econ.to_numpy()
    electric_data = electric.to_numpy()
    d = {}
    columns = {'L2_$/hr': stations['L2 $/hr'], 'L2_flat': stations['L2 flat'],
               'L2_$/kWh': stations['L2 $/kWh'], 'DCFC_$/hr': stations['DCFC $/Hr'],
               'DCFC_flat': stations['DCFC flat'], 'DCFC_$/kWh': stations['DCFC $/kWh']}
    station_data = pd.DataFrame(columns)
    station_data['is_na'] = station_data[station_data.columns].isnull().apply(lambda x: all(x), axis=1)
    station_data['ZIP'] = stations['ZIP']
    station_data['L2_count'] = stations['EV Level2 EVSE Num']
    station_data['DCFC_count'] = stations['EV DC Fast Count']
    L2_stations = np.zeros((len(station_data['ZIP']), 4))
    DCFC_stations = np.zeros((len(station_data['ZIP']), 4))
    for row in ar:
        d[row[0]] = np.array([row[1], row[2], 0, 0, 0, 0, 0])
    L2_counter = 0
    DCFC_counter = 0
    for ind in station_data.index:
        if int(station_data['ZIP'][ind]) in d:
            if not station_data['is_na'][ind]:
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
    for row in ar:
        if (round(row[1]), round(row[2])) in econ_dict:
            d[row[0]][5] = econ_dict[(round(row[1]), round(row[2]))]
    final_data = np.zeros((len(ar), 5))
    for i in range(len(ar)):
        final_data[i][0] = ar[i][0]
        final_data[i][1] = d[ar[i][0]][2]
        if d[ar[i][0]][4] != 0:
            final_data[i][2] = d[ar[i][0]][3] / d[ar[i][0]][4]
        final_data[i][3] = d[ar[i][0]][5]
        final_data[i][4] = d[ar[i][0]][4]
    nonzero_count = 0
    for row in final_data:
        if 0 not in row:
            nonzero_count += 1
    full_data = np.zeros((nonzero_count, 5))
    row_pointer = 0
    for i in range(nonzero_count):
        while 0 in final_data[row_pointer]:
            row_pointer += 1
        full_data[i] = final_data[row_pointer]
        row_pointer += 1
    edited_data = pd.DataFrame(full_data, columns=['ZIP', 'Land Value',
                                                   'Electricity Price', 'Economic Activity', 'Station Count'])
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
    L2_data = np.zeros((len(L2_stations), 5))
    for i in range(len(L2_data)):
        if L2_stations[i][0] != 0:
            L2_data[i][0] = L2_stations[i][2] + L2_stations[i][1] * 3 + L2_stations[i][3] * 20
            if d[int(L2_stations[i][0])][4] != 0:
                L2_data[i][1] = d[int(L2_stations[i][0])][3] / d[int(L2_stations[i][0])][4]
            L2_data[i][2] = d[int(L2_stations[i][0])][2]
            L2_data[i][3] = d[int(L2_stations[i][0])][5]
            L2_data[i][4] = d[int(L2_stations[i][0])][4]
    L2_data = np.zeros((len(L2_stations), 5))
    for i in range(len(L2_data)):
        if L2_stations[i][0] != 0:
            L2_data[i][0] = L2_stations[i][2] + L2_stations[i][1] * 3 + L2_stations[i][3] * 20
            if d[int(L2_stations[i][0])][4] != 0:
                L2_data[i][1] = d[int(L2_stations[i][0])][3] / d[int(L2_stations[i][0])][4]
            L2_data[i][2] = d[int(L2_stations[i][0])][2]
            L2_data[i][3] = d[int(L2_stations[i][0])][5]
            L2_data[i][4] = d[int(L2_stations[i][0])][4]
    DCFC_data = np.zeros((len(DCFC_stations), 5))
    for i in range(len(DCFC_data)):
        if DCFC_stations[i][0] != 0:
            DCFC_data[i][0] = DCFC_stations[i][2] + DCFC_stations[i][1] * .5 + DCFC_stations[i][3] * 20
            if d[int(DCFC_stations[i][0])][4] != 0:
                DCFC_data[i][1] = d[int(DCFC_stations[i][0])][3] / d[int(DCFC_stations[i][0])][4]
            DCFC_data[i][2] = d[int(DCFC_stations[i][0])][2]
            DCFC_data[i][3] = d[int(DCFC_stations[i][0])][5]
            DCFC_data[i][4] = d[int(DCFC_stations[i][0])][4]
    counter = 0
    for row in DCFC_data:
        if 0 not in row:
            counter += 1
    DCFC_clean = np.zeros((counter, 5))
    counter = 0
    for row in DCFC_data:
        if 0 not in row:
            DCFC_clean[counter] = row
            counter += 1
    counter = 0
    for row in L2_data:
        if 0 not in row:
            counter += 1
    L2_clean = np.zeros((counter, 5))
    counter = 0
    for row in L2_data:
        if 0 not in row:
            L2_clean[counter] = row
            counter += 1
    DCFC_df = pd.DataFrame(DCFC_clean, columns=['Converted Cost', 'Electric Price',
                                               'Land Value', 'Economic Activity', 'Station Count'])
    DCFC_df.to_csv('Data/DCFCfinal.csv')
    L2_df = pd.DataFrame(L2_clean, columns=['Converted Cost', 'Electric Price',
                                           'Land Value', 'Economic Activity', 'Station Count'])
    L2_df.to_csv('Data/level2final.csv')
