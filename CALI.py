import numpy as np
import pandas as pd


def correlation(column1, column2, data):
    return data[column1].corr(data[column2])


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
        if lower <= 599:
            lower_factor = 0
            lower = 98107
    return (zip_map[upper][col] * upper_factor + zip_map[lower][col] * lower_factor) / (upper_factor + lower_factor)


def standardize(data):
    return (data - data.mean()) / data.std()


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


def is_valid_zip(zipcode):
    try:
        zipcode += 1
        zipcode -= 1
    except:
        return False
    if zipcode > 600:
        return True
    return False


def num_evs():
    registration_dict = {'CO': pd.read_csv('Data/Registration Data/co_ev_registrations_public.csv'),
                         'CT': pd.read_csv('Data/Registration Data/ct_ev_registrations_public.csv'),
                         'MI': pd.read_csv('Data/Registration Data/mi_ev_registrations_public.csv'),
                         'MN': pd.read_csv('Data/Registration Data/mn_ev_registrations_public.csv', low_memory=False),
                         'NJ': pd.read_csv('Data/Registration Data/nj_ev_registrations_public.csv', low_memory=False),
                         'NY': pd.read_csv('Data/Registration Data/ny_ev_registrations_public.csv'),
                         'OR': pd.read_csv('Data/Registration Data/or_ev_registrations_public.csv', low_memory=False),
                         'TX': pd.read_csv('Data/Registration Data/tx_ev_registrations_public.csv'),
                         'VT': pd.read_csv('Data/Registration Data/vt_ev_registrations_public.csv'),
                         'WA': pd.read_csv('Data/Registration Data/wa_ev_registrations_public.csv', low_memory=False),
                         'WI': pd.read_csv('Data/Registration Data/wi_ev_registrations_public.csv')}
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
    zip_county = pd.read_csv('Data/zip_code_database.csv')
    # County GEOID
    CA = pd.read_csv('Data/Registration Data/ca_ev_registrations_public.csv', low_memory=False), 623891/2542443
    MT = pd.read_csv('Data/Registration Data/mt_ev_registrations_public.csv', low_memory=False), 2895/5730
    TN = pd.read_csv('Data/Registration Data/tn_ev_registrations_public.csv', low_memory=False), 18451/146945
    VA = pd.read_csv('Data/Registration Data/va_ev_registrations_public.csv', low_memory=False), 44344/126989
    # County Name
    FL = pd.read_csv('Data/Registration Data/fl_ev_registrations_public.csv', low_memory=False), 90165/353974
    # ZIP
    CO = pd.read_csv('Data/Registration Data/co_ev_registrations_public.csv'), 50768/1075022
    MN = pd.read_csv('Data/Registration Data/mn_ev_registrations_public.csv', low_memory=False), 26109/124655
    NJ = pd.read_csv('Data/Registration Data/nj_ev_registrations_public.csv', low_memory=False), 63032/309872
    NY = pd.read_csv('Data/Registration Data/ny_ev_registrations_public.csv'), 100580/3012592
    OR = pd.read_csv('Data/Registration Data/or_ev_registrations_public.csv', low_memory=False), 35676/65681
    TX = pd.read_csv('Data/Registration Data/tx_ev_registrations_public.csv'), 117377/1033202
    WA = pd.read_csv('Data/Registration Data/wa_ev_registrations_public.csv', low_memory=False), 90298/2456028
    WI = pd.read_csv('Data/Registration Data/wi_ev_registrations_public.csv', low_memory=False), 6654/29120
    VT = pd.read_csv('Data/Registration Data/vt_ev_registrations_public.csv', low_memory=False), 6449/33140
    MI = pd.read_csv('Data/Registration Data/mi_ev_registrations_public.csv', low_memory=False), 15655/43628
    CT = pd.read_csv('Data/Registration Data/ct_ev_registrations_public.csv', low_memory=False), 6138/11405
    zip_states = (CO, MN, NJ, NY, TX, WA, WI, VT, MI, CT)
    Codes = pd.read_csv('Data/County Codes.csv')
    population = pd.read_csv('Data/Population.csv').to_numpy()
    code_dict = {}
    zip_population_dict = {}
    for row in population:
        if int(row[0]) not in zip_population_dict:
            zip_population_dict[row[0]] = row[1]
    county_population_dict = {}
    for row in Codes.index:
        code_dict[Codes['County'][row]] = Codes['Code'][row]
    county_count = {}
    for row in FL[0].index:
        if FL[0]['County'][row] != 'Unknown County':
            if FL[0]['County'][row] not in county_count:
                county_count[FL[0]['County'][row]] = 0
            county_count[FL[0]['County'][row]] += FL[1]
    for state in (CA, MT, TN, VA):
        for row in state[0].index:
            if state[0]['County GEOID'][row] != 'Unknown':
                if int(state[0]['County GEOID'][row]) not in county_count:
                    county_count[int(state[0]['County GEOID'][row])] = 0
                county_count[int(state[0]['County GEOID'][row])] += state[1]
    county_zip_dict = {}
    for row in zip_county.index:
        county_zip_dict[int(zip_county['zip'][row])] = "".join(zip_county['county'][row].split()[:-1])
    zip_count_dict = {}
    for key in county_zip_dict:
        county = county_zip_dict[key]
        if county not in county_population_dict:
            county_population_dict[county] = 0
        if key in zip_population_dict:
            county_population_dict[county] += zip_population_dict[key]
    for key in county_zip_dict.keys():
        if key in county_zip_dict:
            if county_zip_dict[key] in code_dict:
                if code_dict[county_zip_dict[key]] in county_count:
                    zip_count_dict[key] = \
                        (county_count[code_dict[county_zip_dict[key]]],
                         county_population_dict[county_zip_dict[key]])
    zip_reg_dict = {}
    land = pd.read_csv('Data/LandPrices.csv')
    electric = pd.read_csv('Data/ElectricityData.csv')
    econ = pd.read_csv('Data/EconData.csv')
    stations = pd.read_csv('Data/StationData - Copy.csv', low_memory=False)
    zipcodes = pd.read_csv('Data/Zipcodes.csv')
    sales_tax = pd.read_csv('Data/Sales Tax Data.csv').to_numpy()
    population = pd.read_csv('Data/Population.csv').to_numpy()
    densities = pd.read_csv('Data/State Density.csv')
    state_density = {}
    for row in densities.index:
        state_density[densities['State'][row]] = densities['Density'][row]
    zip_density_dict = {}
    del electric['utility_name']
    del electric['state']
    del electric['service_type']
    del electric['ownership']
    del econ['country']
    ar = zipcodes.to_numpy()
    land_array = land.to_numpy()
    econ_data = econ.to_numpy()
    electric_data = electric.to_numpy()
    num_stats = 8
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
            if not station_data['is_na'][ind] and station_data['State'][ind]:
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
        if row[0] not in d:
            d[row[0]] = np.zeros(num_stats + 3)
        d[row[0]][2] = row[2]
    for row in electric_data:
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
    for state in zip_states:
        for row in state[0].index:
            cur_zip = state[0]['ZIP Code'][row]
            if str(cur_zip).isnumeric():
                if cur_zip in d:
                    d[cur_zip][9] += (state[1] / d[cur_zip][8])
    for i in OR[0].index:
        cur_zip = OR[0]['ZIP Code'][i]
        if not pd.isna(cur_zip):
            if cur_zip in d:
                d[cur_zip][9] += (OR[1] / d[cur_zip][8])
    for row in zip_county.index:
        if zip_county['state'][row] not in ("CA", "CO", "CT", "FL", "MI", "MN", "MT", "NJ", "NY", "OR", "TN"
                                            , "TX", "VA", "VT", "WA", "WI"):
            if zip_county['state'][row] in state_density:
                d[zip_county['zip'][row]][9] = state_density[zip_county['state'][row]]
    for row in ar:
        if (round(row[1]), round(row[2])) in econ_dict and row[0] in d:
            d[row[0]][5] = econ_dict[(round(row[1]), round(row[2]))]
    final_data = np.zeros((len(d.keys()), num_stats))
    i = 0
    for key in d:
        cur_zip = key
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
        try:
            final_data[i][6] = zip_reg_dict[cur_zip] / d[cur_zip][8]
        except:
            print(cur_zip)
        if int(cur_zip) in zip_count_dict:
            final_data[i][6] = zip_count_dict[cur_zip][0] / zip_count_dict[cur_zip][1]
        if int(cur_zip) in zip_density_dict:
            final_data[i][6] = zip_density_dict[cur_zip]
        i += 1
    edited_data = pd.DataFrame(final_data, columns=['ZIP', 'Electric Price', 'Land Value',
                                                    'Economic Activity', 'Station Count', 'Sales Tax', 'EV density'])
    edited_data.to_csv('Data/Zip Density.csv')
