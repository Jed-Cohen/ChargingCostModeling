import numpy as np
import pandas as pd

if __name__ == '__main__':
    zip_county = pd.read_csv('Data/CA zip county.csv')
    stations = pd.read_csv('Data/CA DCFC.csv')
    registration = pd.read_csv('Data/ca_ev_registrations_public.csv', low_memory=False)
    print(registration)
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
                county_count[int(registration['County GEOID'][row])] += 1
            else:
                county_count[int(registration['County GEOID'][row])] = 1
    county_zip_dict = {}
    for row in zip_county.index:
        county_zip_dict[int(zip_county['Zip Code'][row])] = zip_county['County'][row]
    zip_count_dict = {}
    zip_population_dict = {}
    for key in county_zip_dict.keys():
        if key in county_zip_dict:
            if county_zip_dict[key] in code_dict:
                if code_dict[county_zip_dict[key]] in county_count:
                    zip_count_dict[key] = \
                        (county_count[code_dict[county_zip_dict[key]]], population_dict[county_zip_dict[key]])
    populations = np.zeros(len(stations['ZIP']))
    EV_num = np.zeros(len(stations['ZIP']))
    density = np.zeros(len(stations['ZIP']))
    for row in stations.index:
        populations[row] = zip_count_dict[stations['ZIP'][row]][1]
        EV_num[row] = zip_count_dict[stations['ZIP'][row]][0]
        density[row] = zip_count_dict[stations['ZIP'][row]][0] / zip_count_dict[stations['ZIP'][row]][1]
    stations['Population'] = populations
    stations['Number of Evs'] = EV_num
    stations['EV density'] = density
    stations.to_csv("Data/CA advanced.csv")
