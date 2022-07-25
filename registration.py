import numpy as np
import pandas as pd

if __name__ == '__main__':
    CO = pd.read_csv('Data/Registration Data/co_ev_registrations_public.csv')
    MN = pd.read_csv('Data/Registration Data/mn_ev_registrations_public.csv', low_memory=False)
    NJ = pd.read_csv('Data/Registration Data/nj_ev_registrations_public.csv', low_memory=False)
    NY = pd.read_csv('Data/Registration Data/ny_ev_registrations_public.csv')
    OR = pd.read_csv('Data/Registration Data/or_ev_registrations_public.csv', low_memory=False)
    TX = pd.read_csv('Data/Registration Data/tx_ev_registrations_public.csv')
    WA = pd.read_csv('Data/Registration Data/wa_ev_registrations_public.csv', low_memory=False)
    WI = pd.read_csv('Data/Registration Data/wi_ev_registrations_public.csv', low_memory=False)
    VT = pd.read_csv('Data/Registration Data/vt_ev_registrations_public.csv', low_memory=False)
    MI = pd.read_csv('Data/Registration Data/mi_ev_registrations_public.csv', low_memory=False)
    CT = pd.read_csv('Data/Registration Data/ct_ev_registrations_public.csv', low_memory=False)
    zip_list = {}
    states = (CO, MN, NJ, NY, TX, WA, WI, VT, MI, CT)
    ratios = (50768/1075022, 26109/124655, 63032/309872, 100580/3012592, 117377/1033202, 90298/2456028, 6654/29120,
              6449/33140, 15665/43626, 6138/11439)
    for state in states:
        for i in state.index:
            if str(state['ZIP Code'][i]).isnumeric() and not pd.isna(state['ZIP Code'][i]) and \
                    int(state['ZIP Code'][i]) not in zip_list:
                zip_list[int(state['ZIP Code'][i])] = 0
    for i in OR.index:
        if not pd.isna(OR['ZIP Code'][i]) and int(OR['ZIP Code'][i]) not in zip_list:
            zip_list[int(OR['ZIP Code'][i])] = 0
    for state in range(len(states)):
        for i in states[state].index:
            if str(states[state]['ZIP Code'][i]).isnumeric():
                zip_list[int(states[state]['ZIP Code'][i])] += ratios[state]
    for i in OR.index:
        if not pd.isna(OR['ZIP Code'][i]):
            zip_list[int(OR['ZIP Code'][i])] += (35676/65683)
    ar = np.zeros((len(zip_list.keys()), 2))
    counter = 0
    for key in zip_list:
        ar[counter][0] = key
        ar[counter][1] = zip_list[key]
        counter += 1
    pd.DataFrame(ar, columns=("ZIP", "Num EVs")).to_csv('Data/registration.csv')
