# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 11:10:46 2023

@author: emerson.almeida
"""

import numpy as np
import pandas as pd


def read_file(filename):
    return pd.read_excel(filename, header=1)


def get_ab_locations(df):
    a = np.unique(df['A'])
    b = np.copy(a)    # indica o arranjo polo-dipolo
    n_leaps = len(a)
    return a, b, n_leaps


def get_mn_locations(df):
    m = np.array(df['M'].astype(float)).reshape(-1, 1)
    n = np.array(df['N'].astype(float)).reshape(-1, 1)
    return m, n


def calc_norm_v(df):
    norm_v = (df['Delta(mV)'] - df['SP(Mv)']) / df['I (mA)']
    return (-1.0 * norm_v.values).reshape(-1, 1)



def mn_array2str(mn_array):
    mn_string = str([list(i) for i in mn_array])
    mn_string = mn_string.replace(']]', '')
    mn_string = mn_string.replace('[[', '')
    mn_string = mn_string.replace('], [', '\n')
    mn_string = mn_string.replace(', ', '\t\t')
    mn_string = mn_string + '\n\n'
    return mn_string



def convert_format(input_filename, output_filename, elevation=1e-5):
    data = read_file(input_filename)
    loc_a, loc_b, n_groups = get_ab_locations(data)
    header = 'COMMON_CURRENT\n! general FORMAT\n' + f'{n_groups}\n'

    with open(output_filename, 'w') as outfile:
        outfile.write(header)

        for a, b in zip(loc_a, loc_b):
            ind = data[data['A'] == a].index
            n_levels = len(ind)
            curr_elev = np.ones((n_levels, 1)) * elevation
            
            ab_info = f'{a}\t\t{elevation}\t\t{b}\t\t{elevation}\t\t{n_levels}\n'
            outfile.write(ab_info)
            
            
            loc_m, loc_n = get_mn_locations(data.iloc[ind])
            readings = calc_norm_v(data.iloc[ind])
                    
            mn_info = np.concatenate((loc_m, curr_elev, 
                                      loc_n, curr_elev,
                                      readings, np.zeros((len(readings), 1))),
                                     axis=1)
            
            mn_info = mn_array2str(mn_info)
            outfile.write(mn_info)
            
    return None


convert_format('..\dados\C1.xlsx', '..\dados\CE1.dat')







"""

header = 'COMMON_CURRENT\n! general FORMAT\n' + f'{n_avancos}\n'

with open('CE1.dat', 'w') as outfile:
    outfile.write(header)

    for a, b in zip(loc_a, loc_b):
        ind = dados[dados['A'] == a].index
        niveis = len(ind)
        curr_elev = np.ones((niveis, 1)) * elevacao
        
        ab_info = f'{a}\t\t{elevacao}\t\t{b}\t\t{elevacao}\t\t{niveis}\n'
        outfile.write(ab_info)
        
        
        loc_m, loc_n = get_mn_locations(dados.iloc[ind])
        leituras = calc_norm_v(dados.iloc[ind])
                
        mn_info = np.concatenate((loc_m, curr_elev, 
                                  loc_n, curr_elev,
                                  leituras, np.zeros((len(leituras), 1))),
                                 axis=1)
        
        mn_info = mn_array2str(mn_info)
        outfile.write(mn_info)
"""    


