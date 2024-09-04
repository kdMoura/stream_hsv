#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for data generation.

@author: de Moura, K.
"""
import subprocess
import sys
import os

def run_script(script_name, args):
    command = [sys.executable, '-m', script_name] + args
    subprocess.run(command)
    
def data_generation(gpdss_npz_path, mcyt_npz_path, cedar_npz_path):
    
    home_dir = os.path.expanduser('~')
    f_output_path = os.path.join(home_dir, 'stream_hsv/databases')
    
    print("--- DATA GENERATION ---")
    
    print('Creating GPDS-S dissimilarity training datasets')
    for n_u in [2, 5, 10, 50, 581]:  # Number of users
        for n_g in [2, 6, 12]:  # Number of genuine signatures per user
            
            fname = f'u{n_u}_g{n_g}'
            output_folder = os.path.join(f_output_path, 'training', fname)
            
            parameters = [
                'training_generation',
                '--n-data', '5',
                '--input-data', gpdss_npz_path,
                '--f-output-path', output_folder,
                '--n-gen', f'{n_g}',
                '--include-users', '300', f'{300+n_u}'
            ]
            
          
            run_script('shsv.data', parameters)
    print("-----------------------------------")
    
   
    print('Creating GPDS-S dissimilarity test data')
    parameters = [
        'test_generation',
        '--n-data', '5',
        '--input-data', gpdss_npz_path,
        '--f-output-path', os.path.join(f_output_path, 'test', 'sgpds'),
        '--n-ref', '12',
        '--n-query', '10',
        '--include-users', '0', '300'
    ]
    run_script('shsv.data', parameters)
    print("-----------------------------------")
    
    
    print('Creating MCYT dissimilarity test data')
    parameters = [
        'test_generation',
        '--n-data', '5',
        '--input-data', mcyt_npz_path,
        '--f-output-path', os.path.join(f_output_path, 'test', 'mcyt'),
        '--n-ref', '10',
        '--n-query', '5'
    ]
    run_script('shsv.data', parameters)
    print("-----------------------------------")
    
    
    print('Creating CEDAR dissimilarity test data')
    parameters = [
        'test_generation',
        '--n-data', '5',
        '--input-data', cedar_npz_path,
        '--f-output-path', os.path.join(f_output_path, 'test', 'cedar'),
        '--n-ref', '12',  
        '--n-query', '10'
    ]
    run_script('shsv.data', parameters)
    print("-----------------------------------")
  

if __name__ == '__main__':
   
    # Add the path to the .NPZ containing the extracted features  
    home_dir = os.path.expanduser('~')
    # GPDS Synthetic
    gpdss_npz_path = os.path.join(home_dir, 'stream_hsv/databases/extracted_features/sgpds_signets.npz')
    # MCYT
    mcyt_npz_path = os.path.join(home_dir,  'stream_hsv/databases/extracted_features/mcyt_signets.npz')
    # CEDAR
    cedar_npz_path = os.path.join(home_dir, 'stream_hsv/databases/extracted_features/cedar_signets.npz')

    # Create datasets
    data_generation(gpdss_npz_path, mcyt_npz_path, cedar_npz_path)