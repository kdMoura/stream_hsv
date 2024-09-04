#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for model learning and test.

@author: de Moura, K.
"""
import subprocess
import sys
import os

def run_script(script_name, args):
    command = [sys.executable, '-m', script_name] + args
    subprocess.run(command)
    
def batch_training():
    home_dir = os.path.expanduser('~')
    print("--- BATCH TRAINING ---")
    
    print('Training SGD and SVM on each GPDS-S dissimilarity training dataset')
    for m in ['sgd','svm']:
        for n_u in [2, 5, 10, 50, 581]:  # Number of users
            for n_g in [2, 6, 12]:  # Number of genuine signatures per user
                fname = f'u{n_u}_g{n_g}'
                parameters = [
                    'batch_train', 
                    '--f-train-path', os.path.join(home_dir, f'stream_hsv/databases/training/{fname}'), 
                    '--model', m,
                    '--f-output-path', os.path.join(home_dir,f'stream_hsv/models/{m}/{fname}')
                ]
                run_script('shsv.model', parameters)
    print("-----------------------------------")

def batch_test():
    home_dir = os.path.expanduser('~')
    print("--- BATCH TEST ---")
    
    print('Classifying GPDS-S test data with SGD and SVM')
    for m in ['sgd','svm']:
        for n_u in [2, 5, 10, 50, 581]:  # Number of users
            for n_g in [2, 6, 12]:  # Number of genuine signatures per user
                fname = f'u{n_u}_g{n_g}'
                parameters = [
                    'batch_test', 
                    '--f-test-path', os.path.join(home_dir,'stream_hsv/databases/test/sgpds'),
                    '--f-model-path', os.path.join(home_dir,f'stream_hsv/models/{m}/{fname}'),
                    '--f-output-path', os.path.join(home_dir,f'stream_hsv/predictions/batch/batch_{m}_{fname}')
                ]
                run_script('shsv.model', parameters)
    print("-----------------------------------")
            
def prequential_sgpds():
    home_dir = os.path.expanduser('~')
    print("--- PREQUENTIAL ON GPDS-S STREAM ---")
    chunk_size = str(300*2)
    for m in ['sgd','svm']:
        for n_u in [2, 5, 10, 50, 581]:  # Number of users
            for n_g in [2, 6, 12]:  # Number of genuine signatures per user
                for n_ref in [1, 2, 5, 12]:
                    for fnumber in range(5):
                        fname = f'u{n_u}_g{n_g}'
                        parameters = [
                            'prequential', 
                            '--f-test-path', os.path.join(home_dir,'stream_hsv/databases/test/sgpds'),
                            '--f-model-path', os.path.join(home_dir,f'stream_hsv/models/{m}/{fname}'),
                            '--f-output-path', os.path.join(home_dir,f'stream_hsv/predictions/stream_sgpds/stream_{m}_{fname}_r{n_ref}_ch{chunk_size}'),
                            '--chunk-size', chunk_size,
                            '--n-ref',str(n_ref),
                            '--file-number', str(fnumber)
                        ]
                        run_script('shsv.model', parameters)
    print("-----------------------------------")

def prequential_cedar_mcyt():
    home_dir = os.path.expanduser('~')
    print("--- PREQUENTIAL ON CEDAR+MCYT STREAM ---")
    chunk_size = '200'
    for m in ['sgd','svm']:
        for n_u in [2, 5, 10, 50, 581]:  # Number of users
            for n_g in [2, 6, 12]:  # Number of genuine signatures per user
                for n_ref in [1, 2, 5, 10]:
                    for fnumber in range(5):
                        fname = f'u{n_u}_g{n_g}'
                        parameters = [
                            'prequential', 
                            '--f-test-path', os.path.join(home_dir,'stream_hsv/databases/test/cedar'),
                            os.path.join(home_dir,'stream_hsv/databases/test/mcyt'),
                            '--f-model-path', os.path.join(home_dir,f'stream_hsv/models/{m}/{fname}'),
                            '--f-output-path', os.path.join(home_dir,f'stream_hsv/predictions/stream_mixed/stream_{m}_{fname}_r{n_ref}_ch{chunk_size}'),
                            '--chunk-size', chunk_size,
                            '--n-ref',str(n_ref),
                            '--file-number', str(fnumber),
                            '--stream-order', 'random'
                        ]
                        run_script('shsv.model', parameters)
    print("-----------------------------------") 
    
if __name__ == '__main__':
    
    batch_training()
    batch_test()
    prequential_sgpds()
    prequential_cedar_mcyt()