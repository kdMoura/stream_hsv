#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration for evaluation of results.

@author: de Moura, K.
"""
import subprocess
import sys
import os

def run_script(script_name, args):
    command = [sys.executable, '-m', script_name] + args
    subprocess.run(command)

def batch_eval():
    home_dir = os.path.expanduser('~')
    print("--- BATCH EVALUATION ---")
       
    f_input_path = os.path.join(home_dir,'stream_hsv/predictions/batch')
    folders = os.listdir(f_input_path)
    for f in folders:
        parameters = ['batch', 
                                        '--f-input-path', os.path.join(f_input_path,f),
                                        '--f-output-path', os.path.join(home_dir,'stream_hsv/metrics/batch'),
                                        '--n-ref','1', '2', '3', '5', '10', '12',
                                        '--forgery', 'skilled',
                                        '--thr-type', 'global',
                                        '--fusions', 'max'
        ]
        run_script('shsv.evaluation', parameters)
    print("-----------------------------------")               
    
def stream_sgdps_eval():
    home_dir = os.path.expanduser('~')
    print("--- GPDS-S STREAM EVALUATION ---")
       
    f_input_path = os.path.join(home_dir,'stream_hsv/predictions/stream_sgpds')
    folders = os.listdir(f_input_path)
    for f in folders:
        parameters = ['stream', 
                                        '--f-input-path', os.path.join(f_input_path,f),
                                        '--f-output-path', os.path.join(home_dir,'stream_hsv/metrics/stream_sgpds'),
                                        '--window-size', '400',
                                        '--window-step', '400',
                                        '--forgery','skilled',
                                        '--thr-type','global',
                                        '--fusions', 'max'
        ]
        run_script('shsv.evaluation', parameters)
    print("-----------------------------------")  

def stream_cedar_mcyt_eval():
    home_dir = os.path.expanduser('~')
    print("--- CEDAR+MCYT STREAM EVALUATION ---")
       
    f_input_path = os.path.join(home_dir,'stream_hsv/predictions/stream_mixed')
    folders = os.listdir(f_input_path)
    for f in folders:
        parameters = ['stream', 
                                        '--f-input-path', os.path.join(f_input_path,f),
                                        '--f-output-path', os.path.join(home_dir,'stream_hsv/metrics/stream_mixed'),
                                        '--window-size', '200',
                                        '--window-step', '200',
                                        '--forgery','skilled',
                                        '--thr-type','global',
                                        '--fusions', 'max'
        ]
        run_script('shsv.evaluation', parameters)
    print("-----------------------------------")  
    
if __name__ == '__main__':
    batch_eval()
    stream_sgdps_eval()
    stream_cedar_mcyt_eval()
    