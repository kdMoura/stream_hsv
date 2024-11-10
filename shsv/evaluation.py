#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch and stream evaluation.

@author: de Moura, K.
"""

from sklearn import metrics
import numpy as np
import pandas as pd
import os
import json
import argparse
from typing import Union, Tuple, List
import re

def eer(
    true_labels: np.ndarray[int],
    predictions: np.ndarray[float],
    pos_label: int,
    return_threshold: bool = False
) -> Union[float, Tuple[float, float]]:
    """
   Computes the Equal Error Rate (EER) for a binary classification model.

   Parameters:
   ----------
   true_labels : np.ndarray[int]
       Ground truth binary labels, where each label is either the positive or negative class.
   
   predictions : np.ndarray[float]
       Predicted probabilities or decision function scores from the classifier. 
   
   pos_label : int
       The label of the positive class (i.e., the class for which we are interested in calculating the EER).
   
   return_threshold : bool, optional (default=False)
       If True, the function will return a tuple containing the EER and the corresponding threshold at which the EER occurs.
       If False, only the EER will be returned.

   Returns:
   -------
   Union[float, Tuple[float, float]]
       If `return_threshold` is False, returns the EER as a float.
       If `return_threshold` is True, returns a tuple containing the EER and the corresponding threshold.

   """
    fpr, tpr, thresholds = metrics.roc_curve(
        np.array(true_labels),
        np.array(predictions),
        pos_label=pos_label,
        drop_intermediate=False
    )
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    
    if return_threshold:
        return eer, thresholds[min_index]
    return eer
    
def fusion_function(
    file_path: str,
    n_ref: int,
    forgery: str,
    fusions: List[str], 
    df_window: pd.DataFrame = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
   Computes fusion functions on classification results for genuine and forgery signatures.

   Parameters:
   ----------
   file_path : str
       Path to the input csv file containing batch predictions from models.py.
   
   n_ref : int
       The number of reference signatures.
   
   forgery : str
       The type of forgery to consider. Should be either 'random' or 'skilled'. 
   
   fusions : List[str]
       List of fusion functions to apply (e.g., 'max', 'mean'). Defaults to ['max'].
     
   df_window : pd.Dataframe, optional
       Window of data when evaluting a stream.

   Returns:
   -------
   Tuple[pd.DataFrame, List[str]]
       - pd.DataFrame: A DataFrame containing the fusion results.
       - List[str]: The list of fusion functions used.
    """
    
    if df_window is None: # Batch fusion function
        df_f = pd.read_csv(file_path, sep = '\t')
        
        # Get the number of n_query in the file
        h = df_f[df_f['q_type']==1].groupby(['ref_users', 'ref_idxs']).count()
        n_q = h['q_type'].iloc[0]
         
        # Order data by ref_user (user ID) and ref_idxs (ref sig. ID)
        df_f.sort_values(by=['ref_users', 'ref_idxs'], ascending=[True, True], inplace=True)
       
        # Filter data to consider only the first n_ref sig. per query sig. of each user
        df = df_f.groupby(['ref_users','q_type']).head(n_ref * n_q)
        
        qtype_forg = 2 if forgery == 'random' else 3 if forgery == 'skilled' else -1
        
        df =  df[(df['q_type']==1)  | (df['q_type']==qtype_forg) ]
    
    else:
        df = df_window
    
    # Compute fusion function for each query sig. of each user in relation to all n_ref sig.
    df_proba_class0 = df.groupby(['ref_users','q_idxs', 'q_type', 'label'])['proba_class0'].agg(fusions).reset_index() # proba_class0 is relevant only for probabilistic models, otherwise all values are 0
    
    df_proba_class1 = df.groupby(['ref_users','q_idxs', 'q_type', 'label'])['proba_class1'].agg(fusions).reset_index()
    
    df_proba = df_proba_class0.merge(df_proba_class1, on=['ref_users', 'q_idxs', 'q_type', 'label'], how='inner', suffixes=('0', '1'))

    gen_and_forg = df_proba.copy()
    
    for f in fusions:
        gen_and_forg.loc[:, f'pred_{f}'] = df_proba.apply(lambda row: 0 if row[f'{f}0'] > row[f'{f}1'] else 1, axis=1) 
    

    return gen_and_forg
    
def compute_batch_metrics(
                        file_path: str,
                        n_ref: int,
                        forgery: str,
                        pos_label: int,
                        threshold_type: str,
                        fusions : List[str]) -> pd.Series:
    """
       Computes the Equal Error Rate (EER) on batch predictions file from models.py.

    Parameters:
    ----------
    file_path : str
        Path to the input csv file containing batch predictions from models.py.
    
    n_ref : int
        The number of reference signatures.
    
    forgery : str
        The type of forgery to consider. Should be either 'random' or 'skilled'.
    
    pos_label : int
        The label of the positive class (e.g., 1 for genuine signatures).
    
    threshold_type : str
        If 'user', calculates the EER for each user individually (user-specific threshold).
        If 'global', calculates the EER globally across all users (global threshold).

    fusions : List[str]
        List of fusion functions to apply (e.g., 'max', 'mean').
        
    Returns:
    -------
    pd.Series
        A pandas Series containing the EER values for each fusion function used.
        The index of the Series represents the fusion function names.
    """
    
    
    gen_and_forg = fusion_function(file_path, n_ref, forgery, fusions) 
    
    fusion_col_p = [f + str(pos_label) for f in fusions] # Can be max1, min1.. or max0, min0
    
    # Initialize EER Series for each fusion function
    m_eer = pd.Series(np.zeros(len(fusions)), 
                      index=fusion_col_p)
    
    users = gen_and_forg['ref_users'].unique()
    
    if threshold_type == 'user': # User threshold
        
        for u in users: 
            df_u = gen_and_forg[gen_and_forg['ref_users'] == u]
      
            # EER considering all query signatures of each user
            eer_thr = df_u[fusion_col_p].apply(lambda col: 
                                               eer(df_u['label'], 
                                                   col, 
                                                   pos_label, 
                                                   return_threshold=True)
                                               )
            m_eer += eer_thr.iloc[0]
            
          
        m_eer /= len(users)
     
    else:  # Global threshold
        eer_thr = gen_and_forg[fusion_col_p].apply(lambda col: 
                                                   eer(gen_and_forg['label'], 
                                                       col, 
                                                       pos_label, 
                                                       return_threshold=True))
        m_eer = eer_thr.iloc[0]
        
        
    m_eer.index = fusions
       
    return m_eer


def compute_stream_metrics(file_path: str,
                        n_ref: int,
                        forgery: str,
                        pos_label: int,
                        threshold_type: str,
                        fusions: List[str],
                        window_size: int,
                        window_step: int) -> pd.Series:
                            
    """
       Computes the Equal Error Rate (EER) on stream predictions file from models.py.

    Parameters:
    ----------
    file_path : str
        Path to the input csv file containing stream predictions from models.py.
    
    n_ref : int
        The number of reference signatures.
    
    forgery : str
        The type of forgery to consider. Should be either 'random' or 'skilled'.
    
    pos_label : int
        The label of the positive class (e.g., 1 for genuine signatures).
    
    threshold_type : str
        If 'user', calculates the EER for each user individually (user-specific threshold).
        If 'global', calculates the EER globally across all users (global threshold).
    
    window_size : int
        The number of samples used for evaluation.
    
    window_step : int
        The frequency (in number of samples) the evaluation should be performed.        

    fusions : List[str]
        List of fusion functions to apply (e.g., 'max', 'mean'). 
        
    Returns:
    -------
    pd.Series
        A pandas Series containing the EER values for each fusion function used.
        The index of the Series represents the fusion function names.
    """
 
    df_f = pd.read_csv(file_path, sep = '\t')
    
    qtype_forg = 2 if forgery == 'random' else 3 if forgery == 'skilled' else -1
    
    # Filter with only type of forgery
    df_f = df_f[ (df_f['q_type'] == 1)  |  (df_f['q_type'] == qtype_forg)]
    
    n_samples = len(df_f)
    
    fusion_col_p = [f + str(pos_label) for f in fusions] # Can be max1, min1.. or max0, min
    
    r_df = pd.DataFrame()
    current_pos = 0 
    
    # Evaluate every window_step on the last window_size samples
    while current_pos + window_size <= n_samples:
        df_window = df_f.iloc[current_pos : current_pos + window_size]
        
        current_pos+= window_step        
       
        gen_and_forg = fusion_function(None, None, None, fusions, df_window )
        
        if threshold_type == 'user': # User threshold
        
            # Initialize EER Series for each fusion function
            m_eer = pd.Series(np.zeros(len(fusions)), 
                              index=fusion_col_p)
        
            users = gen_and_forg['ref_users'].unique()
            
            for u in users:
               df_u = gen_and_forg[gen_and_forg['ref_users'] == u]
               
               # EER considering all query signatures of each user
               eer_thr = df_u[fusion_col_p].apply(lambda col: 
                                                  eer(df_u['label'], 
                                                      col, 
                                                      pos_label, 
                                                      return_threshold=True)
                                                  )
               m_eer += eer_thr.iloc[0]
              
             
            m_eer /= len(users)
    
        else:  # Global threshold
            eer_thr = gen_and_forg[fusion_col_p].apply(lambda col: 
                                                       eer(gen_and_forg['label'], 
                                                           col, 
                                                           pos_label, 
                                                           return_threshold=True))
            m_eer = eer_thr.iloc[0]

            
        m_eer.index = fusions
        m_eer.name = 'eer'
       
        new_df = pd.DataFrame({
                'n_signatures': [current_pos * window_size // window_step // n_ref] * len(fusions),
                'fusion': fusions
                })
        
        new_df.index = new_df['fusion']
        data_frames = [new_df, m_eer]
        merged_df = pd.concat(data_frames, axis=1, join='inner' )
        
        r_df = pd.concat([r_df, merged_df])
    
    return r_df

   
    
def main_batch(args):
    
    print("--- COMPUTING BATCH METRICS ---")
    print(args)
    print("------------------------")
    
    input_folder_path = args.f_input_path 
    output_folder_path = args.f_output_path
    forgeries = args.forgery # ['skilled', 'random']
    threshold_types = args.thr_type # ['user', 'global']
    pos_label = args.pos_label # 1 or 0
    n_refs = args.n_ref # [1, 2, 3, 5, 10, 12]
    fusions = args.fusions # ['max', 'min']
    
    # Create output path
    if not os.path.exists(output_folder_path):
        print(f"Creating folder: {output_folder_path}")
        os.makedirs(output_folder_path)
    
    # Get csv prediction files
    pred_files = [ f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    
    # Create an empty list to store DataFrames
    dfs = []

    for forgery in forgeries:
        for threshold_type in threshold_types:
            for n_ref in n_refs:
                n_ref = int(n_ref)
                all_eer = []
                
                print(f'Configuration: n_ref: {n_ref} thr: {threshold_type} forg: {forgery}')
                
                # Compute metrics for each prediction file
                for pf in pred_files:
                    
                    print(f"File: {pf}")

                    file_path = os.path.join(input_folder_path, pf)
                    
                    m_eer =  compute_batch_metrics(file_path, 
                                                   n_ref, 
                                                   forgery, 
                                                   pos_label, 
                                                   threshold_type, 
                                                   fusions)
                    
                    all_eer.append(m_eer)

               
                # Compute mean EER and std
                df = pd.DataFrame(all_eer)
                df = df.transpose()
                mean_result = df.mean(axis=1)
                std_result = df.std(axis=1)
                all_metrics = pd.concat([mean_result,
                                         std_result], 
                                        axis=1, keys=['eer', 'eer_std'])
                
                print('Result by fusion:')
                print(f'{all_metrics}')
                print("------------------------")
                
                # Create a DataFrame for additional information
                data = {
                    'threshold_type': [threshold_type],
                    'forgery': [forgery],
                    'n_ref': [n_ref]
                }
                df_info = pd.DataFrame(data)

                result_df = df_info.merge(all_metrics, 
                                          left_index=True, 
                                          right_index=True,
                                          how='outer').ffill().dropna()
                
                result_df['fusion'] = result_df.index

                # Append the result DataFrame to the list
                dfs.append(result_df)
    
    # Concatenate all DataFrames in the list into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Save the DataFrame to a CSV file
    final_df.to_csv(os.path.join(output_folder_path, f'{os.path.basename(input_folder_path)}.csv'), sep= "\t", index=False)

def main_stream(args):
    
    print("--- COMPUTING STREAM METRICS ---")
    print(args)
    print("------------------------")
    
    input_folder_path = args.f_input_path 
    output_folder_path = args.f_output_path
    forgeries = args.forgery # ['skilled', 'random']
    threshold_types = args.thr_type # ['user', 'global']
    pos_label = args.pos_label # 1 or 0
    window_size = args.window_size 
    window_step=args.window_step
    fusions = args.fusions
    
    # Create output path
    if not os.path.exists(output_folder_path):
        print(f"Creating folder: {output_folder_path}")
        os.makedirs(output_folder_path)

    # Get csv prediction files
    pred_files = [ f for f in os.listdir(input_folder_path) if f.endswith('.csv')]
    
    # Get config prediction file
    with open(os.path.join(input_folder_path,pred_files[0].replace(".csv", ".json") ), 'r') as json_file:
        config_file = json.load(json_file)
    json_file.close()
    
    # Get n_ref from configuration file
    n_ref = config_file["n_ref"]
    
    # Compute window in terms of dissimilarity samples
    diss_window_size = window_size * n_ref 
    diss_window_step = window_step * n_ref 
    
    # Create an empty list to store DataFrames
    dfs = []
    metric = 'eer'
    
    for forgery in forgeries:
        for threshold_type in threshold_types:
            
            print(f'Configuration: n_ref: {n_ref} thr: {threshold_type} forg: {forgery} ws: {window_size} wp: {window_step}' )
            
            
            all_eer = []
            # Compute metrics for each prediction file
            for pf in pred_files:
                
                print(f"File: {pf}")
                file_path = os.path.join(input_folder_path, pf)
                m_eer = compute_stream_metrics(file_path, 
                                                n_ref,         
                                                forgery,
                                                pos_label,
                                                threshold_type,
                                                fusions,
                                                diss_window_size, 
                                                diss_window_step, 
                                                )
                all_eer.append(m_eer)
                
            all_metrics = m_eer[m_eer.columns.difference([metric])].copy()
            mean_series = pd.concat([df[metric] for df in all_eer],
                                    axis=1).mean(axis=1)
            std_series  = pd.concat([df[metric] for df in all_eer], 
                                    axis=1).std(axis=1)

            all_metrics[metric] = mean_series
            all_metrics[f'{metric}_std'] = std_series
            
            print('Last window result by fusion:')  #TODO: show the last all_metrics.iloc[-2:]
            print(f'{all_metrics.iloc[-len(fusions):]}')
            print("------------------------")
            
            # Additional information
            
            all_metrics['threshold_type']= threshold_type
            all_metrics['forgery']= forgery
            all_metrics['n_ref']= n_ref
            all_metrics['model']= config_file["model"]
            
            dfs.append(all_metrics)
                

    # Concatenate all DataFrames in the list into a single DataFrame
    final_df = pd.concat(dfs, ignore_index=True)
    
    filename = os.path.basename(input_folder_path)
    
    # Save the DataFrame to a CSV file
    final_df.to_csv(os.path.join(output_folder_path, f'ws{window_size}_wp{window_step}_{filename}.csv'), sep= "\t", index=False)
            
    return all_metrics        
 
    
if __name__ == '__main__':
    
    def common_args(p):

        p.add_argument('--f-input-path', type=str, required=True, help='Directory path with prediction CSV files.')
        p.add_argument('--f-output-path', type=str, required=True, help='Directory path where a evalution CSV file will be created') 
        p.add_argument('--pos-label', default=1, help=' The label of the positive class (i.e., the class for which we are interested in calculating the EER). Default is 1.') 
        p.add_argument('--forgery',  nargs='+', required=True, help='List of forgery types. Should be "random" or "skilled" or both. Ex: --forgery random skilled')
        p.add_argument('--thr-type',  nargs='+', required=True, help='List of threshold types. Should be "user" or "global" or both. Ex: --thr-type user global')
        p.add_argument('--fusions',  nargs='+', default=['max'], help='List of fusion functions to apply (e.g., "max", "mean"). Defaults to ["max"].')
       
    main_parser = argparse.ArgumentParser()
    
    subparsers = main_parser.add_subparsers(dest='type')
    
    # Stream evaluation parameters
    stream = subparsers.add_parser('stream', help='Prequential metrics evaluation')
    stream.add_argument('--window-size', type=int, required=True, help='Number of the last query sig. used for computing metrics.') 
    stream.add_argument('--window-step', type=int, required=True, help='Sliding window step in terms of number of query sig. Ex: if 10, metric will be compute every new 10 query signatures.')  
    common_args(stream)
    stream.set_defaults(func=main_stream)
    
    # Batch evaluation parameters
    batch = subparsers.add_parser('batch', help='Batch metrics evaluation')
    batch.add_argument('--n-ref',  nargs='+', required=True, help='List of reference signatures. Ex: --n-ref 1 2 12.')
    common_args(batch)
    batch.set_defaults(func=main_batch)
   
    args = main_parser.parse_args()
    args.func(args)