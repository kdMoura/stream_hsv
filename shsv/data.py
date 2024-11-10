#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generation. 

@author: de Moura, K.
"""
import argparse
import math
import os
from enum import Enum
from typing import List, Tuple, Union, Optional

import numpy as np

class QueryType(Enum):
    GENUINE = 1
    RANDOM_FORGERY = 2
    SKILLED_FORGERY = 3
 
class DissimilarityData:
    def __init__(self):
        self.data = []      # dissimilarity data
        self.target = []    # positive (1) or negative (0)
        self.ref_idxs = []  # indices (from orignal data) used for "reference" signature
        self.q_idxs = []    # indices (from orignal data) used for "query" signature
        self.ref_users = [] # user_id (from orignal data) used for "reference" signature
        self.q_users = []   # user_id (from orignal data) used for "query" signature 
        self.q_type = []    # QueryType

    def get_data(self,return_indices=True):
        if return_indices:
            return (np.array(self.data),
                np.array(self.target),
                np.array(self.ref_idxs),
                np.array(self.q_idxs),
                np.array(self.ref_users),
                np.array(self.q_users),
                np.array(self.q_type))
        return np.array(self.data), np.array(self.target)
    
    def add_data(self, data, target, ref_idxs, q_idxs, ref_users, q_users, q_type):
        self.data.extend(data)
        self.target.extend(target)
        self.ref_idxs.extend(ref_idxs)
        self.q_idxs.extend(q_idxs)
        self.ref_users.extend(ref_users)
        self.q_users.extend(q_users)
        self.q_type.extend(q_type)
        
def _get_data_file(folder_path: str, file_number: int) -> Tuple[str, str]:
    """
    Retrieve the full path and name of a .npz test data with specific number.
    It's designed to work with a specific file naming convention.

    Parameters:
    ----------
    folder_path (str): 
        The path to the folder containing the data files.
    file_number (int): 
        The number used to identify the specific file.

    Returns:
    ----------        
    Tuple[str, str]: A tuple containing two elements:
        - The full path to the matched file (including filename and extension)
        - The base name of the matched file (without the .npz extension)

    Raises:
    ----------
    IndexError: If no file matching the criteria is found.

    """
    test_files = os.listdir(folder_path)
    f_name = [f for f in test_files 
              if f.endswith(".npz") 
                  and f.count(f"_n{file_number}") >0 
                  and f.count("_ts__") >0][0]
    return os.path.join(folder_path, f_name), f_name.replace('.npz','')

def get_stream_data(
    rng: np.random.RandomState,
    f_path_list: List[str],
    file_number: int,
    n_ref: int,
    stream_order: str
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, 
                np.ndarray, np.ndarray, np.ndarray, np.ndarray], 
          np.ndarray, 
          List[int]]:
    
    """
    Generate a stream of signature data from one or multiple files.

    This function processes multiple data files, extracts signature information,
    and organizes it into a stream. It can return the data in sequential or
    randomized order.

    Parameters:
    -----------
    rng (numpy.random.Generator): 
        Random number generator for shuffling.
    f_path_list (List[str]): 
        List of folder paths containing the data files.
    file_number (int): 
        Identifier for selecting specific files.
    n_ref (int):
        Number of reference signatures to use.
    stream_order (str): 
        Order of the returned stream ('sequential' or 'random').

    Returns:
    -----------
    Tuple containing:
        1. Tuple of seven numpy.ndarray:
           - x: Dissimilarity feature data
           - y: Target labels
           - ref_idxs: Reference signature indices
           - q_idxs: Query signature indices
           - ref_users: Reference user IDs
           - q_users: Query user IDs
           - q_type: Query types (genuine, random forgery, skilled forgery)
        2. List[str]: Data sources for each entry
        3. List[int]: Stream index order
    """
                 
    data_list,      target_list  = [], []
    ref_idxs_list,  q_idxs_list  = [], []
    ref_users_list, q_users_list = [], []
    q_type_list = []
    data_source_list = []
    
    claimed_sig_list = []
    
    for f in f_path_list:
        test_file_path, test_file_name  = _get_data_file(f, file_number)
        x, y, ref_idxs, q_idxs, ref_users, q_users, q_type = generate_stream_from_batch(test_file_path, None, n_ref)
        data_list.append(x)     
        target_list.append(y)
        ref_idxs_list.append(ref_idxs)
        q_idxs_list.append(q_idxs)
        ref_users_list.append(ref_users)
        q_users_list.append(q_users)
        q_type_list.append(q_type)
        data_source_list.extend([test_file_name]*len(y))
        
        sig_queries = np.unique(q_idxs[(q_type==1)])
        result = np.char.add(sig_queries.astype(str), f"#{test_file_name}")
        claimed_sig_list.extend(result)
        
    
    x, y, ref_idxs, q_idxs, ref_users, q_users, q_type = (
            np.concatenate(data_list),   
            np.concatenate(target_list),
            np.concatenate(ref_idxs_list),
            np.concatenate(q_idxs_list),
            np.concatenate(ref_users_list),
            np.concatenate(q_users_list),
            np.concatenate(q_type_list),
            )
    
    data_sources = np.array(data_source_list)
    if stream_order == 'sequential':
        stream_index = list(range(0,len(y)))
        return ((x, y, ref_idxs, q_idxs, ref_users, q_users, q_type ), 
                data_sources, 
                stream_index)
                
    # If ordering is random
   
    rng.shuffle(claimed_sig_list)
    stream_index = []
    for c_sig in claimed_sig_list: # For each G claimed sig, find the 3-tuple (G, RF, SK)
        q, ds = c_sig.split('#')
    
        first_idx = min(np.where((q_idxs==int(q)) & (data_sources == ds) & (q_type ==1))[0])
        stream_index.extend(list(range(first_idx, first_idx + (n_ref*3))))
    
    return ((x, y, ref_idxs, q_idxs, ref_users, q_users, q_type ), 
            data_sources, 
            stream_index)
    

def generate_stream_from_batch(
    filename: str,
    stream_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    n_ref: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
   Generate a stream of handwriting signatures from a batch test data. 
   
   Parameters:
   -----------
   filename : str
       The path to a `.npz` file containing the dataset if `stream_data` is not provided.
       
   stream_data : tuple or None
       If provided, this should be a tuple containing:
       - data (np.ndarray): The features array.
       - target (np.ndarray): The target labels corresponding to the data.
       - ref_idxs (np.ndarray): Indices (from orignal data) used for "reference" signature
       - q_idxs (np.ndarray): Indices (from orignal data) used for "query" signature
       - ref_users (np.ndarray): User identifiers ("user_id" from orignal data) used for "reference" signatures.
       - q_users (np.ndarray): User identifiers ("user_id" from orignal data) used for "query" signatures.
       - q_type (np.ndarray): The type of query (e.g., genuine, random, skilled).
   
   n_ref : int
       The number of reference signatures to be used for each user.
   
   Returns:
   --------
   tuple:
       - data (np.ndarray): The subset of data points that match the generated stream.
       - target (np.ndarray): The corresponding target labels for the selected signatures.
       - ref_idxs (np.ndarray): The reference indices of the selected signatures.
       - q_idxs (np.ndarray): The query indices of the selected signatures.
       - ref_users (np.ndarray): The user identifiers corresponding to the reference signatures.
       - q_users (np.ndarray): The user identifiers corresponding to the query signatures.
       - q_type (np.ndarray): The type of queries (e.g., genuine, random, skilled) for the selected signatures.
   
   Description:
   ------------
   The function generates a stream of data points by iterating over each user and selecting reference and query indices based on the query type (genuine, random, or skilled). The resulting stream contains the data points in the order required for evaluation.
   """
    if stream_data is None:
        with np.load(filename, allow_pickle=True) as ds:
            data, target = ds['data'], ds['target']
            ref_idxs     = ds['ref_idxs']
            q_idxs       = ds['q_idxs']
            ref_users    = ds['ref_users']
            q_users      = ds['q_users']
            q_type       = ds['q_type']
        filename = os.path.basename(filename)
    else:
        data, target = stream_data[0], stream_data[1]
        ref_idxs, q_idxs = stream_data[2], stream_data[3]
        ref_users, q_users, q_type = stream_data[4], stream_data[5], stream_data[6]
        
    #_, _, _, _, _, sn_ref, sn_q, *_ = filename.split("_")
    # data_n_ref = int(sn_ref.replace('r',''))
    # data_n_q = int(sn_q.replace('q',''))
    # Find n_query from data without relying on filename:
    mask = np.where( (ref_users == ref_users[0]) & (q_type==1))[0]
    data_n_q = len(np.unique(q_idxs[mask]))
    
    ordered_indexes = []
    
    sel_v_ref_idxs_by_user = {} 
    
    def get_diss_idxs(ref_idxs, v_ri, 
                      q_type, qt,
                      ref_users, ru,
                      q_idxs, qi):
        
        condition = (np.isin(ref_idxs, v_ri) 
                           & (q_type == qt) 
                           & (ref_users == ru)
                           & (q_idxs == qi)
                           )
  
        return np.where(condition)[0]
    

    for q in range(data_n_q):
        
        for u in np.unique(ref_users):
           
            if u not in sel_v_ref_idxs_by_user: # First time user appears in stream
                # In batch mode the first n_ref is used for evaluation, should do the same here
                v_unique_ref_idxs, indices = np.unique(ref_idxs[ref_users == u], return_index=True)
                sorted_indices = np.argsort(indices)
            
                sel_v_ref_idxs_by_user[u] = v_unique_ref_idxs[sorted_indices][0:n_ref]
                
            v_ref_idxs = sel_v_ref_idxs_by_user[u]
               
            gen_q_idxs = q_idxs[ (ref_users == u ) & (q_type == 1)][0:data_n_q][q]
            rf_q_idxs  = q_idxs[ (ref_users == u ) & (q_type == 2)][0:data_n_q][q]
            sk_q_idxs  = q_idxs[ (ref_users == u ) & (q_type == 3)][0:data_n_q][q]
            
            sig_gen_idxs = get_diss_idxs(ref_idxs, v_ref_idxs, 
                              q_type, 1,
                              ref_users, u,
                              q_idxs, gen_q_idxs)
           
            sig_rf_idxs = get_diss_idxs(ref_idxs, v_ref_idxs, 
                              q_type, 2,
                              ref_users, u,
                              q_idxs, rf_q_idxs)
            
            sig_sk_idxs = get_diss_idxs(ref_idxs, v_ref_idxs, 
                              q_type, 3,
                              ref_users, u,
                              q_idxs, sk_q_idxs)
            
            ordered_indexes.extend([*sig_gen_idxs, *sig_rf_idxs, *sig_sk_idxs])
        
    return (data[ordered_indexes], 
            target[ordered_indexes], 
            ref_idxs[ordered_indexes], 
            q_idxs[ordered_indexes], 
            ref_users[ordered_indexes], 
            q_users[ordered_indexes], 
            q_type[ordered_indexes])


def load_extracted_features(data_path:str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-extracted signature features from a numpy file.

    Parameters:
    -----------
    data_path (str): Path to the .npz file containing the extracted features.

    Returns:
    -----------
    tuple: Contains:
        - features (numpy.ndarray): Extracted features of shape (samples, features).
        - y (numpy.ndarray): Labels with user IDs.
        - yforg (numpy.ndarray): Forgery flag (if 1 it is a forgery).
    """
    
    data_path = os.path.join(data_path)
    
    with np.load(data_path, allow_pickle=True) as data:
        features, y, yforg = data['features'], data['y'], data['yforg']
        #user_mapping, filenames = data['user_mapping'], data['filenames']
    # return features, y, yforg, user_mapping, filenames
    return features, y, yforg

def load_diss_data(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dissimilarity data and information from a numpy file.

    Parameters:
    -----------
    data_path (str): Path to the .npz file containing the data.

    Returns:
    -----------
    tuple: Contains:
        - data (np.ndarray): The subset of data points that match the generated stream.
        - target (np.ndarray): The corresponding target labels for the selected signatures.
        - ref_idxs (np.ndarray): The reference indices of the selected signatures.
        - q_idxs (np.ndarray): The query indices of the selected signatures.
        - ref_users (np.ndarray): The user identifiers corresponding to the reference signatures.
        - q_users (np.ndarray): The user identifiers corresponding to the query signatures.
        - q_type (np.ndarray): The type of queries (e.g., genuine, random, skilled) for the selected signatures.
    """
    
    with np.load(data_path, allow_pickle=True) as ds:
        data = ds['data']
        target = ds['target']
        
        ref_idxs = ds['ref_idxs'] if 'ref_idxs' in ds else np.array([])
        q_idxs = ds['q_idxs'] if 'q_idxs' in ds else np.array([])
        ref_users = ds['ref_users'] if 'ref_users' in ds else np.array([])
        q_users = ds['q_users'] if 'q_users' in ds else np.array([])
        q_type = ds['q_type'] if 'q_type' in ds else np.array([])
        
        return data, target, ref_idxs, q_idxs, ref_users, q_users, q_type



def generate_diss_training_data(input_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  rng: np.random.RandomState,
                                  n_data: int = 3,
                                  n_gen: int = 2,
                                  ir: int = 1,
                                  return_indices: bool = False)-> Tuple[Union[Tuple[np.ndarray, np.ndarray],                                                            Tuple[np.ndarray, np.ndarray,                                                                            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        
    """
    Generate dissimilarity training data using features extracted from a dataset.

    This function generates dissimilarity data for genuine signatures using pairwise absolute distances.
    Positive samples are obtained by calculating the pairwise absolute distance between randomly selected n_gen signatures from the same user, for each user.
    Negative samples are obtained by calculating the pairwise absolute distance between n_gen-1 genuine signatures from a user and n_gen/2 genuine signatures from other users in the dataset.

    Parameters
    ----------
    input_data : Tuple[np.ndarray, np.ndarray, np.ndarray] 
        The extracted features dataset (features, y, yforg).
    rng : np.random.RandomState
        The random number generator (for reproducibility).
    n_data : int, optional
        Number of datasets to be generated, by default 3.
    n_gen : int, optional
        Number of genuine samples per user, by default 2. 
        (2 is the minimum required value)
    ir : int, optional
        The imbalance ratio. For each positive sample, generates ir negative samples, by default 1.
    return_indices : bool, optional
        If enabled, the indices of selected signatures are also returned,
        by default False.

    Yields
    ------
    Tuple (dissimilarity_data, target)
    
    if return_indices is True
    Tuple (dissimilarity_data, 
           target, 
           ref_idxs, #indices (from orignal data) used for "reference" signature
           q_idxs,  #indices (from orignal data) used for "query" signature
           ref_users, #user_id (from orignal data) used for "reference" signature
           q_users  #user_id (from orignal data) used for "query" signature 
           ) 

    Note
    ----
    The input_data tuple should be in the format (features, y, yforg), where:
    - features: An array containing signature features.
    - y: An array containing corresponding labels.
    - yforg: An array containing if a signature is a forgery (1) or genuine (0).

    """

    features, y, yforg = input_data
    
    unique_users = np.unique(y)
    
    n_pos = math.comb(n_gen,2)
    n_neg = n_pos * ir
    
    for repeat in range(1, n_data+1):

        diss = DissimilarityData()
        
        rng.shuffle(unique_users) 

        for user_id in unique_users:
            
            ##Within = positive label
            # Get n_gen genuine signatures for user_id     
            user_indices = np.where((y == user_id) & (yforg == 0))[0]
            gen_idxs = rng.choice(user_indices, size=n_gen, replace=False)
            f_gen = features[gen_idxs]
            
            # Compute dissimilarities
            within_diff = np.abs(f_gen[:, None] - f_gen)
            
            # Get indices of the upper triangle
            sig_idxs = np.triu_indices(n_gen, k=1) 
            
            # Get real idxs used from gen_idxs
            real_sig_idxs = (gen_idxs[sig_idxs[0]], gen_idxs[sig_idxs[1]])
            
            # Update variables with user_id result
            n_sig = real_sig_idxs[0].shape[0]
            diss.add_data(within_diff[sig_idxs], 
                          [1] * n_sig, 
                          real_sig_idxs[0], 
                          real_sig_idxs[1], 
                          [user_id] * n_sig, 
                          [user_id] * n_sig, 
                          [QueryType.GENUINE.value] * n_sig)

             
            ##Between = negative label
            # Get n_rf genuine signatures of other users
            
            diff_user_indices = np.where((y != user_id) & (yforg == 0))[0]     

            n_rf = n_gen // 2
            rf_idxs = rng.choice(diff_user_indices, size=n_rf, replace=False)    
            
            f_rf = features[rf_idxs]
            
            between_diff, mesh_gen_idxs, mes_rf_idxs = _compute_dissimilarity(f_gen, 
                                                                     f_rf, 
                                                                     gen_idxs, 
                                                                     rf_idxs)
            # Update variables with user_id result
            diss.add_data(between_diff[0:n_neg], 
                          [0] *  n_neg, 
                          mesh_gen_idxs.ravel()[0:n_neg], 
                          mes_rf_idxs.ravel()[0:n_neg], 
                          [user_id]* n_neg, 
                          y[mes_rf_idxs.ravel()[0:n_neg]], 
                          [QueryType.RANDOM_FORGERY.value] * n_neg)
            

        yield diss.get_data(return_indices)


def generate_diss_test_data(input_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  rng: np.random.RandomState,
                                  n_data: int = 3,
                                  n_ref: int = 1,
                                  n_query: int = 1,
                                  include_skilled_forgery: bool = False,
                                  return_indices: bool = False)-> Tuple[Union[Tuple[np.ndarray, np.ndarray],
                                                                              Tuple[np.ndarray, np.ndarray,                                                                               np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        
    """
    Generate test dissimilarity data from extracted dataset features.
    
    This function creates dissimilarity data tailored for testing. It calculates dissimilarity samples between `n_ref` and `n_query` signatures of three types: genuine, random forgery, and skilled forgery (if `include_skilled_forgery` is enabled). 
    
    - **Positive Samples**: Generated by computing the pairwise absolute distances between `n_ref` and `n_query` genuine signatures from the same user.
    - **Negative Samples (Random Forgeries)**: Derived from the pairwise absolute distances between `n_ref` genuine signatures of one user and `n_query` genuine signatures of different users.
    - **Negative Samples (Skilled Forgeries)**: If `include_skilled_forgery` is enabled, these are calculated by comparing `n_ref` genuine signatures with `n_query` skilled forgery signatures from the same user.
    
    If `return_indices` is `True`, the function also returns the indices used in these calculations.

    Parameters
    ----------
    input_data : Tuple[np.ndarray, np.ndarray, np.ndarray] 
        The extracted features dataset (features, y, yforg).
    rng : np.random.RandomState
        The random number generator (for reproducibility).
    n_data : int, optional
        Number of datasets to be generated, by default 3.
    n_ref: int, optional
        Number of reference signatures (genuine), by default 1.
    n_query : int, optional
        Number of genuine, random forgery and skilled forgery singatures, by default 1.
 
    include_skilled_forgery : bool, optional
        If enabled, negative samples will also include skilled forgery singatures, 
        by default False.        
    return_indices : bool, optional
        If enabled, the indices of selected signatures are also returned,
        by default False.

    Yields
    ------
    Tuple (dissimilarity_data, target)
    
    if return_indices is True
    Tuple (dissimilarity_data, 
           target, 
           ref_idxs, #indices (from orignal data) used for "reference" signature
           q_idxs,  #indices (from orignal data) used for "query" signature
           ref_users, #user_id (from orignal data) used for "reference" signature
           q_users  #user_id (from orignal data) used for "query" signature 
           q_type # (1) if genuine, (2) if random forgery, or (3) if skilled forgery
           ) 

    Note
    ----
    The input_data tuple should be in the format (features, y, yforg), where:
    - features: An array containing signature features.
    - y: An array containing corresponding labels.
    - yforg: An array containing if a signature is a forgery (1) or genuine (0).

    """
    
    features, y, yforg = input_data
    
    unique_users = np.unique(y)
    
    for repeat in range(1, n_data+1):
        diss = DissimilarityData()
        
        for user_id in unique_users:
            
            ##Within = positive label
            # get (n_ref+ n_query) genuine signatures for user_id     
            user_indices = np.where((y == user_id) & (yforg == 0))[0]
            g_idxs = rng.choice(user_indices, size= (n_ref + n_query), replace=False)
            r_idxs, gen_idxs = g_idxs[0:n_ref], g_idxs[n_ref:]
            
            f_ref = features[r_idxs]
            f_gen = features[gen_idxs]
            diff, mesh_ref_idxs, mesh_q_idxs = _compute_dissimilarity(f_ref, 
                                                                     f_gen, 
                                                                     r_idxs, 
                                                                     gen_idxs)
            n_pos = (n_ref * n_query)

            diss.add_data(diff, 
                          [1] *  n_pos, 
                          mesh_ref_idxs.ravel(), 
                          mesh_q_idxs.ravel(), 
                          [user_id]* n_pos, 
                          y[mesh_q_idxs.ravel()], 
                          [QueryType.GENUINE.value] * n_pos)
            
            ##Between = negative label RANDOM FORGERY
            diff_user_indices = np.where((y != user_id) & (yforg == 0))[0]          
            rf_idxs = rng.choice(diff_user_indices, size=n_query, replace=False)
            f_rf = features[rf_idxs]
            diff, mesh_ref_idxs, mesh_q_idxs = _compute_dissimilarity(f_ref, 
                                                                     f_rf, 
                                                                     r_idxs, 
                                                                     rf_idxs)
            n_neg = (n_ref * n_query)
            diss.add_data(diff, 
                          [0] *  n_neg, 
                          mesh_ref_idxs.ravel(), 
                          mesh_q_idxs.ravel(), 
                          [user_id]* n_neg, 
                          y[mesh_q_idxs.ravel()], 
                          [QueryType.RANDOM_FORGERY.value] * n_neg)
            
            if include_skilled_forgery: 
                ##Between = negative label SKILLED FORGERY
                diff_user_indices = np.where((y == user_id) & (yforg == 1))[0]     
                
                if len(diff_user_indices) == 0:
                    raise ValueError(f"Input data does not contain skilled forgery samples for user ID: {user_id}.")
                
                sf_idxs = rng.choice(diff_user_indices, size=n_query, replace=False)
                f_sf = features[sf_idxs]
                diff, mesh_ref_idxs, mesh_q_idxs = _compute_dissimilarity(f_ref, 
                                                                         f_sf, 
                                                                         r_idxs, 
                                                                         sf_idxs)
                n_neg = (n_ref * n_query)
                diss.add_data(diff, 
                              [0] *  n_neg, 
                              mesh_ref_idxs.ravel(), 
                              mesh_q_idxs.ravel(), 
                              [user_id]* n_neg, 
                              y[mesh_q_idxs.ravel()],
                              [QueryType.SKILLED_FORGERY.value] * n_neg) 
             
        
        yield diss.get_data(return_indices)

def filter_users(data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 select_users: Optional[List[int]] = None,
                 remove_users: Optional[List[int]] = None,
                 user_idx: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter data according to users (the second array in data).
    Data has the following np.ndarray variables: 
    data = (features, y, yforg).
    user_idx = y index.

    Parameters
    ----------
    data : Tuple (features, y, yforg)
        The extracted features dataset.
    select_users : list, optional
        The list of users to include in a half-open interval [0,300), by default None.
    remove_users : list, optional
        The list of users to exclude in a half-open interval [5000,7000), by default None.
    user_idx : int, optional
        The index in data that refers to the users (usually index=1),
        by default 1.

    Returns
    ------
    Tuple (features, y, yforg)
        The dataset containing the filtered data.
    """
    if select_users is not None and remove_users is not None:
        # Check for overlapping values. Obs: [0,10] and [10,26] do not overlap as
        # it is considered as an half-open interval [0,10) and [10, 26)
        if len(set(range(*select_users)).intersection(set(range(*remove_users)))) > 0:
            raise ValueError("select_users and remove_users have overlapping values.") 
        
    filtered_data = data
    if select_users is not None: 
        if select_users[0] > select_users[1]:
            raise ValueError("select_users: interval should start with samaller number")
        
        select = np.isin(data[user_idx], range(*select_users))
        filtered_data = tuple(d[select] for d in data)
        
    if remove_users is not None:
        if remove_users[0] > remove_users[1]:
            raise ValueError("remove_users: interval should start with samlaler number")
        
        remove = ~np.isin(filtered_data[user_idx], range(*remove_users))
        filtered_data = tuple(d[remove] for d in filtered_data)
        
    return filtered_data

def save_diss_data(datasets: List[Tuple[np.ndarray,...]],
                   output_folder: str, 
                   filename: str,
                   return_indices: bool = True,
                   )-> None:
    """
    Save dissimilarity training data to NPZ files.

    This function takes a list of datasets and saves each dataset as an NPZ file in the specified output folder. 
    The filename is constructed based on various parameters to uniquely identify each dataset.

    Parameters:
    -----------
    datasets : List of tuples 
        A list of tuples containing data, target, ...
    output_folder : str
        The path to the folder where the NPZ files will be saved.
    filename : str, optional
        Base filename for the saved NPZ files. The index of the dataset in the input list is used as file number.
    return_indices : bool, optional
        If True, save additional index and user information. Default is True.

    Returns:
    --------
    None

    Side Effects:
    -------------
    - Creates NPZ files in the specified output folder.
    - Prints information about each created file.
    """
    for i,s in enumerate(datasets):
        
        file_name = filename.replace('###', str(i)) ## Add file number
        
        path = os.path.join(output_folder, file_name)
        if return_indices:
            np.savez(path, 
                 data=s[0],
                 target=s[1],
                 ref_idxs=s[2],
                 q_idxs=s[3],
                 ref_users=s[4],
                 q_users=s[5],
                 q_type=s[6])

        else:
            data, target = s
            np.savez(path, 
                     data=data,
                     target=target,
                     )
        
        print(f'Created: {file_name} with {len(s[1])} samples')

def _compute_dissimilarity(    
        features1: np.ndarray, 
        features2: np.ndarray, 
        idxs1: np.ndarray, 
        idxs2: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the absolute dissimilarity between two sets of feature vectors.

    Parameters:
    -----------
        features1 (np.ndarray): A 2D array where each row represents a feature vector from the first set.
        features2 (np.ndarray): A 2D array where each row represents a feature vector from the second set.
        idxs1 (np.ndarray): Indexes associated with the feature vectors in `features1`.
        idxs2 (np.ndarray): Indexes associated with the feature vectors in `features2`.

    Returns:
    --------
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - diff (np.ndarray): The absolute differences between feature vectors.
            - mesh_ref_idxs (np.ndarray): The meshgrid of indexes from `idxs1`.
            - mesh_q_idxs (np.ndarray): The meshgrid of indexes from `idxs2`.
    """
    
    # compute absolute distance between feature vectors
    diff = np.abs(features1[:, None] - features2)
    
    diff = diff.reshape(-1, diff.shape[-1])
    
    mesh_ref_idxs, mesh_q_idxs = np.meshgrid(idxs1, idxs2, indexing='ij')
    
    return diff, mesh_ref_idxs, mesh_q_idxs

 
def create_diss_training_data(args):
  
    incl_users = args.include_users #range(*args.include_users) if args.include_users else None
    excl_users = args.exclude_users #range(*args.exclude_users) if args.exclude_users else None
    iu_txt = f"_iu{incl_users[0]}-{incl_users[1]}" if incl_users else ''
    eu_txt = f"_eu{excl_users[0]}-{excl_users[1]}" if excl_users else ''
     
    f = os.path.basename(args.input_data).replace('.npz','')

    o_folder_path = os.path.join(args.f_output_path)
    if not os.path.exists(o_folder_path):
        os.makedirs(o_folder_path)
    
    input_data = load_extracted_features(args.input_data)

    if incl_users and excl_users:
        filtered_data = filter_users(input_data, select_users=incl_users)
        filtered_data = filter_users(filtered_data, remove_users=excl_users)
    elif incl_users:
        filtered_data = filter_users(input_data, select_users=incl_users)
    elif excl_users:
        filtered_data = filter_users(input_data, remove_users=excl_users)
    else:
        filtered_data = input_data
    
    datasets = generate_diss_training_data(input_data=filtered_data,
                              rng=np.random.default_rng(42), 
                              n_data=args.n_data, 
                              n_gen=args.n_gen, 
                              ir=args.ir, 
                              return_indices=True )
    
    file_name = f'{f}_tr__n###_g{args.n_gen}_ir{args.ir}{iu_txt}{eu_txt}.npz'
    
    save_diss_data(datasets,
                   o_folder_path, 
                   file_name,
                   return_indices=True)
    
        
def create_diss_test_data(args):
  
    incl_users = args.include_users #range(*args.include_users) if args.include_users else None
    excl_users = args.exclude_users #range(*args.exclude_users) if args.exclude_users else None
    iu_txt = f"_iu{incl_users[0]}-{incl_users[1]}" if incl_users else ''
    eu_txt = f"_eu{excl_users[0]}-{excl_users[1]}" if excl_users else ''
     
    f = os.path.basename(args.input_data).replace('.npz','')
    
    o_folder_path = os.path.join(args.f_output_path)
    if not os.path.exists(o_folder_path):
        os.makedirs(o_folder_path)
    
    input_data = load_extracted_features(args.input_data)

    if incl_users and excl_users:
        filtered_data = filter_users(input_data, select_users=incl_users)
        filtered_data = filter_users(filtered_data, remove_users=excl_users)
    elif incl_users:
        filtered_data = filter_users(input_data, select_users=incl_users)
    elif excl_users:
        filtered_data = filter_users(input_data, remove_users=excl_users)
    else:
        filtered_data = input_data
    
    
    datasets = generate_diss_test_data(input_data=filtered_data,
                              rng=np.random.default_rng(42), 
                              n_data=args.n_data, 
                              n_ref=args.n_ref, 
                              n_query=args.n_query, 
                              include_skilled_forgery = args.sk_query, 
                              return_indices=True)
    
    sk = int(args.sk_query)
    file_name = f'{f}_ts__n###_r{args.n_ref}_q{args.n_query}_sk{sk}_{iu_txt}{eu_txt}.npz'
    save_diss_data(datasets,
                   o_folder_path, 
                   file_name,
                   return_indices=True)


def _check_min_value(min_value):
    def validate(value):
        if int(value) < min_value:
            raise argparse.ArgumentTypeError(f'Value must be greater than or equal to {min_value}')
        return int(value)
    return validate


if __name__ == '__main__':

    main_parser = argparse.ArgumentParser()
    
    def common_args(p):
        p.add_argument('--n-data',  type=_check_min_value(1), default=1, required=True, help='Number of datasets')
        p.add_argument('--input-data', type=str, required=True, help='Path to the NPZ file containing the extracted features dataset.')
        p.add_argument('--f-output-path', type=str, required=True, help='Path to where a folder will be created with data files') 
        p.add_argument('--include-users',  type=_check_min_value(0), nargs=2, help='If only a specific range of users should be considered to create the data. Example: --include-users 0 300. This is equal to [0,300)')
        p.add_argument('--exclude-users',  type=_check_min_value(0), nargs=2, help='If a specific range of users should be disconsidered to create the data. Example: --exclude-users 5000 7000. This is equal to [5001,7000)')    
        
    subparsers = main_parser.add_subparsers(dest='data_type')
    
    # training data parameters
    training_data = subparsers.add_parser('training_generation', help='Training data type parameters')
    training_data.add_argument('--n-gen', type=_check_min_value(2), default=2, help='Number of genuine signatures (min=2)')
    training_data.add_argument('--ir', type=_check_min_value(1), default=1, help='Imbalance ratio. A higher value means more negative instances for each positive class. For example, ir=1 indicates 1 negative instance for each positive class, ir=2 indicates 2 negatives for each positive, and so on.')
    common_args(training_data)
    training_data.set_defaults(func=create_diss_training_data)
    
    # test data parameters
    test_data = subparsers.add_parser('test_generation', help='Test data type parameters')
    test_data.add_argument('--n-ref', type=_check_min_value(1), default=1, help='Number of reference signatures')
    test_data.add_argument('--no-sk-query', action='store_false', dest='sk_query', help='Disable creation of query signatures for skilled forgery')
    test_data.add_argument('--n-query', type=_check_min_value(1), default=1, help='Number of query signatures. For example: if 10, then 10 genuine, 10 random forgery and 10 skilled forgeries, will be created.')
    common_args(test_data)
    test_data.set_defaults(func=create_diss_test_data)
    
    args = main_parser.parse_args()
    args.func(args)
