"""
Model learning and testing. 

@author: de Moura, K.
"""
import argparse
import json
import os 
import pickle
from tqdm import tqdm
from typing import List, Tuple, Union
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from shsv.data import get_stream_data, load_diss_data

warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn.linear_model')

MODEL_CHOICES = ['sgd', 'svm']

def batch_training(args):
    print("--- BATCH TRAINING ---")
    print(args)

    file_n = '' if args.file_number is None else args.file_number

    output_path = args.f_output_path   
    
    model_choice = args.model
    
    seed = args.seed
    
    
    if model_choice not in MODEL_CHOICES:
        raise ValueError(f"Invalid model choice. Expected one of: {MODEL_CHOICES}, but got: {model_choice}")
   
    train_data_folder = args.f_train_path
    files = os.listdir(train_data_folder)
    
    # Separate filenames into "tr" and "ts" lists
    tr_files =  [files for files in files if "_tr" in files]
    

    tr_files.sort(key=lambda x: int(x.split("__n")[1].split("_")[0]))
    print(f'Available training files: {tr_files}')
    
    # Create result folder
    if not os.path.exists(output_path):
        print(f"Creating folder: {output_path}")
        os.makedirs(output_path)
    
    for tr_f in tr_files:

        if not 'n'+str(file_n) in tr_f:
            continue
        print(f"Processing: {tr_f}")
        
       
        
        tr_x, tr_y, *_ = load_diss_data(os.path.join(train_data_folder, tr_f))
        
        if model_choice == 'sgd':
            
             model = SGDClassifier(loss='hinge', 
                                   random_state=seed,
                                   alpha=0.1,
                                   eta0=1,
                                   max_iter=2000,
                                   tol=0.001)
               
        else: #model_choice == 'svm':
            model = SVC(C=1, gamma=2**-11) 
       
        
        model.fit(tr_x, tr_y)

        # Save model
        o_filename = f"model_{model_choice}_"+tr_f.replace(".npz",".pkl")
        with open(os.path.join(output_path, o_filename), 'wb') as f:
            pickle.dump(model, f)

def batch_test(args):
    
    print("--- BATCH TEST ---")
    print(args)
    print('---------------')
    
    models_path = args.f_model_path
    
    test_set_path =  args.f_test_path 
    
    output_path =  args.f_output_path
    
    m_files_list  = os.listdir(models_path) 
    ts_files_list = os.listdir(test_set_path)
  
    m_files =  [file for file in m_files_list if ".pkl" in file]
    ts_files = [files for files in ts_files_list if "_ts" in files]
    
    
    m_files.sort(key=lambda x: int(x.split("__n")[1].split("_")[0]))
    ts_files.sort(key=lambda x: int(x.split("__n")[1].split("_")[0]))
    
    batch = 1000
    
    for m, ts in zip(m_files, ts_files[0:len(m_files)]):
        print(f'model: {m}')
        print(f'test:{ts} ')
        
        model = None 
        test_x, test_y, test_ds = None, None, None
        
        with open(os.path.join(models_path, m), 'rb') as f:
            model = pickle.load(f)
    
        test_x, test_y, *test_ds = load_diss_data(os.path.join(test_set_path, ts))
            
        o_pred = []
        o_proba_class0 = []
        o_proba_class1 = []
        o_label = []
               
        for bi in range(0,test_x.shape[0], batch):
            #print(bi)
            ts_x = test_x[bi:bi+batch]
            ts_y = test_y[bi:bi+batch]
            
            pred = model.predict(ts_x)
            o_pred.extend(pred)
            o_proba_class0.extend(np.zeros_like(pred))
            decisionf = model.decision_function(ts_x) 
            o_proba_class1.extend(decisionf)
            o_label.extend(ts_y)

        #Create result folder
        if not os.path.exists(output_path):
            print(f"Creating folder: {output_path}")
            os.makedirs(output_path)
            
        o_filename = "pred#"+m.replace(".pkl","#")+ts.replace(".npz",".csv")
        
        pd.DataFrame({'pred': o_pred, 
                      'proba_class0': o_proba_class0 , 
                      'proba_class1': o_proba_class1,
                      'label': o_label,
                      'ref_idxs': test_ds[0],
                      'q_idxs': test_ds[1],
                      'ref_users': test_ds[2],
                      'q_users': test_ds[3],
                      'q_type': test_ds[4]
                      }
                     ).to_csv(os.path.join(output_path, o_filename), sep='\t')

        print(f'Saving predictions in: {o_filename}') 
        print('---------------')
      
    
def _test_chunk( model: Union[SGDClassifier, SVC], test_x: np.ndarray) -> Tuple[List[int], List[float]]:
    """
    Tests a model on a chunk from the stream, returning the predictions and decision function scores.

    Parameters:
    -----------
        model (BaseEstimator): 
            The trained machine learning model with `predict` and `decision_function` methods.
        test_x (np.ndarray): 
            The input features for the test data, with shape (n_samples, n_features).

    Returns:
    --------
        Tuple[List[int], List[float]]: A tuple containing two lists:
            - `y_preds`: The predicted class labels for the test data.
            - `y_dfunctions`: The decision function scores for the test data.
    """
    batch = 1000
    y_preds = []
    y_dfunctions = []
    
    for bi in range(0,test_x.shape[0], batch):

        ts_x = test_x[bi:bi+batch]
        
        y_pred  = model.predict(ts_x)
   
        y_dfunction = model.decision_function(ts_x)
        
        y_preds.extend(y_pred)
        y_dfunctions.extend(y_dfunction)
        
    return y_preds, y_dfunctions

def _train_chunk(rng: np.random.RandomState, 
                model: Union[SGDClassifier, SVC], 
                model_choice: str, 
                x_values: np.ndarray, 
                y_values: np.ndarray, 
                tr_classes: np.ndarray) -> Union[SGDClassifier, SVC]:
    """
    Train a chunk from the stream using the provided model.

    Parameters:
    -----------
        rng (np.random.Generator): 
            A random number generator for shuffling data.
        model (Union[SGDClassifier, SVC]): 
            The model used for training.
        model_choice (str): 
            The type of model being used ('sgd' or 'svm').
        x_values (np.ndarray): 
            The input features for training, shape (n_samples, n_features).
        y_values (np.ndarray): 
            The target labels for training, shape (n_samples,).
        tr_classes (np.ndarray): 
            The unique classes for training.

    Returns:
    --------
        Union[SGDClassifier, SVC]: The updated model.
    """
    if len(x_values) == 1: # Treat shape
        x_values = x_values[0].reshape(1,-1)
        y_values = np.atleast_1d(y_values[0])
       
    if model_choice == 'sgd': 
        model.partial_fit(x_values, y_values, classes=tr_classes)

        indices = np.arange(len(x_values))
        partial_itr = 100 #repeat 100 times for better results. By default it does one iteration
        for i in range(partial_itr-1):
            rng.shuffle(indices)
            model.partial_fit(x_values[indices], y_values[indices], classes=tr_classes)
        
    return model

def _store_predictions( existing_df: pd.DataFrame, 
                        chunk: Union[int, slice], 
                        tr_y: np.ndarray, 
                        y_pred: List[int], 
                        decision: List[float], 
                        ds: List[np.ndarray], 
                        n_ref: int, 
                        test_filename_list: List[str]) -> pd.DataFrame:
    """
    Stores predictions, decision functions, and associated metadata into a DataFrame.

    Parameters:
    -----------
    existing_df : DataFrame
        The existing DataFrame to append the new predictions to.
    chunk : Union[int, slice]
        The index or slice of the chunk being processed.
    tr_y : ndarray
        The ground truth labels for the chunk.
    y_pred : List[int]
        The predicted labels for the chunk.
    decision : List[float]
        The decision function values for the chunk.
    ds : List[ndarray]
        A list of ndarrays containing the metadata:
        - ds[0]: ref_idxs
        - ds[1]: q_idxs
        - ds[2]: ref_users
        - ds[3]: q_users
        - ds[4]: q_type
    n_ref : int
        The number of reference signatures.
    test_filename_list : List[str]
        The list of test filenames corresponding to the chunk.

    Returns:
    --------
    DataFrame
        A DataFrame containing the predictions, decision functions, and metadata, appended to the existing DataFrame.
    """
    
    new_df = pd.DataFrame({
        "pred": y_pred,
        "proba_class0": len(y_pred)*[0],
        "proba_class1": decision,
        "label": tr_y[chunk],
        "ref_idxs":  ds[0][chunk], # 'ref_idxs':
        "q_idxs":  ds[1][chunk], # 'q_idxs': 
        "ref_users": ds[2][chunk], #'ref_users'
        "q_users":   ds[3][chunk], #'q_users'
        "q_type": ds[4][chunk], #'q_type'
        "stream_name": test_filename_list,
        "n_ref":   len(y_pred)*[n_ref]
    })
        
    if existing_df.empty:
        return new_df
  
    return pd.concat([existing_df, new_df], ignore_index=True)
    
  

def prequential(args):

    print("--- PREQUENTIAL (TEST-THEN-TRAIN) ---")
    print(args)
    print('---------------')
    

    rng = np.random.RandomState(args.seed)
    
    n_ref = args.n_ref
    f_number = args.file_number
    output_pred_path = args.f_output_path
    f_test_path = args.f_test_path
    stream_order = args.stream_order
    chunk_size = args.chunk_size # Total number of sig used for training (only G and RF)
    
    chunk_size = chunk_size + (chunk_size // 2) # Add skilled forgery for testing. It considers the stream is a sequence of G, RF, SK, G, RF, SK.. so forth. 
    diss_chunk = chunk_size * n_ref
    
    
    # Load model
    files = os.listdir(args.f_model_path)
    file_path = [f for f in files if f.endswith('.pkl') and f.count(f"_tr__n{f_number}_")][0]
    
    if len(file_path) == 0:
        raise ValueError(f"No model found with file number {f_number} in {args.f_model_path}!")
    model_path = os.path.join(args.f_model_path, file_path)
    model = None
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    model_choice = 'svm' if type(model).__name__ == 'SVC' else 'sgd' 
    print(f'Model loaded: name: {model_choice}, file: {file_path}')
    
    # Create prediction output file
    df_predictions = pd.DataFrame(columns=["pred", "proba_class0", "proba_class1", "label", "ref_idxs", "q_idxs", 
                           "ref_users", "q_users", "q_type", "stream_name", "n_ref"])
    
    if not os.path.exists(output_pred_path):
            print(f"Creating predictions output folder: {output_pred_path}")
            os.makedirs(output_pred_path)
            
    model_name = os.path.basename(model_path).replace(".pkl","")
    
    csv_name = f'pred#prequential_r{n_ref}_ch{args.chunk_size}#{model_name}.csv'
    csv_file_path = os.path.join(output_pred_path, csv_name)
    
    if os.path.exists(csv_file_path):
        os.remove(csv_file_path)
        print(f"Prediction file '{csv_file_path}' existed and has been deleted.")
        
    # Load stream
    print('Loading stream...')
    (tr_x, tr_y, *tr_ds), data_sources, stream_index = get_stream_data(rng, f_test_path, f_number, n_ref, stream_order)
    stream_data_sources = np.unique(data_sources).tolist()
    print(f'Stream data source(s): {stream_data_sources}')
    
    # Create prediction information file (json)
    print('Creating information json file..')
    config_file_data = {'eval_type':'prequential', 
                        'n_ref': n_ref, 
                        'model': model_name.replace("model_", "").split("_")[0],
                        'model_info': model_name.replace("model_", ""),
                        'model_settings': repr(model),
                        'data_sources': ','.join(stream_data_sources),
                        'arg_chunk_size': args.chunk_size,
                        'chunk_size': chunk_size,
                        'diss_chunk': diss_chunk,
                        'stream_order':stream_order
                        }
    
    json_file_path = csv_file_path.replace(".csv",".json")
    with open(json_file_path, 'w') as json_file:
        json.dump(config_file_data, json_file)
    
    tr_classes = np.unique(tr_y)
    
    print('Starting prequential evaluation..')
    
    for samples_counter in tqdm(range(0, len(stream_index), diss_chunk), desc="Processing", file=sys.stdout):
        chunk = stream_index[samples_counter:samples_counter + diss_chunk]
       
        x_values = tr_x[chunk] 
        y_values = tr_y[chunk] 
        test_filename_list = data_sources[chunk]
        
        # Test
        y_pred, y_dfunction = _test_chunk(model, x_values)
        
        # Store results
        df_predictions = _store_predictions(df_predictions, chunk, tr_y, y_pred, y_dfunction, tr_ds, n_ref, test_filename_list )
        
        # Remove skilled forgeries
        ch_without_sk = np.where(tr_ds[-1][chunk] != 3)[0]
        x_values = tr_x[chunk][ch_without_sk] 
        y_values = tr_y[chunk][ch_without_sk]
        
        # Update model
        model = _train_chunk(rng, model, model_choice, x_values, y_values, tr_classes)
    
    # Save predictions        
    df_predictions.to_csv(csv_file_path, sep='\t', index=False)


if __name__ == '__main__': 
    
    main_parser = argparse.ArgumentParser()
    
    subparsers = main_parser.add_subparsers(dest='type')
    
    # Batch training parameters
    batch_train_args = subparsers.add_parser('batch_train', help='batch training type parameters')
    batch_train_args.add_argument('--file-number',  type=int, help='File number to process')
    batch_train_args.add_argument('--f-train-path',  type=str, required=True, help='Folder path containing npz training files')
    batch_train_args.add_argument('--model',  type=str, choices=MODEL_CHOICES, default='sgd', required=True, help='sgd: Stochastic Gradient Descent, svm: Support Vector Machine')
    batch_train_args.add_argument('--f-output-path',  type=str,  required=True, help='Absolute path to a folder')
    batch_train_args.add_argument('--seed',  type=int,  default=42, help='Seed for reproducibility')
    batch_train_args.set_defaults(func=batch_training)
    
    # Batch test parameters
    batch_test_args = subparsers.add_parser('batch_test', help='batch test parameters')
    batch_test_args.add_argument('--f-test-path',  type=str, required=True, help='Folder path containing npz test files')
    batch_test_args.add_argument('--f-model-path',  type=str, required=True, help='Folder path containing .pkl model files')
    batch_test_args.add_argument('--f-output-path',  type=str, required=True, help='Folder path to write the output predictions')
    batch_test_args.set_defaults(func=batch_test)
    
    # Prequential parameters
    prequential_args = subparsers.add_parser('prequential', help='prequential evaluation')
    prequential_args.add_argument('--file-number',  type=int, default=1, required=True, help='File number to process')
    prequential_args.add_argument('--n-ref',  type=int, default=1, required=True, help='Number of reference samples')
    prequential_args.add_argument('--f-model-path',  type=str,  required=True, help='Absolute path to a folder containing .pkl models')
    prequential_args.add_argument('--f-output-path',  type=str,  required=True, help='Absolute path to a folder')
    prequential_args.add_argument('--chunk-size',  type=int,  required=True, help='The training chunk size contains the number of signatures (genuine and random forgeries) the system should wait for before updating the model. ')

    prequential_args.add_argument('--seed',  type=int,  default=42, help='Seed for reproducibility')
    prequential_args.add_argument('--f-test-path',  nargs='+', required=True, help='List of folders path containing npz batch test files')
    prequential_args.add_argument('--stream-order', choices=['sequential', 'random'], default='sequential', help="Mode of operation (sequential or random)")
    prequential_args.set_defaults(func=prequential)
    
    args = main_parser.parse_args()
    args.func(args)