from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

import os
import json
from sklearn.metrics import accuracy_score, f1_score 
import numpy as np

def main(args):

    # 1. Load Feature Maps
    print(f"Loading feature maps...", flush=True)
    eventname_edgefeats = json.load( open( args.eventname_edgefeats_path, "r") )
    nodetype_nodefeats = json.load( open( args.nodetype_nodefeats_path, "r") )
    
    # 2. Load Datasets (Using default path from parser)
    print(f"Loading datasets from: {args.dataset_path}", flush=True)
    
    train_dataset = load_dataset( 
        benign_data_path = os.path.join( args.dataset_path, "train/benign"),  
        malware_data_path = os.path.join( args.dataset_path, "train/malware"), 
        dim_node = len(nodetype_nodefeats),  
        dim_edge= len(eventname_edgefeats) + 1 
    ) 
    
    test_dataset = load_dataset( 
        benign_data_path =  os.path.join( args.dataset_path,"test/benign"), 
        malware_data_path =  os.path.join( args.dataset_path,"test/malware"), 
        dim_node = len(nodetype_nodefeats), 
        dim_edge= len(eventname_edgefeats) + 1 
    )

    # 3. Handle Arguments
    # Fix for nargs=1 returning list
    n_gram_val = args.N[0] if isinstance(args.N, list) else args.N
    pool_val = args.pool[0] if isinstance(args.pool, list) else args.pool

    # 4. Initialize Improved Model (XGBoost)
    print(f"Initializing Graphite (XGBoost) with N={n_gram_val}...", flush=True)
    graphite_ngram = Graphite_Ngram( 
        N = n_gram_val, 
        pool= pool_val,
        n_estimators=args.estimators,
        learning_rate=args.lr 
    )

    # 5. Training
    graphite_ngram.fit( 
        train_dataset = train_dataset,  
        nodetype_nodefeats = nodetype_nodefeats,  
        eventname_edgefeats= eventname_edgefeats 
    )

    # 6. Testing
    print("Running evaluation on test set...", flush=True)
    preds, truths = [], []
    for idx, test_data in enumerate(test_dataset):
        pred = graphite_ngram.predict( test_data )
        truth  = [ 1 if "malware" in test_data.name.lower() else 0 ][0]
        
        # Reduced print frequency to keep terminal clean
        if idx % 50 == 0:
            print(f"Predicted: {pred} | Truth: {truth} --- {test_data.name}", flush=True)
            
        preds.append(pred)
        truths.append(truth)

    # 7. Results
    test_acc = accuracy_score(y_true = truths, y_pred = preds)
    test_f1 = f1_score(y_true = truths, y_pred = preds)

    print("\n" + "="*50, flush=True)
    print(f"🔥 Final Results (XGBoost Improved):", flush=True)
    print(f"Test-Acc: {test_acc:.4f} ({test_acc*100:.2f}%)", flush=True)
    print(f"Test-F1 : {test_f1:.4f}", flush=True)
    print("="*50, flush=True)

    return


if __name__ == "__main__":
    args = param_parser()
    main(args)