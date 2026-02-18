from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

import os
import json
import shap  # <--- [NEW] for Explainable AI
import matplotlib.pyplot as plt # <--- [NEW] for plotting
import numpy as np
from sklearn.metrics import accuracy_score, f1_score 

def main(args):

    # 1. Load Feature Maps
    print(f"Loading feature maps...", flush=True)
    eventname_edgefeats = json.load( open( args.eventname_edgefeats_path, "r") )
    nodetype_nodefeats = json.load( open( args.nodetype_nodefeats_path, "r") )
    
    # 2. Load Datasets
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
    n_gram_val = args.N[0] if isinstance(args.N, list) else args.N
    pool_val = args.pool[0] if isinstance(args.pool, list) else args.pool

    # 4. Initialize Model
    print(f"Initializing Graphite (XGBoost) with N={n_gram_val}...", flush=True)
    graphite_ngram = Graphite_Ngram( 
        N = n_gram_val, 
        pool= pool_val,
        n_estimators=args.estimators,
        learning_rate=args.lr 
    )

    # 5. Training (With SMOTE inside)
    graphite_ngram.fit( 
        train_dataset = train_dataset,  
        nodetype_nodefeats = nodetype_nodefeats,  
        eventname_edgefeats= eventname_edgefeats 
    )

    # 6. Testing & Collecting Data for SHAP
    print("Running evaluation on test set...", flush=True)
    preds, truths = [], []
    test_embeddings = [] # Store embeddings for SHAP

    for idx, test_data in enumerate(test_dataset):
        # Generate embedding manually to store it
        emb = graphite_ngram.generate_graph_embedding(test_data)
        emb_list = emb.tolist()
        test_embeddings.append(emb_list)
        
        # Predict using the embedding
        pred = graphite_ngram.base_model.predict([emb_list]).item()
        
        truth  = [ 1 if "malware" in test_data.name.lower() else 0 ][0]
        
        if idx % 50 == 0:
            print(f"Predicted: {pred} | Truth: {truth} --- {test_data.name}", flush=True)
            
        preds.append(pred)
        truths.append(truth)

    # 7. Results
    test_acc = accuracy_score(y_true = truths, y_pred = preds)
    test_f1 = f1_score(y_true = truths, y_pred = preds)

    print("\n" + "="*50, flush=True)
    print(f"🔥 Final Results (Augmented + Ensemble):", flush=True)
    print(f"Test-Acc: {test_acc:.4f} ({test_acc*100:.2f}%)", flush=True)
    print(f"Test-F1 : {test_f1:.4f}", flush=True)
    print("="*50, flush=True)

    # --- [NEW] Explainable AI (SHAP) Logic ---
    print("\nGenerating Explainable AI (XAI) Plots...", flush=True)
    try:
        # 1. Get Feature Names
        feature_names = graphite_ngram.get_feature_names()
        
        # 2. Extract the internal XGBoost model (index 0 in VotingClassifier)
        # We explain XGBoost because it's the strongest component
        xgb_model = graphite_ngram.base_model.estimators_[0]
        
        # 3. Create Explainer
        X_test_matrix = np.array(test_embeddings)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test_matrix)

        # 4. Plot: Summary Bar Chart (Top 20 Features)
        plt.figure(figsize=(10, 8))
        plt.title("Top 20 Features Driving Malware Detection (SHAP)")
        shap.summary_plot(shap_values, X_test_matrix, feature_names=feature_names, plot_type="bar", max_display=20, show=False)
        plt.tight_layout()
        plt.savefig("shap_summary_bar.png")
        print(">> Saved 'shap_summary_bar.png'")

        # 5. Plot: Detailed Beeswarm (Top 20 Features)
        plt.figure(figsize=(10, 8))
        plt.title("Feature Impact Direction (SHAP Beeswarm)")
        shap.summary_plot(shap_values, X_test_matrix, feature_names=feature_names, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig("shap_beeswarm.png")
        print(">> Saved 'shap_beeswarm.png'")
        
        print("XAI Analysis Complete. Check the .png files.")

    except Exception as e:
        print(f"XAI Error: {e}")
        import traceback
        traceback.print_exc()

    return


if __name__ == "__main__":
    args = param_parser()
    main(args)