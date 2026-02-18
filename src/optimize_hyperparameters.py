import optuna
import os
import json
import numpy as np
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# তোমার এক্সিস্টিং মডিউলগুলো ইমপোর্ট করছি
from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

warnings.filterwarnings("ignore")

def main(args):
    print("🚀 Starting Hyperparameter Optimization with Optuna...", flush=True)

    # ---------------------------------------------------------
    # ১. ডেটা এবং ফিচার লোড করা (আগের মতোই)
    # ---------------------------------------------------------
    eventname_edgefeats = json.load(open(args.eventname_edgefeats_path, "r"))
    nodetype_nodefeats = json.load(open(args.nodetype_nodefeats_path, "r"))

    print(f"Loading datasets...", flush=True)
    
    # শুধু ট্রেইনিং ডেটা লোড করলেই হবে অপটিমাইজেশনের জন্য
    train_dataset = load_dataset(
        benign_data_path=os.path.join(args.dataset_path, "train/benign"),
        malware_data_path=os.path.join(args.dataset_path, "train/malware"),
        dim_node=len(nodetype_nodefeats),
        dim_edge=len(eventname_edgefeats) + 1
    )

    # ---------------------------------------------------------
    # ২. গ্রাফ প্রসেসিং এবং এম্বেডিং জেনারেশন (একবার করলেই হবে)
    # ---------------------------------------------------------
    # আমরা Graphite ক্লাসটা ব্যবহার করছি শুধু ফিচার ভেক্টর বানানোর জন্য
    # মডেল ট্রেইনিং আমরা পরে Optuna এর ভেতরে করব
    n_gram_val = args.N[0] if isinstance(args.N, list) else args.N
    
    print("Generating Embeddings (This happens once)...", flush=True)
    temp_graphite = Graphite_Ngram(N=n_gram_val)
    
    # ভেক্টরাইজার ফিট করা
    temp_graphite.nodetype_nodefeats = nodetype_nodefeats
    temp_graphite.eventname_edgefeats = eventname_edgefeats
    temp_graphite.fit_count_vectorizer(train_dataset)
    
    # গ্রাফ থেকে ভেক্টর (X) তৈরি করা
    X_raw = []
    y_raw = []
    
    cnt = 0
    for data in train_dataset:
        emb = temp_graphite.generate_graph_embedding(data).tolist()
        X_raw.append(emb)
        label = 1 if "malware" in data.name.lower() else 0
        y_raw.append(label)
        cnt += 1
        if cnt % 100 == 0:
            print(f"Processed {cnt} graphs...", flush=True)

    X_raw = np.array(X_raw)
    y_raw = np.array(y_raw)

    # ---------------------------------------------------------
    # ৩. SMOTE দিয়ে ডেটা ব্যালেন্স করা
    # ---------------------------------------------------------
    print("Applying SMOTE before optimization...", flush=True)
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X_raw, y_raw)
    print(f"Optimization Dataset Shape: {X.shape}", flush=True)

    # ---------------------------------------------------------
    # ৪. Optuna Objective Function (যেখানে ম্যাজিক হবে)
    # ---------------------------------------------------------
    def objective(trial):
        # (ক) XGBoost এর জন্য রেঞ্জ সেট করা
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 500, 2000)
        xgb_lr = trial.suggest_float('xgb_lr', 0.01, 0.1, log=True)
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 3, 12)
        xgb_subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
        
        # (খ) LightGBM এর জন্য রেঞ্জ সেট করা
        lgbm_n_estimators = trial.suggest_int('lgbm_n_estimators', 500, 2000)
        lgbm_lr = trial.suggest_float('lgbm_lr', 0.01, 0.1, log=True)
        lgbm_num_leaves = trial.suggest_int('lgbm_num_leaves', 20, 100)
        
        # (গ) Random Forest এর জন্য রেঞ্জ সেট করা
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 500, 1500)
        rf_max_depth = trial.suggest_int('rf_max_depth', 10, 50)
        
        # (ঘ) ভোটিং ওয়েট (Weights) অপটিমাইজ করা! (এটা গেম চেঞ্জার হতে পারে)
        w_xgb = trial.suggest_float('w_xgb', 1.0, 3.0)
        w_lgbm = trial.suggest_float('w_lgbm', 1.0, 3.0)
        w_rf = trial.suggest_float('w_rf', 0.5, 2.0)

        # মডেল ইনিশিলাইজেশন
        clf_xgb = XGBClassifier(
            n_estimators=xgb_n_estimators, learning_rate=xgb_lr, max_depth=xgb_max_depth,
            subsample=xgb_subsample, colsample_bytree=0.7, n_jobs=-1, random_state=42, verbosity=0
        )
        
        clf_lgbm = LGBMClassifier(
            n_estimators=lgbm_n_estimators, learning_rate=lgbm_lr, num_leaves=lgbm_num_leaves,
            n_jobs=-1, random_state=42, verbose=-1
        )
        
        clf_rf = RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, n_jobs=-1, random_state=42
        )
        
        # ভোটিং ক্লাসিফায়ার
        model = VotingClassifier(
            estimators=[('xgb', clf_xgb), ('lgbm', clf_lgbm), ('rf', clf_rf)],
            voting='soft',
            weights=[w_xgb, w_lgbm, w_rf]
        )
        
        # 3-Fold Cross Validation দিয়ে চেক করা (Accuracy)
        # StratifiedKFold নিশ্চিত করে যে প্রত্যেক ফোল্ডে সমান ম্যালওয়্যার থাকে
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        return scores.mean()

    # ---------------------------------------------------------
    # ৫. স্টাডি রান করা (৫০ বার ট্রাই করবে)
    # ---------------------------------------------------------
    study = optuna.create_study(direction='maximize')
    print("\n🔍 Optuna is searching for the best hyperparameters... (This will take time)")
    study.optimize(objective, n_trials=30) # সময় বাঁচাতে ৩০ দিলাম, তুমি চাইলে ৫০ বা ১০০ দিতে পারো

    # ---------------------------------------------------------
    # ৬. বেস্ট রেজাল্ট দেখানো
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🎉 Optimization Finished!")
    print(f"Best Trial Accuracy: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    # রেজাল্ট সেভ করে রাখা
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print("Saved best parameters to 'best_params.json'")

if __name__ == "__main__":
    args = param_parser()
    main(args)