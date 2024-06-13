# Interspeech 2024: Analyzing Multimodal Features of Spontaneous Voice Assistant Commands for Mild Cognitive Impairment Detection
import os
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.svm import SVR, LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_intent_embeddings_all(text_dir_all, task):
    all_embeddings = []
    for session_dir in sorted(os.listdir(text_dir_all)):
        if session_dir == ".DS_Store": continue
        session_path = f"{text_dir_all}/{session_dir}"
        task_path = f"{session_path}/{task}"
        for file in sorted(os.listdir(task_path)):
            file_path = f"{task_path}/{file}"
            if file.endswith('.npy'):
                print(f"generate embedding for participant {file.split(".")[0]}")
                participant_embedding = np.load(file_path)
                all_embeddings.append(participant_embedding)

    return all_embeddings

def load_multi_embeddings_all(embedding_dir_all, task):
    print(".....................")
    averaged_embeddings_list = []
    for session_dir in sorted(os.listdir(hubert_dir_all)):
        session_path = f"{embedding_dir_all}/{session_dir}"
        task_path = f"{session_path}/{task}"
        for par_dir in sorted(os.listdir(task_path)):
            par_path = f"{task_path}/{par_dir}"
            if os.path.isdir(par_path):
                participant_embeddings = []
                for filename in os.listdir(par_path):
                    if filename.endswith(".npy"):
                        embeddings = np.load(os.path.join(par_path, filename))
                        participant_embeddings.append(embeddings)

                if participant_embeddings:
                    average_embedding = np.mean(participant_embeddings, axis=0).flatten().tolist()
                    averaged_embeddings_list.append(average_embedding)

    print("averaged embeddings", len(averaged_embeddings_list))
    return averaged_embeddings_list

def classification_and_metrics(model, features, labels, splits:int, reps:int):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    folds = KFold(n_splits=splits, shuffle=True, random_state=42)

    for _ in range(reps):
        for train_index, test_index in folds.split(features, labels):
            X_train, X_test, y_train, y_test = np.array(features)[train_index], np.array(features)[test_index], \
                np.array(labels)[train_index], np.array(labels)[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred))

    mean_accuracy = np.mean(accuracy_scores)
    max_accuracy = np.max(accuracy_scores)
    print("accuracy: ", accuracy_scores)

    mean_precision = np.mean(precision_scores)
    max_precision = np.max(precision_scores)
    print("precision: ", precision_scores)

    mean_recall = np.mean(recall_scores)
    max_recall = np.max(recall_scores)
    print("recall: ", recall_scores)

    mean_f1 = np.mean(f1_scores)
    max_f1 = np.max(f1_scores)
    print("f1: ", f1_scores)

    return mean_accuracy, max_accuracy, mean_precision, max_precision, mean_recall, max_recall, mean_f1, max_f1

def get_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    max_target = np.max(y_test)
    min_target = np.min(y_test)
    mse = mean_squared_error(y_test, y_predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_predicted)
    rrmse = rmse / (max_target - min_target)
    rrmse_v2 = rmse / np.sqrt(np.sum((y_test-0) ** 2))
    return rmse, mae, rrmse, rrmse_v2

def regression_and_metrics(features, labels, model, splits:int, reps:int):
    rmses = []
    maes = []
    rrmses = []
    rrmse_v2s = []

    folds = KFold(n_splits=splits, shuffle=True, random_state=2022)

    for _ in range(reps):
        for train_index, test_index in folds.split(features, labels):
            X_train, X_test, y_train, y_test = np.array(features)[train_index], np.array(features)[test_index], \
                np.array(labels)[train_index], np.array(labels)[test_index]
            rmse, mae, rrmse, rrmse_v2 = get_metrics(model, X_train, X_test, y_train, y_test)

            rmses.append(rmse)
            maes.append(mae)
            rrmses.append(rrmse)
            rrmse_v2s.append(rrmse_v2)

    mean_rmse = mean(rmses)
    mean_mae = mean(maes)
    mean_rrmse = mean(rrmses)
    mean_rrmse_v2 = mean(rrmse_v2s)

    return mean_mae, mean_rmse, mean_rrmse, mean_rrmse_v2

def save_results_to_excel(results, file_path):
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)

    df.to_excel(file_path, index=False)

hubert_dir_all = "data/HuBERT_Embeddings_PhaseII"
intent_dir_all = "data/Intent_Embeddings_PhaseII"
text_dir_all = "data/BERT_Embeddings_PhaseII"
task = "task2"

print("------------- Loading FEATURES --------------")
INTENT_FEATURES = load_intent_embeddings_all(intent_dir_all, task)
HUBERT_FEATURES = load_multi_embeddings_all(hubert_dir_all, task)
BERT_FEATURES = load_multi_embeddings_all(text_dir_all, task)
COMBINED_FEATURES_1 = [INTENT_FEATURES[i] + BERT_FEATURES[i] for i in range(len(INTENT_FEATURES))]
COMBINED_FEATURES_2 = [INTENT_FEATURES[i] + HUBERT_FEATURES[i] for i in range(len(INTENT_FEATURES))]
COMBINED_FEATURES_3 = [HUBERT_FEATURES[i] + BERT_FEATURES[i] for i in range(len(HUBERT_FEATURES))]
COMBINED_FEATURES_4 = [INTENT_FEATURES[i] + HUBERT_FEATURES[i] + BERT_FEATURES[i] for i in range(len(INTENT_FEATURES))]
# print(len(HUBERT_FEATURES))
# print(len(BERT_FEATURES))
# print(len(COUNT_FEATURES)

# Classification
# Loading labels
df = pd.read_csv('labels.csv')
Labels = df['label'].tolist()

FEATURES = [INTENT_FEATURES, HUBERT_FEATURES, BERT_FEATURES, COMBINED_FEATURES_1, COMBINED_FEATURES_2, COMBINED_FEATURES_3, COMBINED_FEATURES_4]
FEATURE_NAMES = ["INTENT_FEATURES", "HUBERT_FEATURES", "BERT_FEATURES", "COMBINED_FEATURES_1", "COMBINED_FEATURES_2", "COMBINED_FEATURES_3", "COMBINED_FEATURES_4"]

dt_model = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=9, min_samples_split=9)
svm_model = LinearSVC(C=0.1, dual=True, loss="squared_hinge", penalty="l2", tol=0.001)
knn_model = KNeighborsClassifier(n_neighbors=1)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

MODELS = [dt_model, svm_model, knn_model, rf_classifier]
model_names = ['dt_model', 'svm_model', 'knn_model', 'rf_classifier']

for i, model in enumerate(MODELS):
    for j, feature in enumerate(FEATURES):
        mean_accuracy, max_accuracy, mean_precision, max_precision, mean_recall, max_recall, mean_f1, max_f1 = classification_and_metrics(model, feature, Labels, 10, 10)

        result = {
            'model': model_names[i],
            'feature': FEATURE_NAMES[j],
            'mean_accuracy': mean_accuracy,
            'std_accuracy': max_accuracy,
            'mean_precision': mean_precision,
            'std_precision': max_precision,
            'mean_recall': mean_recall,
            'std_recall': max_recall,
            'mean_f1': mean_f1,
            'std_f1': max_f1
        }
        save_results_to_excel([result], f'Results/Classification_results.xlsx')

print("Classification Done!")


# Regression
print("-------------------- Loading domain Scores ----------------------")
MoCA = df['moca'].tolist()
language = df['attention'].tolist()
orientation = df['orientation'].tolist()
visuospatial = df['visuospatial'].tolist()
executive = df['executive'].tolist()
memory = df['memory'].tolist()
attention = df['attention'].tolist()

domain_dict = {
    "MoCA": MoCA,
    "Language": language,
    "Orientation": orientation,
    "Visuospatial": visuospatial,
    "Executive Function": executive,
    "Memory": memory,
    "Attention": attention,
}

range_dict = {
    "MoCA": 30,
    "Language": 6,
    "Orientation": 6,
    "Visuospatial": 7,
    "Executive Function": 13,
    "Memory": 15,
    "Attention": 18,
}

for domain in domain_dict.keys():
    scores = domain_dict[domain]
    print(f"{domain} list length: {len(scores)}")

    # normalization
    scores = [x * 30 / range_dict[domain] for x in scores]

    # regressiion labels
    regression_labels = scores

    dtr_model, svr_model, ridge_model = DecisionTreeRegressor(), SVR(), Ridge()
    models_regression = [dtr_model, svr_model, ridge_model]
    model_names_regression = ['dtr_model', 'svr_model', 'ridge_model']
    idx = 0
    splits = 10
    for i, model in enumerate(models_regression):
        model_name = model_names_regression[i]
        print(model_name)
        mae, rmse, rrmse, rrmse_v2 = regression_and_metrics(COMBINED_FEATURES_4, regression_labels, model, splits, 10)

        result = {
            'domain': domain,
            'model': model_names[i],
            'MAE': mae,
            'RMSE': rmse,
            'RRMSE': rrmse
        }
        save_results_to_excel([result], f'Results/Regression_results.xlsx')

print("Regression Done!")