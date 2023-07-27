from sklearn.model_selection import KFold
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import f1_score
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTENC, RandomOverSampler
import matplotlib.pyplot as plt 

def from_probs_to_class(probs, threshold=0.5):
    preds = []
    for prob in probs:
        if prob[0] > threshold:
            preds.append(1)
        else:
            preds.append(0)
    return preds



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test, batch_size=64, verbose=1)
    y_pred_bool = from_probs_to_class(y_pred, threshold=0.5)
    print(classification_report(y_test, y_pred_bool))
    print("fscore on test set: ", f1_score(y_test, y_pred_bool, zero_division=True))
    print("confusion matrix:")
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_bool)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.show()
    
    
# CROSSVALIDATION FUNCTION TO TUNE THE MODEL


def crossvalidate_kerasmodel(model, X_train, y_train, epochs=20, n_splits=5, print_folds_scores = False, is_convolutional=False ):
    kf = KFold(n_splits=n_splits)
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()

    accuracy_list = []
    f1_list = []
    auc_list = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train_np), 1):
    
        X_train_split = X_train_np[train_index]
        y_train_split = y_train_np[train_index]  
        X_val_split = X_train_np[val_index]
        y_val_split = y_train_np[val_index]
    
        cat_columns = [1, 3, 4, 5, 6, 7, 8, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
        smote = SMOTENC(cat_columns, sampling_strategy=0.6)
        X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_split, y_train_split)
        smote = RandomUnderSampler( sampling_strategy="auto")
        X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_split, y_train_split)


        if is_convolutional:
            X_train_oversampled = X_train_oversampled.reshape(X_train_oversampled.shape[0],X_train_oversampled.shape[1],1)
            X_val_split = X_val_split.reshape(X_val_split.shape[0],X_val_split.shape[1],1)
          
        history = model.fit(X_train_oversampled, y_train_oversampled, verbose= False, epochs=10,  validation_data=(X_val_split, y_val_split))
    
        y_pred_probs = model.predict(X_val_split, verbose=False)
        y_pred = from_probs_to_class(y_pred_probs)
    
        f1 = f1_score(y_val_split, y_pred)
        auc = metrics.roc_auc_score(y_val_split, y_pred_probs)
        val_accuracy = history.history["val_accuracy"][-1:][0]
        train_accuracy = history.history["accuracy"][-1:][0]
        if print_folds_scores:
            print(f'For fold {fold}:')
            print(f'Accuracy on train set: {train_accuracy}')
            print(f'Accuracy on val set: {val_accuracy}')
            print(f'f-score on val set: {f1}')
            print(f'AUC score on val set: {auc}')
    
        accuracy_list.append(val_accuracy)
        f1_list.append(f1)
        auc_list.append(auc)

    print(f"Mean accuracy on the validation set {list_mean(accuracy_list)}")
    print(f"Mean f1-score on the validation set {list_mean(f1_list)}")
    print(f"Mean AUC score on the validation set {list_mean(auc_list)}")

    
def list_mean(lst):
    return sum(lst) / len(lst)