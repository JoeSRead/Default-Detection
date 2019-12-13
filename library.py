import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.externals.six import StringIO
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

random_state = 137

def prediction_threshold(model, X_test_data, threshold=0.5):
    probas = model.predict_proba(X_test_data)[:,1]
    threshold_preds = np.where(probas>=threshold, 1, 0)
    return threshold_preds, probas

def plot_matrix(cnf_matrix, class_names=[0,1]):
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap='Blues' ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    
def plot_precision_recall_threshold(precision, recall, thresholds): 
    plt.plot(thresholds, precision[:-1], 'b--', label = 'Precision')
    plt.plot(thresholds, recall[:-1], 'g-', label = 'Recall')
    plt.legend()
    plt.grid()
    plt.xlabel('Threshold')