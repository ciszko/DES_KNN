from des_knn import *
import numpy as np
import sys
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from deslib.des import DESKNN
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
from scipy.stats import ttest_ind
from tabulate import tabulate
from copy import deepcopy

state = 66

clfs = {
    "My DES_KNN": DES_KNN(random_state=state),
    "DES_KNN": DESKNN(random_state=state),
    "KNORA-U": KNORAU(),
    "KNORA-E": KNORAE(),
    "ADABoost": AdaBoostClassifier(),
}


def test(clf_pool, data, method=None):
    dataset = "./datasets/" + data
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    scores = np.zeros((len(clf_pool), n_splits * n_repeats))
    amount = (n_splits * n_repeats) * len(clf_pool)
    prc = "[%-" + str(amount) + "s] %d%%"
    progress = 0
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clf_pool):
            sys.stdout.write("\r")
            if method == "diversity":
                clf = deepcopy(clf_pool[clf_name])
            elif clf_name == "My DES_KNN":
                clf = deepcopy(clf_pool[clf_name])
            else:
                clf = clone(clf_pool[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
            progress += 1
            sys.stdout.write(prc % ("=" * progress, 100 / amount * progress,))

            sys.stdout.flush()
    if method == None:
        np.save("./results/" + data, scores)
    elif method == "diversity":
        np.save("./results/div" + data, scores)


def statistics(clf_pool, input_file, output_file, method=None):
    if method == "diversity":
        scores = np.load("./results/div" + input_file + ".npy")
    elif method == None:
        scores = np.load("./results/" + input_file + ".npy")
    scrs = []
    for clf_id, clf in enumerate(clf_pool):
        scrs.append([clf, np.mean(scores[clf_id])])
    # print("Folds:\n", scores)
    alfa = 0.05
    t_statistic = np.zeros((len(clf_pool), len(clf_pool)))
    p_value = np.zeros((len(clf_pool), len(clf_pool)))

    for i in range(len(clf_pool)):
        for j in range(len(clf_pool)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
    if method == None:
        headers = ["My DeS_KNN", "DES_KNN", "KNORA-U", "KNORA-E", "ADABoost"]
        names_column = np.array(
            [["My DES_KNN"], ["DES_KNN"], ["KNORA-U"], ["KNORA-E"], ["ADABoost"]]
        )
    elif method == "diversity":
        headers = ["double_fault", "Q", "p", "disagreement"]
        names_column = np.array([["double_fault"], ["Q"], ["p"], ["disagreement"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
    advantage = np.zeros((len(clf_pool), len(clf_pool)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(
        np.concatenate((names_column, advantage), axis=1), headers
    )
    print("\nAdvantage:\n", advantage_table)
    significance = np.zeros((len(clf_pool), len(clf_pool)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(
        np.concatenate((names_column, significance), axis=1), headers
    )
    print("Statistical significance (alpha = 0.05):\n", significance_table)
    stat_better = significance * advantage
    stat_better_table = tabulate(
        np.concatenate((names_column, stat_better), axis=1), headers
    )
    print("Statistically significantly better:\n", stat_better_table)

    with open("./results/" + output_file, "w") as f:
        f.write(f"SCORES: \n {tabulate(scrs)} \n\n")
        f.write(
            "#####################################################################\n\n"
        )
        f.write(
            f"t-statistic:\n  {t_statistic_table}  \n\np-value:\n {p_value_table}\n\n"
        )
        f.write(
            "#####################################################################\n\n"
        )
        f.write(f"Advantage:\n {advantage_table}\n\n")
        f.write(
            "#####################################################################\n\n"
        )
        f.write(f"Statistical significance (alpha = 0.05):\n {significance_table}\n\n")
        f.write(
            "#####################################################################\n\n"
        )
        f.write(f"Statistically significantly better:\n {stat_better_table}\n")


datasets = ["wisconsin", "german", "breast_cancer", "australian"]

# for dataset in datasets:
#     test(clfs, dataset)
#     statistics(clfs, dataset, dataset + ".txt")

double_fault = DES_KNN(random_state=state)
q = deepcopy(double_fault)
q.div_method = "Q"
p = deepcopy(double_fault)
p.div_method = "p"
dis = deepcopy(double_fault)
dis.div_method = "disagreement"
divs = {"double-fault": double_fault, "Q": q, "p": p, "disagreement": dis}

for dataset in datasets:
    test(divs, dataset, method="diversity")
    statistics(divs, dataset, "div" + dataset + "txt", method="diversity")
