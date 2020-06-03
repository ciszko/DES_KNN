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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.preprocessing import normalize
import os

state = 2404

clf_pool = {
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
    n_repeats = 3
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
            if method == "diversity" or "param" or "distance":
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
    elif method == "param":
        np.save("./results/par" + data, scores)
    elif method == "distance":
        np.save("./results/dis" + data, scores)


def statistics(clf_pool, input_file, output_file, method=None):
    if method == "diversity":
        scores = np.load("./results/div" + input_file + ".npy")
    elif method == None:
        scores = np.load("./results/" + input_file + ".npy")
    elif method == "distance":
        scores = np.load("./results/dis" + input_file + ".npy")
    scrs = []
    for clf_id, clf in enumerate(clf_pool):
        scrs.append([clf, np.mean(scores[clf_id])])
    print(f"SCORES: \n {tabulate(scrs)} \n\n")
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
    elif method == "distance":
        headers = ["euclidean", "manhattan"]
        names_column = np.array([["euclidean"], ["manhattan"]])        
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

def global_stats():
    for i,file in enumerate(os.listdir("./results/")):
        if file.endswith('.npy'):
            x = np.load("./results/" + file)
            if i == 0:
                scores = x
            else:
               scores = np.dstack((scores, x))
    scores = np.transpose(scores, [0, 2, 1])
    mean_scores = np.mean(scores, axis=2).T
    print("\nMean scores:\n", mean_scores)
    scrs = []
    for clf_id, clf in enumerate(clf_pool):
        scrs.append([clf, np.mean(mean_scores.T[clf_id])])
    # print("Folds:\n", scores)
    alfa = 0.05
    t_statistic = np.zeros((len(clf_pool), len(clf_pool)))
    p_value = np.zeros((len(clf_pool), len(clf_pool)))

    for i in range(len(clf_pool)):
        for j in range(len(clf_pool)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(mean_scores.T[i], mean_scores.T[j])
    headers = ["My DeS_KNN", "DES_KNN", "KNORA-U", "KNORA-E", "ADABoost"]
    names_column = np.array(
        [["My DES_KNN"], ["DES_KNN"], ["KNORA-U"], ["KNORA-E"], ["ADABoost"]]
        )
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


    with open("./results/global.txt", "w") as f:
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
    



# datasets = ["australian", "banknote", "breastcancoimbra", "cryotherapy", "heart", "liver",
# "lol_ranked_10min", "monkone", "monkthree", "sonar"]

# for dataset in datasets:
#     test(clf_pool, dataset)
#     statistics(clf_pool, dataset, dataset + ".txt")

# global_stats()

def test_param():

    datasets = ["australian", "banknote", "breastcancoimbra", "cryotherapy"]

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
        statistics(divs, dataset, "div" + dataset + ".txt", method="diversity")

# dataset = "australian"

# base = DES_KNN()
# clfs = {}
# for i in range(10, 31):
#     clfs[i] = deepcopy(base)
#     clfs[i].n_estimators = i

# test(clfs, dataset, method="param")


def stat_to_plot(clf_pool):
    scores = np.load("./results/paraustralian.npy")
    scrs = []
    for clf_id, clf in enumerate(clf_pool):
        scrs.append([np.mean(scores[clf_id])])

    plt.plot(list(range(10, 31)), scrs, '-o')
    plt.ylabel("Accuracy")
    plt.xlabel("Number of base estimators")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.gca().set_xlim([10,30])
    plt.gca().set_ylim([0.5,1])
    plt.grid(axis='x')
    plt.show()

def test_distance():

    datasets = ["banknote", "breastcancoimbra", "cryotherapy"]

    euclidean = DES_KNN(random_state=state)
    manhattan = deepcopy(euclidean)
    manhattan.knn_metrics = 'manhattan'
    divs = {"euclidean": euclidean, "manhattan": manhattan}

    for dataset in datasets:
        test(divs, dataset, method="distance")
        statistics(divs, dataset, "dis" + dataset + ".txt", method="distance")

test_distance()