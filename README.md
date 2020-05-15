# DES_KNN
Dynamic Ensemble Selection K-nearest Neighbours Classifier project for university.


# Usage

DES_KNN supports similiar methods to sklearn classifiers such as: `predict()` and  `fit()`. It uses sklearn built-in classifier `DecisionTreeClassifier()` to make predictions but the base classifier can be changed to everything that supports two mentioned methods.

## Test

In /results folder you can find statistic test between this implementation of `DES_KNN`, `deslib.DESKNN`, `deslib.KNORAU`, `deslib.KNORAE` and `sklearn.ADABoost`

## Example test resuts

SCORES:

|  Classifier| Mean accuracy |
|--|--|
|  My DES_KNN| 0.848551 |
| DES_KNN | 0.846377 |
|  KNORA-U |  0.85 |
|KNORA-E |0.828986|
|ADABoost  |0.839855|

---------

t-statistic:

|          | My DeS_KNN   | DES_KNN  |  KNORA-U   | KNORA-E  |  ADABoost|
|---------- | ------------ | ---------|  --------- | ---------  |----------
|My DES_KNN  |        0.00    |   0.13   |   -0.09    |   1.16     |   0.55
|DES_KNN     |       -0.13     |  0.00   |   -0.23   |    1.07      |  0.43
|KNORA-U     |        0.09   |    0.23   |    0.00     |  1.30      |  0.68
|KNORA-E       |     -1.16    |  -1.07   |   -1.30     |  0.00      | -0.69
|ADABoost     |      -0.55    |  -0.43    |  -0.68     |  0.69     |   0.00  

--------
p-value:

   |  |    My DeS_KNN  |  DES_KNN   | KNORA-U  |  KNORA-E   | ADABoost|
|---------- | ------------|  --------- | --------- | ---------  |----------|
| My DES_KNN     |  1.00    |   0.90   |    0.93   |    0.26     |   0.59|
|DES_KNN         |  0.90     |  1.00     |  0.82   |    0.30  |      0.67|
|KNORA-U       |    0.93   |    0.82    |   1.00   |    0.21    |    0.50|
|KNORA-E       |   0.26    |   0.30   |    0.21   |    1.00   |     0.50|
|ADABoost    |     0.59    |   0.67    |   0.50     |  0.50      |  1.00|

-----
Advantage:
| |  My DeS_KNN  |  DES_KNN   | KNORA-U  |  KNORA-E   | ADABoost|
|---------- | ------------ | --------- | --------- | ---------|----------|
|My DES_KNN   |      0     |     1      |    0      |    1      |     1|
|DES_KNN   |   0     |     0     |     0      |    1      |     1|
|KNORA-U    |    1      |    1    |      0       |   1     |      1|
|KNORA-E      |     0      |    0    |      0     |     0    |       0|
|ADABoost    |       0      |    0     |     0     |     1     |      0|

---------------------

Statistical significance (alpha = 0.05):

 |       |       My DeS_KNN |   DES_KNN  |  KNORA-U |   KNORA-E    |ADABoost|
|---------- | ------------ | ---------  |---------|  ---------  |----------
|My DES_KNN    |         0  |        0      |    0      |    0     |      0
|DES_KNN          |      0    |      0       |   0   |       0        |   0
|KNORA-U         |       0     |     0      |    0    |      0       |    0
|KNORA-E        |        0      |    0     |     0     |     0      |     0
|ADABoost      |         0       |   0    |      0      |    0     |      0

-----------------

Statistically significantly better:
|  | My DeS_KNN   | DES_KNN  |  KNORA-U |   KNORA-E   | ADABoost
|---------- | ------------|  --------- | --------- | ---------  |----------|
|My DES_KNN  |           0  |        0  |        0  |        0    |       0
|DES_KNN        |        0      |    0     |     0      |    0       |    0
|KNORA-U       |         0      |    0    |      0      |    0      |     0
|KNORA-E       |         0       |   0    |      0       |   0      |     0
|ADABoost     |          0        |  0   |       0        |  0     |      0

