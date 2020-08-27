# Screening Data Points for Empirical Risk Minimization

This is the code associated to the AISTATS 2020 paper [Screening Data Points in Empirical Risk Minimization via Ellipsoidal Regions and Safe Loss Functions](http://proceedings.mlr.press/v108/mialon20a).

## API

The API is similar to scikit-learn.

```python
from sklearn.model_selection import train_test_split
from utils.loaders import load_experiment

X, y = load_experiment(dataset='cifar10_kernel', synth_params=None, size=10000, redundant=0, 
                        noise=None, classification=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# random initialization of the ellipsoid and the screener object.
z_init = np.random.rand(X_train.shape[1])
screener = EllipsoidScreener(lmbda=0, mu=0, loss='safe_logistic', penalty='l2', 
                            intercept=False, classification=True, n_ellipsoid_steps=2000, 
                            better_init=20, better_radius=1, cut=False, clip_ell=False, 
                            sgd=False, acceleration=True, dc=False, use_sphere=False,
                            ars=True).fit(X_train, y_train)

prop = np.unique(y_test, return_counts=True)[1]
print('BASELINE : ', 1 - prop[1] / prop[0])
print('SCORE SCREENER : ', screener.score(X_test, y_test))

# screening score of the samples
scores = screener.screen(X_train, y_train)
idx_safeell = np.where(scores > 0)[0]
print('NB TO KEEP', len(idx_safeell))

# we check that the scores are similar for the whole and screened datasets
if len(idx_safeell) !=0:
    estimator_whole = fit_estimator(X_train, y_train, loss='safe_logistic', penalty='l2', 
                                    mu=0, lmbda=0, intercept=False)
    print(y_train[idx_safeell][:10])
    print(estimator_whole.score(X_test, y_test))
    estimator_screened = fit_estimator(X_train[idx_safeell], y_train[idx_safeell], 
                                        loss='safe_logistic', penalty='l2', mu=0, 
                                        lmbda=0, intercept=False)
    print(estimator_screened.score(X_test, y_test))
```

### Installation

```
- Python 3.6
- numpy
- scipy
- matplotlib
- scikit-learn=0.19
- pandas
```

Run 

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

and then 

```bash
python experiment.py
```

Possible arguments are detailed in `experiment.py`.

#### Interval Regression : Install plasp

```bash
git clone https://github.com/VivienCabannes/plasp
cd plasp
pip install -e .
```

