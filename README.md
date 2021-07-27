# Sklearn_Ex_master
Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is also known in the literature as logit regression, maximum-entropy classification (MaxEnt) or the log-linear classifier. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

Logistic regression is implemented in LogisticRegression. This implementation can fit binary, One-vs-Rest, or multinomial logistic regression with optional 
, 
 or Elastic-Net regularization.

Note Regularization is applied by default, which is common in machine learning but not in statistics. Another advantage of regularization is that it improves numerical stability. No regularization amounts to setting C to a very high value.
As an optimization problem, binary class 
 penalized logistic regression minimizes the following cost function:

 
 
 
Similarly, 
 regularized logistic regression solves the following optimization problem:

 
 
Elastic-Net regularization is a combination of 
 and 
, and minimizes the following cost function:

 
 
 
where  controls the strength of 
 regularization vs. 
 regularization (it corresponds to the l1_ratio parameter).

Note that, in this notation, it’s assumed that the target 
 takes values in the set  at trial . We can also see that Elastic-Net is equivalent to 
 when  and equivalent to 
 when .

The solvers implemented in the class LogisticRegression are “liblinear”, “newton-cg”, “lbfgs”, “sag” and “saga”:

The solver “liblinear” uses a coordinate descent (CD) algorithm, and relies on the excellent C++ LIBLINEAR library, which is shipped with scikit-learn. However, the CD algorithm implemented in liblinear cannot learn a true multinomial (multiclass) model; instead, the optimization problem is decomposed in a “one-vs-rest” fashion so separate binary classifiers are trained for all classes. This happens under the hood, so LogisticRegression instances using this solver behave as multiclass classifiers. For 
 regularization sklearn.svm.l1_min_c allows to calculate the lower bound for C in order to get a non “null” (all feature weights to zero) model.

The “lbfgs”, “sag” and “newton-cg” solvers only support 
 regularization or no regularization, and are found to converge faster for some high-dimensional data. Setting multi_class to “multinomial” with these solvers learns a true multinomial logistic regression model 5, which means that its probability estimates should be better calibrated than the default “one-vs-rest” setting.

The “sag” solver uses Stochastic Average Gradient descent 6. It is faster than other solvers for large datasets, when both the number of samples and the number of features are large.

The “saga” solver 7 is a variant of “sag” that also supports the non-smooth penalty="l1". This is therefore the solver of choice for sparse multinomial logistic regression. It is also the only solver that supports penalty="elasticnet".

The “lbfgs” is an optimization algorithm that approximates the Broyden–Fletcher–Goldfarb–Shanno algorithm 8, which belongs to quasi-Newton methods. The “lbfgs” solver is recommended for use for small data-sets but for larger datasets its performance suffers. 9

The following table summarizes the penalties supported by each solver:

Solvers

Penalties

‘liblinear’

‘lbfgs’

‘newton-cg’

‘sag’

‘saga’

Multinomial + L2 penalty

no

yes

yes

yes

yes

OVR + L2 penalty

yes

yes

yes

yes

yes

Multinomial + L1 penalty

no

no

no

no

yes

OVR + L1 penalty

yes

no

no

no

yes

Elastic-Net

no

no

no

no

yes

No penalty (‘none’)

no

yes

yes

yes

yes

Behaviors

Penalize the intercept (bad)

yes

no

no

no

no

Faster for large datasets

no

no

no

yes

yes

Robust to unscaled datasets

yes

yes

yes

no

no

The “lbfgs” solver is used by default for its robustness. For large datasets the “saga” solver is usually faster. For large dataset, you may also consider using SGDClassifier with ‘log’ loss, which might be even faster but requires more tuning.

Examples:

L1 Penalty and Sparsity in Logistic Regression

Regularization path of L1- Logistic Regression

Plot multinomial and One-vs-Rest Logistic Regression

Multiclass sparse logistic regression on 20newgroups

MNIST classification using multinomial logistic + L1

Differences from liblinear:

There might be a difference in the scores obtained between LogisticRegression with solver=liblinear or LinearSVC and the external liblinear library directly, when fit_intercept=False and the fit coef_ (or) the data to be predicted are zeroes. This is because for the sample(s) with decision_function zero, LogisticRegression and LinearSVC predict the negative class, while liblinear predicts the positive class. Note that a model with fit_intercept=False and having many samples with decision_function zero, is likely to be a underfit, bad model and you are advised to set fit_intercept=True and increase the intercept_scaling.

Note Feature selection with sparse logistic regression
A logistic regression with 
 penalty yields sparse models, and can thus be used to perform feature selection, as detailed in L1-based feature selection.

Note P-value estimation
It is possible to obtain the p-values and confidence intervals for coefficients in cases of regression without penalization. The statsmodels package <https://pypi.org/project/statsmodels/> natively supports this. Within sklearn, one could use bootstrapping instead as well.

LogisticRegressionCV implements Logistic Regression with built-in cross-validation support, to find the optimal C and l1_ratio parameters according to the scoring attribute. The “newton-cg”, “sag”, “saga” and “lbfgs” solvers are found to be faster for high-dimensional dense data, due to warm-starting (see Glossary).
