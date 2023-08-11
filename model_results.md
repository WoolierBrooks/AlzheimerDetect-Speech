### TimeSeriesSVC
@1 Hz
Accuracy Scores: [0.55454545 0.62727273 0.54545455 0.48181818 0.52293578]  
Mean Accuracy: 0.5464053377814846  
Standard Deviation Accuracy: 0.04760046440237219  

Precision Scores: [0.57647059 0.64705882 0.56790123 0.52941176 0.56521739]  
Mean Precision: 0.5772119604685674  
Standard Deviation Precision: 0.038456603941097595  

Recall Scores: [0.79032258 0.72131148 0.75409836 0.59016393 0.63934426]  
Mean Recall: 0.6990481226864093  
Standard Deviation Recall: 0.07386139675004359  

F1 Scores: [0.66666667 0.68217054 0.64788732 0.55813953 0.6       ]  
Mean F1: 0.6309728136259418  
Standard Deviation F1: 0.04570636921958017

Roc_auc Scores: [0.56922043 0.66008699 0.55035129 0.43860823 0.53756831]  
Mean Roc_auc: 0.5511670479931793  
Standard Deviation Roc_auc: 0.070773661955492  

#@16
#C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\tslearn\metrics\softdtw_variants.py:259: RuntimeWarning: invalid value encountered in matmul
#  return diagonal_left @ unnormalized_matrix @ diagonal_right
#C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\tslearn\metrics\softdtw_variants.py:259: RuntimeWarning: invalid value encountered in matmul
#  return diagonal_left @ unnormalized_matrix @ diagonal_right
#C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\tslearn\metrics\softdtw_variants.py:259: RuntimeWarning: invalid value encountered in matmul
#  return diagonal_left @ unnormalized_matrix @ diagonal_right
#C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\tslearn\metrics\softdtw_variants.py:259: RuntimeWarning: invalid value encountered in matmul
##  return diagonal_left @ unnormalized_matrix @ diagonal_right
#C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\tslearn\metrics\softdtw_variants.py:259: RuntimeWarning: invalid value encountered in matmul
#  return diagonal_left @ unnormalized_matrix @ diagonal_right
#Traceback (most recent call last):
#  File "c:\Users\Lenovo\Desktop\IC\AlzheimerDetect-Speech\scripts\ml_algorithms\svm_5cv.py", line 41, in <module>
#    scores = cross_val_score(clf, X, y, cv=5, scoring=metric)
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\model_selection\_validation.py", line 562, in cross_val_score
#    cv_results = cross_validate(
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\utils\_param_validation.py", line 211, in wrapper
#    return func(*args, **kwargs)
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\model_selection\_validation.py", line 328, in cross_validate
#    _warn_or_raise_about_fit_failures(results, error_score)
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\model_selection\_validation.py", line 414, in _warn_or_raise_about_fit_failures
#    raise ValueError(all_fits_failed_message)
#ValueError: 
#All the 5 fits failed.
#It is very likely that your model is misconfigured.
#You can try to debug the error by setting error_score='raise'.

#Below are more details about the failures:
#--------------------------------------------------------------------------------
#5 fits failed with the following error:
#Traceback (most recent call last):
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\model_selection\_validation.py", line 732, in _fit_and_score
#    estimator.fit(X_train, y_train, **fit_params)
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\tslearn\svm\svm.py", line 275, in fit
#    self.svm_estimator_.fit(sklearn_X, y, sample_weight=sample_weight)
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\base.py", line 1151, in wrapper
#    return fit_method(estimator, *args, **kwargs)
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\svm\_base.py", line 190, in fit
#    X, y = self._validate_data(
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\base.py", line 621, in _validate_data
#    X, y = check_X_y(X, y, **check_params)
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\utils\validation.py", line 1147, in check_X_y
#    X = check_array(
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\utils\validation.py", line 959, in check_array
#    _assert_all_finite(
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\utils\validation.py", line 124, in _assert_all_finite
#    _assert_all_finite_element_wise(
#  File "C:\Users\Lenovo\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310\site-packages\sklearn\utils\validation.py", line 173, in _assert_all_finite_element_wise
#    raise ValueError(msg_err)
#ValueError: Input X contains NaN.
#SVC does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values

-----------------------------------------------------------------------------------------------------------------------------------
### KNeighborsTimeSeriesClassifier
@1 Hz n_neighbors=2  
Accuracy Scores: [0.5        0.52727273 0.44545455 0.5        0.6146789 ]  
Mean Accuracy: 0.5174812343619684  
Standard Deviation Accuracy: 0.05539366459291622

Precision Scores: [0.56862745 0.57377049 0.5        0.55172414 0.6557377 ]  
Mean Precision: 0.5699719571265476  
Standard Deviation Precision: 0.05020170862871416

Recall Scores: [0.46774194 0.57377049 0.42622951 0.52459016 0.6557377 ]  
Mean Recall: 0.529613960867266  
Standard Deviation Recall: 0.0804968684513135

F1 Scores: [0.51327434 0.57377049 0.46017699 0.53781513 0.6557377 ]  
Mean F1: 0.548154930041072  
Standard Deviation F1: 0.06525932202150338

Roc_auc Scores: [0.50537634 0.5192372  0.44245567 0.52224824 0.61577869]  
Mean Roc_auc: 0.5210192300082381  
Standard Deviation Roc_auc: 0.05549950232057674

-----------------------------------------------------------------------------------------------------------------------------------
@1 Hz n_neighbors=3
Accuracy Scores: [0.54545455 0.52727273 0.51818182 0.56363636 0.59633028]  
Mean Accuracy: 0.5501751459549624  
Standard Deviation Accuracy: 0.02786351812812944

Precision Scores: [0.57692308 0.55294118 0.55       0.57831325 0.6       ]  
Mean Precision: 0.5716355012811427  
Standard Deviation Precision: 0.01841015104625532

Recall Scores: [0.72580645 0.7704918  0.72131148 0.78688525 0.83606557]  
Mean Recall: 0.7681121099947118  
Standard Deviation Recall: 0.04232218080320857

F1 Scores: [0.64285714 0.64383562 0.62411348 0.66666667 0.69863014]  
Mean F1: 0.6552206076251543  
Standard Deviation F1: 0.025555924876472187

Roc_auc Scores: [0.49613575 0.53680161 0.46453663 0.54098361 0.60843579]  
Mean Roc_auc: 0.5293786783618788  
Standard Deviation Roc_auc: 0.04847552839462101

-----------------------------------------------------------------------------------------------------------------------------------
@1 Hz n_neighbors=4  
Accuracy Scores: [0.48181818 0.49090909 0.43636364 0.6        0.6146789 ]  
Mean Accuracy: 0.5247539616346956  
Standard Deviation Accuracy: 0.07007177281771236

Precision Scores: [0.57142857 0.56097561 0.48484848 0.71794872 0.71111111]  
Mean Precision: 0.6092624990185966  
Standard Deviation Precision: 0.09102495333034194

Recall Scores: [0.32258065 0.37704918 0.26229508 0.45901639 0.52459016]  
Mean Recall: 0.3891062929666843  
Standard Deviation Recall: 0.09369676670752773

F1 Scores: [0.41237113 0.45098039 0.34042553 0.56       0.60377358]  
Mean F1: 0.47351012859960717  
Standard Deviation F1: 0.09633913167627944

Roc_auc Scores: [0.5        0.53696889 0.50217464 0.63716962 0.6489071 ]  
Mean Roc_auc: 0.565044050407048  
Standard Deviation Roc_auc: 0.06512544184015788

-----------------------------------------------------------------------------------------------------------------------------------
@1 Hz n_neighbors=5  
Accuracy Scores: [0.52727273 0.55454545 0.53636364 0.59090909 0.65137615]    
Mean Accuracy: 0.5720934111759799  
Standard Deviation Accuracy: 0.04523985690893477

Precision Scores: [0.6        0.60714286 0.57575758 0.64285714 0.66666667]  
Mean Precision: 0.6184848484848484  
Standard Deviation Precision: 0.03228450576389829

Recall Scores: [0.48387097 0.55737705 0.62295082 0.59016393 0.75409836]  
Mean Recall: 0.6016922263352723  
Standard Deviation Recall: 0.08905969120348647

F1 Scores: [0.53571429 0.58119658 0.5984252  0.61538462 0.70769231]  
Mean F1: 0.6076825973676367  
Standard Deviation F1: 0.056617858737225975

Roc_auc Scores: [0.53360215 0.58732017 0.49799264 0.6189361  0.67793716]  
Mean Roc_auc: 0.5831576443374812  
Standard Deviation Roc_auc: 0.06322880001480716

-----------------------------------------------------------------------------------------------------------------------------------
@1 Hz n_neighbors=6  
Accuracy Scores: [0.53636364 0.56363636 0.5        0.57272727 0.66055046]  
Mean Accuracy: 0.5666555462885738  
Standard Deviation Accuracy: 0.05333441208720832

Precision Scores: [0.64864865 0.65116279 0.56818182 0.65909091 0.73076923]  
Mean Precision: 0.6515706794776562  
Standard Deviation Precision: 0.05155845460399141

Recall Scores: [0.38709677 0.45901639 0.40983607 0.47540984 0.62295082]  
Mean Recall: 0.4708619777895294  
Standard Deviation Recall: 0.08250186422602991  

F1 Scores: [0.48484848 0.53846154 0.47619048 0.55238095 0.67256637]  
Mean F1: 0.5448895647125735  
Standard Deviation F1: 0.07031828138322756

Roc_auc Scores: [0.55275538 0.57929073 0.51773168 0.6231181  0.68647541]  
Mean Roc_auc: 0.5918742602805268  
Standard Deviation Roc_auc: 0.0585001009160348

-----------------------------------------------------------------------------------------------------------------------------------
@1 Hz n_neighbors=7  
Accuracy Scores: [0.52727273 0.59090909 0.51818182 0.57272727 0.63302752]  
Mean Accuracy: 0.5684236864053378  
Standard Deviation Accuracy: 0.04222657155276311

Precision Scores: [0.58928571 0.63333333 0.5625     0.62962963 0.65217391]  
Mean Precision: 0.6133845180584311  
Standard Deviation Precision: 0.03266944310958965

Recall Scores: [0.53225806 0.62295082 0.59016393 0.55737705 0.73770492]  
Mean Recall: 0.608090957165521  
Standard Deviation Recall: 0.0716442608733127

F1 Scores: [0.55932203 0.62809917 0.576      0.59130435 0.69230769]  
Mean F1: 0.6094066495171607  
Standard Deviation F1: 0.04727400899179268

Roc_auc Scores: [0.54250672 0.5938441  0.48427568 0.62094346 0.67554645]  
Mean Roc_auc: 0.5834232800735313  
Standard Deviation Roc_auc: 0.06558574124648849
