
from tsai.basics import *
from tsai.data.external import *
from tsai.data.preprocessing import *
from tsai.models.InceptionTimePlus import *
     

# With validation split
X, y, splits = get_classification_data('OliveOil', split_data=False)
tfms = [None, TSClassification()]
batch_tfms = [TSStandardize(by_sample=True)]
learn = TSClassifier(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, metrics=accuracy, arch=InceptionTimePlus, arch_config=dict(fc_dropout=.5), 
                     train_metrics=True)
learn.fit_one_cycle(1)