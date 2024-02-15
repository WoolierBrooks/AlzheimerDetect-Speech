import ktrain
from ktrain import vision as vis

# Set the path to your CSV files
control_dir = r"C:\Users\Lenovo\Desktop\IC\[99] ADRESS-20 Database DL\train\cc"
dementia_dir = r"C:\Users\Lenovo\Desktop\IC\[99] ADRESS-20 Database DL\train\cd"
test_dir = r"C:\Users\Lenovo\Desktop\IC\[99] ADRESS-20 Database DL\test"

# Replace `"your_image_column_name"` and `"your_label_column_name"` with the actual columns
# in your CSV files
(control_labels, control_data) = vis.preprocess_csv(
    control_dir,
    x_col="your_image_column_name",
    y_col="your_label_column_name",
    suffix=".csv",
    # Specify the output CSV file for control data
    csv_out="preprocessed_control.csv"
)

trn_control, val_control, preproc_control = vis.images_from_csv(
    control_data,
    "image_name",
    directory=control_dir,
    val_filepath=None,
    label_columns=control_labels,
    data_aug=vis.get_data_aug(horizontal_flip=True, vertical_flip=True)
)

# Do the same for dementia data, updating filenames and column names
(dementia_labels, dementia_data) = vis.preprocess_csv(
    dementia_dir,
    x_col="your_image_column_name",
    y_col="your_label_column_name",
    suffix=".csv",
    csv_out="preprocessed_dementia.csv"
)

trn_dementia, val_dementia, preproc_dementia = vis.images_from_csv(
    dementia_data,
    "image_name",
    directory=dementia_dir,
    val_filepath=None,
    label_columns=dementia_labels,
    data_aug=vis.get_data_aug(horizontal_flip=True, vertical_flip=True)
)

# Combine the control and dementia data
trn = trn_control + trn_dementia
val = val_control + val_dementia

# Create a LeNet model
model = vis.image_classifier('lenet', trn, val_data=val)

# Get a learner
learner = ktrain.get_learner(model, train_data=trn, val_data=val, 
                             batch_size=64, workers=8, use_multiprocessing=False)

# Train the model using one cycle policy
learner.lr_find()
learner.lr_plot()
learner.fit_onecycle(1e-4, 20)

# Evaluate the model
y_pred = learner.model.predict_generator(val)
y_true = val.labels

# Use the provided f2 function to calculate the F2 score
f2_score = f2(y_pred, y_true)

# Corrected version:
print('F2 Score: {f2_score}')
