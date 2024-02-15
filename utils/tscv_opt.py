import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from kerastuner import BayesianOptimization
from keras_tuner.src.engine import tuner_utils

class TSCVBayesianOptimization(BayesianOptimization):
    '''A modified version of the BayesianOptimization class of keras-tuner that allows for TimeSeriesSplit cross validation
    '''
    def __init__(self, time_series_splits=5, *args, **kwargs):
        # Initializes the TSCVTuner class with the number of splits for the TimeSeriesSplit cross validation
        super(TSCVBayesianOptimization, self).__init__(*args, **kwargs)
        self.time_series_splits = time_series_splits
        self.tscv = TimeSeriesSplit(n_splits=self.time_series_splits)

    def run_trial(self, trial, *args, **kwargs):
        # Unpack the data as the first two arguments
        X, y, *remaining_args = args

        # Callback to save the best epoch
        model_checkpoint = tuner_utils.SaveBestEpoch(
            objective=self.oracle.objective,
            filepath=self._get_checkpoint_fname(trial.trial_id),
        )
        original_callbacks = kwargs.pop("callbacks", [])
        

        # Track the histories
        val_losses = []

        for execution in range(self.executions_per_trial):

            for train_index, val_index in self.tscv.split(X.squeeze()):
                X_train_split, X_val_split = X[:,train_index], X[:,val_index]
                y_train_split, y_val_split = y[:,train_index], y[:,val_index]

                # Build the model for this trial's hyperparameters
                model = self.hypermodel.build(trial.hyperparameters)

                # Set up callbacks
                copied_callbacks = self._deepcopy_callbacks(original_callbacks)
                self._configure_tensorboard_dir(copied_callbacks, trial, execution)
                copied_callbacks.append(tuner_utils.TunerCallback(self, trial))
                copied_callbacks.append(model_checkpoint)
                

                # Train the model for this split
                model.fit(
                    X_train_split,
                    y_train_split,
                    validation_data=(X_val_split, y_val_split),
                    callbacks=copied_callbacks,
                    **kwargs
                    )
                val_losses.append(model.evaluate(X_val_split, y_val_split))
            self.oracle.update_trial(trial.trial_id, {"val_loss": np.mean(val_losses)})
            # self.save_model(trial.trial_id, model)

    # def save_model(self, trial_id, model):
    #     # Save the model for the given trial
    #     fname = os.path.join(self.get_trial_dir(trial_id), "model.keras")
    #     model.save(fname)