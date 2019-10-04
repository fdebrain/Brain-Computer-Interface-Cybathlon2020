import comet_ml
from comet_ml import Experiment, Optimizer

import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.utils import shuffle

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.set_random_seed(0)

import keras
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
import keras.backend as K
K.set_image_data_format("channels_first")

import sys
sys.path.append('..')
from data_loading_functions.data_loader import EEGDataset
from feature_extraction_functions.convnets import EEGNet, ShallowConvNet, DeepConvNet, DevConvNet
from visualization_functions.metric_visualizers import plot_full_barchart, plot_cm, plot_loss
from visualization_functions.signal_visualizers import tsne_plot
from utils.keras_utils import EarlyStoppingByLossVal, VizCallback

class Pipeline_DL:
    def __init__(self, dataloader_params, training_params, weights_savename_prefix, comet_optimize=False, verbose_checkpoints=False, log_pilots=False, comet_log=False, tsne_layer_idxs=[], load_test=True):
        ''' TODO '''
        # Exp parameters
        self.dataloader_params = dataloader_params
        self.training_params = training_params
        
        # Training parameters
        self.n_pilots = training_params['n_pilots']
        self.n_classes = training_params['n_classes']
        self.loss = 'binary_crossentropy' if self.n_classes==2 else 'categorical_crossentropy'
        self.pretraining = training_params['pretraining']
        self.lr = training_params['lr']
        self.train_after_es = training_params['train_after_es']
        self.max_after_es_epochs = training_params['max_after_es_epochs']
        self.model_name = training_params['model_name']
        self.n_epochs = training_params['n_epochs']
        self.batch_size = training_params['batch_size']
        self.monitor_val = training_params['monitor_val']
        self.patience = training_params['patience']
        self.opt = training_params['opt']
        self.l2_reg = training_params['l2_reg']
        self.initializer = training_params['initializer']
        self.bn = training_params['bn']
        self.dropout = training_params['dropout']
        self.dropoutRate = training_params['dropoutRate']
        self.activation = training_params['activation']
        
        # Logging / Comet parameters
        self.weights_savename_prefix = weights_savename_prefix
        self.verbose_checkpoints = verbose_checkpoints
        self.comet_optimize = comet_optimize
        self.log_pilots = log_pilots
        
        # Scoring datastructures
        self.train_score = { 'accuracy' : [None]*self.n_pilots, 'kappa' : [None]*self.n_pilots }
        self.test_score = { 'accuracy' : [None]*self.n_pilots, 'kappa' : [None]*self.n_pilots }
        self.histories = []
        self.histories_es = []
        self.conf_matrices = {} 
        
        # Reproducibility (small variations on GPU though, <0.01)
        np.random.seed(0)
        tf.set_random_seed(0)
        
        self.comet_log = comet_log
        self.tsne_layer_idxs = tsne_layer_idxs
        
    def load_data(self, pilot_idx, valid_ratio):
        ''' Load data of given pilot and split into train, validation and test sets. 
            The validation set size can be adjusted (usually 0.2).
            Inputs:
                - pilot_idx: (int)
                - valid_ratio: (float)
            Outputs:
                - dataset: X_train, y_train, X_valid, y_valid, X_test, y_test
        '''
        
        X_train, y_train, X_valid, y_valid, X_test, y_test = EEGDataset(pilot_idx, **self.dataloader_params, valid_ratio=valid_ratio).load_dataset()

        # Reshape inputs
        self.n_trials, self.n_channels, self.n_samples = X_train.shape
        X_train = np.reshape(X_train, (self.n_trials, 1, self.n_channels, self.n_samples))

        n_trials, n_channels, n_samples = X_valid.shape
        X_valid = np.reshape(X_valid, (n_trials, 1, n_channels, n_samples))

        n_trials, n_channels, n_samples = X_test.shape
        X_test = np.reshape(X_test, (n_trials, 1, n_channels, n_samples))
        
        return X_train, y_train, X_valid, y_valid, X_test, y_test
        
        
    def build_model(self, load_weights=False, weights_filename='load.h5'):
        ''' Initialize model graph & load weights if necessary.
        Inputs:
            - load_weights: (boolean).
            - weights_filename: Filename were weights to load are stored (string).
        Output:
            - model: Keras model with given architecture and initialized/loaded weights. 
        '''
        
        # Create model
        model_name = self.model_name
        n_classes, n_channels, n_samples = self.n_classes, self.n_channels, self.n_samples
        l2_reg = self.l2_reg
        activation = self.activation
        initializer = self.initializer
        bn = self.bn
        dropout = self.dropout
        dropoutRate = self.dropoutRate
        
        model = None
        if model_name == 'Shallow':
            model = ShallowConvNet(n_classes, n_channels, n_samples, l2_reg=l2_reg)
        elif model_name == 'Deep':
            model = DeepConvNet(n_classes, n_channels, n_samples, l2_reg=l2_reg)
        elif model_name == 'EEGNet':
            model = EEGNet(n_classes, n_channels, n_samples)
        elif model_name == 'perso':
            model = devConvNet(n_classes, n_channels, n_samples, l2_reg=l2_reg, initializer=initializer, bn=bn, dropout=dropout, dropoutRate=dropoutRate, activation=activation)
        else:
            raise Exception("Please choose a model among {Shallow, Deep, EEGNet}")
            
        # Load weights
        if load_weights:
            print('Loading weights from ' + weights_filename)
            model.load_weights(weights_filename)
        else:
            print('Training model from scratch')
        return model
        
    def pretrain(self, save_filename, patience):
        ''' Train & validate the model on full training dataset (all pilots) from scratch. 
        
            Inputs:
                - save_filename: (string: 'weights_save_filename.h5').
                - patience: Number of epochs without improvement before training stops (int).
            Output:
                - hist: Contains loss, acc, val_loss, val_acc metrics over epochs. (dictionnary) 
                    
        '''
        X_train, y_train, X_valid, y_valid, X_test, y_test = [], [], [], [], [], []
        
        if self.log_pilots:
            experiment = Experiment(api_key='cSZq9kuH2I87ezvm2dEWTx6op', project_name='All pilots', log_code=False, auto_param_logging=False)
        
        for pilot_idx in range(1, self.n_pilots + 1):
            X_train_pilot, y_train_pilot, X_valid_pilot, y_valid_pilot, X_test_pilot, y_test_pilot = self.load_data(pilot_idx, valid_ratio=0.2)
            
            # Stacking training/validation datasets of each pilot into a big dataset
            if len(X_train)==0 and len(X_valid)==0:
                X_train = X_train_pilot
                y_train = y_train_pilot
                X_valid = X_valid_pilot
                y_valid = y_valid_pilot
                X_test = X_test_pilot
                y_test = y_test_pilot
            else:        
                X_train = np.concatenate([X_train, X_train_pilot], axis=0)
                y_train = np.concatenate([y_train, y_train_pilot], axis=0)
                X_valid = np.concatenate([X_valid, X_valid_pilot], axis=0)
                y_valid = np.concatenate([y_valid, y_valid_pilot], axis=0)
                X_test = np.concatenate([X_test, X_test_pilot], axis=0)
                y_test = np.concatenate([y_test, y_test_pilot], axis=0)
                
        # Shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)
        X_test, y_test = shuffle(X_test, y_test, random_state=0)
        
        # Build model
        self.model = self.build_model(load_weights=False)
        
        # Train on all pilots
        hist, _ = self.train(X_train, y_train, X_valid, y_valid, save_filename, patience)
        
        # Get some metrics
        score = self.test('all', save_filename, X_train, y_train, X_test, y_test, False)
        
        if self.log_pilots:
            experiment.log_metrics({'Test accuracy' : score})
            experiment.end()
        
        return hist

        
    def train(self, X_train, y_train, X_valid, y_valid, save_filename, patience, experiment=None, after_es=False):
        ''' Train & validate model on given data. Save the best model weights and return metrics and best epoch.
        
        Inputs:
            - X_train, y_train, X_valid, y_valid: Training & validation data
            - save_filename: (string: 'weights_save_filename.h5').
            - patience: Number of epochs without improvement before training stops (int).
        Outputs:
            - hist: Contains loss, acc, val_loss, val_acc metrics over epochs. (dictionnary) 
            - best_epoch: Epoch with the lowest val_loss/val_acc. (int)
        '''

        # Define callbacks
        es = EarlyStopping(patience=patience, monitor=self.monitor_val, verbose=1)
        mc = ModelCheckpoint(save_filename, self.monitor_val, self.verbose_checkpoints, save_best_only=True, save_weights_only=True) 
        
        #viz_model1 = build_viz_model(self.model, 1)
        #viz_model2 = build_viz_model(self.model, 2)
        #viz1 = VizCallback(viz_model1, X_train[0:1], "hidden1-epoch-%s.gif", experiment)
        #viz2 = VizCallback(viz_model2, X_train[0:1], "hidden2-epoch-%s.gif", experiment)
        
        # Initialize optimizer, loss & metrics
        self.model.compile(loss=self.loss, optimizer=self.opt(lr=self.lr), metrics=['accuracy'])
        
        # Train model & get metrics evolution w.r.t epochs
        hist = self.model.fit(x=X_train, y=to_categorical(y_train), batch_size=self.batch_size, epochs=self.n_epochs, shuffle=True, verbose=0, 
                              validation_data=(X_valid, to_categorical(y_valid)), callbacks=[es, mc])
        
        best_epoch = np.argmin(hist.history['val_loss']) if self.monitor_val=='val_loss' else np.argmax(hist.history['val_acc'])
        print('Best epoch: {} - Valid loss: {:.2} - Valid acc: {:.2}'.format(best_epoch + 1, 
                                                                             hist.history['val_loss'][best_epoch],
                                                                             hist.history['val_acc'][best_epoch]))
        
        return hist, best_epoch
        
        
    def extra_train(self, pilot_idx, X_valid, y_valid, weights_filename, loss_to_reach, max_epochs_es):
        ''' Extra training after early stopping using validation data among training data. Training stops once validation loss reach a threshold. 
            
            Inputs:
                - pilot_idx: Dataset index to load (int).
                - X_valid, y_valid: Validation data.
                - weights_filename: Weights filename to load/save (string).
                - loss_to_reach: Validation loss threshold, stop training once reached (float).
                - max_epochs_es: Stop training if n_epochs > max_epochs_es (int).
            Output:
                - hist: Contains loss, acc, val_loss, val_acc metrics over epochs. (dictionnary) 
        '''
        print('\t Second phase training - Loss to reach: {} - Max epochs: {}'.format(loss_to_reach, max_epochs_es))
        
        # Build model & load weights of previous training step
        self.model = self.build_model(load_weights=True, weights_filename=weights_filename)
        
        # Loading full preprocessed training set
        X_train, y_train, _, _, _, _ = self.load_data(pilot_idx, valid_ratio=0.)

        # Early stop when val_loss is below previous training loss
        eslv = EarlyStoppingByLossVal(monitor='val_loss', value=loss_to_reach, verbose=1) 

        # Training on full training set
        self.model.compile(loss=self.loss, optimizer=self.opt(self.lr), metrics=['accuracy'])
        hist = self.model.fit(x=X_train, y=to_categorical(y_train), batch_size=self.batch_size, epochs=max_epochs_es,
                              shuffle=True, verbose=0, validation_data=(X_valid, to_categorical(y_valid)), callbacks=[eslv])
        
        # Overwrite old model weights
        self.model.save_weights(weights_filename)
        
        return hist
        
        
    def test(self, pilot_idx, weights_filename, X_train, y_train, X_test, y_test, after_es):
        ''' Load best model and compute scores.'''
        
        # Build model & load weights of best model
        self.model = self.build_model(load_weights=True, weights_filename=weights_filename)
        
        # Compute training scores
        y_pred_train = self.model.predict(X_train)
        y_pred_train = np.argmax(y_pred_train, axis=1)
        train_acc = accuracy_score(y_train, y_pred_train)
        train_kappa = cohen_kappa_score(y_train, y_pred_train)
        
        # Compute testing scores
        y_pred_test = self.model.predict(X_test)
        y_pred_test = np.argmax(y_pred_test, axis=1)          
        test_acc = accuracy_score(y_test, y_pred_test)
        test_kappa = cohen_kappa_score(y_test, y_pred_test)
    
        # Saving scores
        if pilot_idx in range(self.n_pilots + 1):
            self.train_score['accuracy'][pilot_idx-1] = train_acc
            self.train_score['kappa'][pilot_idx-1] = train_kappa
            self.test_score['accuracy'][pilot_idx-1] = test_acc
            self.test_score['kappa'][pilot_idx-1] = test_kappa
            self.conf_matrices['y_pred{}'.format(pilot_idx)] = y_pred_test
            self.conf_matrices['y_true{}'.format(pilot_idx)] = y_test
            
        print('Pilot {} - Train: {:.3f}/{:.3f} - Test: {:.3f}/{:.3f}. ({} ES+)\n'.format(pilot_idx, train_acc, train_kappa, test_acc, test_kappa,
                                                                                         'after' if after_es else 'before'))
        return test_acc

    
    def log(self, experiment=None):
        ''' Export all logs in the Comet.ml environment.
            See https://www.comet.ml/ for more details
        '''
        
        # Initialize Comet.ml experience (naming, tags) for automatic logging
        project_name = 'Optimization' if self.comet_optimize else 'Summary'
        experiment_name = '{} - {} '.format(self.model_name, str(self.batch_size)) + ('ES+' if self.train_after_es else '')
        experiment_tags = [ self.model_name, self.monitor_val ] + (['ES+'] if self.train_after_es else []) +  (['Pre-train'] if self.pretraining else [])
        
        if experiment == None:
            experiment = Experiment(api_key='cSZq9kuH2I87ezvm2dEWTx6op', project_name=project_name, log_code=False, auto_param_logging=False, auto_metric_logging=False)
        experiment.set_name(experiment_name)
        experiment.add_tags(experiment_tags)
        
        # Export hyperparameters
        experiment.log_parameters(self.dataloader_params)
        experiment.log_parameters(self.training_params)   
        
        # Export metrics values
        experiment.log_metrics({'Average accuracy' : np.mean(self.test_score['accuracy']), 'Std accuracy' : np.std(self.test_score['accuracy'])})
        
        # Export metrics graphs for each pilot (accuracy, loss, confusion matrix)
        [ experiment.log_figure(figure_name='Confusion matrix {}'.format(pilot_idx), figure=plot_cm(self.conf_matrices, pilot_idx)) for pilot_idx in range(1,self.n_pilots+1)]
        [ experiment.log_figure(figure_name='Loss pilot {}'.format(pilot_idx), figure=plot_loss(self.histories[pilot_idx-1], pilot_idx)) for pilot_idx in range(1,self.n_pilots+1)]
        
        fig, ax = plt.subplots(figsize=(10,6))
        plot_full_barchart(self.test_score, n_pilots=self.n_pilots, title=' {} ConvNet model'.format(self.model_name), fig=fig)
        experiment.log_figure(figure_name='Accuracy barchart', figure=fig)
        
        if self.train_after_es:
            [ experiment.log_figure(figure_name='Loss pilot {} (ES+)'.format(pilot_idx), figure=plot_loss(self.histories_es[pilot_idx-1], pilot_idx)) for pilot_idx in range(1,self.n_pilots+1)]
        
        # Export model weights for each pilot
        [ experiment.log_asset('{}{}.h5'.format(self.weights_savename_prefix, pilot_idx)) for pilot_idx in range(1,self.n_pilots+1)]
        experiment.end()
        
        
    def run_exp(self):
        ''' Run the experiment on given number of pilots. '''
        
        # Pre-train
        pretrain_weights_filename  = '{}{}.h5'.format(self.weights_savename_prefix, '_all')
        if self.pretraining:
            print('Pretraining...')                                                                                                                 
            _ = self.pretrain(pretrain_weights_filename, self.patience)
                
        for pilot_idx in range(1, self.n_pilots + 1):
            if self.log_pilots:
                experiment = Experiment(api_key='cSZq9kuH2I87ezvm2dEWTx6op', project_name='pilot{}'.format(pilot_idx), log_code=False, auto_param_logging=False)
            else:
                experiment = None
            
            # Load pilot data
            X_train, y_train, X_valid, y_valid, X_test, y_test = self.load_data(pilot_idx, valid_ratio=0.2)
            
            # Construct model & load pre-trained weights if available
            weights_filename  = '{}{}.h5'.format(self.weights_savename_prefix, pilot_idx)
            self.model = self.build_model(load_weights=True if self.pretraining else False,
                                          weights_filename=pretrain_weights_filename)
            
            # Train
            print('First phase training - Pilot {}'.format(pilot_idx))
            hist, best_epoch = self.train(X_train, y_train, X_valid, y_valid,
                                          save_filename='{}{}.h5'.format(self.weights_savename_prefix, pilot_idx),
                                          patience=self.patience, experiment=experiment)
            self.histories.append(hist)
             
            # Test (before extra training)
            self.test(pilot_idx, weights_filename, X_train, y_train, X_test, y_test, False)
                                
            # Extra-train
            if self.train_after_es:
                hist_es = self.extra_train(pilot_idx, X_valid, y_valid, weights_filename, hist.history['loss'][best_epoch], self.max_after_es_epochs) # New
                self.histories_es.append(hist_es)
                
                # Test (after extra training)
                self.test(pilot_idx, weights_filename, X_train, y_train, X_test, y_test, True)
            
            if self.log_pilots:
                experiment.log_metrics({'Test accuracy' : self.test_score['accuracy'][-1]})

                # Get t-SNE from intermediary outputs
                layer_idxs = self.tsne_layer_idxs
                get_output_functions = [ K.function([self.model.layers[0].input], [self.model.layers[idx].output]) for idx in layer_idxs]
                
                # Training dataset
                layer_outputs = [ get_output([X_train])[0] for get_output in get_output_functions ]
                [ experiment.log_figure(figure_name='tsne_raw_train{}_layer{}'.format(pilot_idx, layer_idxs[idx]),
                                        figure=tsne_plot(layer_outputs[idx], y_train, 20, title="t-SNE - Pilot {} - Layer {} (train)".format(pilot_idx, layer_idxs[idx]))) 
                                        for idx in range(len(layer_idxs)) ]
                                        
                # Testing dataset
                layer_outputs = [ get_output([X_test])[0] for get_output in get_output_functions ]
                [ experiment.log_figure(figure_name='tsne_raw_test{}_layer{}'.format(pilot_idx, layer_idxs[idx]),
                                        figure=tsne_plot(layer_outputs[idx], y_test, 20, title="t-SNE - Pilot {} - Layer {} (test)".format(pilot_idx, layer_idxs[idx]))) 
                                        for idx in range(len(layer_idxs)) ]
                plt.close('all')
                experiment.end()
            
        # Export logs to Comet.ml
        if self.comet_log:
            self.log()
        
        return self.test_score
