import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, cohen_kappa_score

class CropsAveragingMetrics(Callback):
    ''' Compute evaluation metrics after averaging the predictions of the crops of a same trial. '''
    def __init__(self, validation_data, trial2crops, n_trials, savename='best_model.h5', verbose=0):
        super(Callback, self).__init__()
        self.X_valid, self.y_valid = validation_data
        self.trial2crops= trial2crops
        self.n_trials = n_trials
        self.best = -np.Inf
        self.best_epoch = -1
        self.savename = savename
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        y_pred_crops = np.argmax(self.model.predict(self.X_valid, verbose=0), axis=1) # Proba array to int
        y_pred = np.empty(self.n_trials)
        for trial_idx in range(self.n_trials):
            labels, cnts = np.unique(y_pred_crops[self.trial2crops[trial_idx]], return_counts=True)
            y_pred[trial_idx] = labels[np.argmax(cnts)]

        val_acc = accuracy_score(self.y_valid, y_pred)
        val_kappa = cohen_kappa_score(self.y_valid, y_pred)

        # Save best model
        if np.greater(val_acc, self.best):
            print('Saving model - epoch: {:d} - : val acc: {:.2f}'.format(epoch+1, val_acc))
            self.model.save_weights(self.savename, overwrite=True)
            self.best_epoch = epoch+1
            self.best = val_acc
    
        if self.verbose:
            print("Crop averaging evaluation - epoch: {:d} - : val acc: {:.2f} - kappa: {:.2f}".format(epoch+1, 
                                                                                                   val_acc,
                                                                                                   val_kappa))

class EarlyStoppingByLossVal(Callback):
    ''' Early stopping by validation loss is using during the second part of the model training (train + validation set).
        We usually stop the training once the validation loss reaches the training loss of first part training (only train set).'''
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires {} available!".format(self.monitor), RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Early stopping by loss val - epoch {}".format(epoch))
            self.model.stop_training = True
            
## Source: https://github.com/comet-ml/comet-examples/blob/master/notebooks/keras.ipynb
class VizCallback(Callback):
    ''' Extract activation of a given model's layer as a 2D image.'''
    def __init__(self, model, tensor, filename, experiment=None):
        self.mymodel = model
        self.tensor = tensor
        self.filename = filename
        self.experiment = experiment

    def on_epoch_end(self, epoch, logs=None):
        if epoch%50==0:
            filename = self.filename % (epoch,)
            log_image(self.mymodel, self.tensor, filename,
                      self.experiment)

    def array_to_image(array, scale):
        from keras.preprocessing import image
        sh = array.shape
        img = image.array_to_img(array.reshape([1, sh[0]*sh[1], sh[2]]))
        x, y = img.size
        img = img.resize((x * scale, y * scale))
        return img

    def log_image(model, tensor, filename, experiment):
        output = model.predict(tensor)
        img = array_to_image(output[0], 1)
        img.save(filename)
        if experiment:
            experiment.log_image(filename)

def build_viz_model(model, visualize_layer):
    viz_model = Model(inputs=[model.input],
                      outputs=[model.layers[visualize_layer].output])
    return viz_model