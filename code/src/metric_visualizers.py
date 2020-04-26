import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

def plot_full_barchart(test_score, n_pilots, title="Test accuracy - BCIC IV 2a", fig=None, ax_idx=0):
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()

    w = 0.40

    average_kappa = np.mean(test_score['kappa'])
    average_acc = np.mean(test_score['accuracy'])

    # Accuracy score
    ax.bar(x=np.arange(1 + 1.25*w, n_pilots + 1), height=test_score['accuracy'], width=-w, 
            tick_label=range(1, n_pilots+1), alpha=0.9, align='edge')
    ax.plot([0,n_pilots+1], 2*[average_acc], '--b')

    ax.set_xlim([1, n_pilots + 1])
    ax.set_ylim([0,1])
    ax.grid(True, axis='y')
    ax.set_xlabel('Pilot')
    ax.set_ylabel('Accuracy / Kappa score')

    # Kappa score
    ax.bar(x=np.arange(1 + 1.25*w, n_pilots + 1), height=test_score['kappa'], width=w,
            color='orange', alpha=0.9, align='edge')
    ax.plot([0, n_pilots + 1], 2*[average_kappa], '--r')
    ax.set_title(title)
    ax.legend(['Average accuracy', 'Average kappa', 'Accuracy score', 'Kappa score'])

    fig.text(1.01, average_acc-0.01, "{:.2f}".format(average_acc), fontsize='large', color='b', transform = ax.transAxes)
    fig.text(1.01, average_kappa-0.01, "{:.2f}".format(average_kappa), fontsize='large', color='r', transform = ax.transAxes)

    return fig

def plot_accuracy_barchart(test_score, model_name, n_pilots):
    plt.figure(figsize=(10,6))
    plt.plot([0,n_pilots+1], 2*[np.mean(test_score['accuracy'])], '--r')
    plt.legend(['Average accuracy'])
    plt.bar(x=range(1, n_pilots+1), height=test_score['accuracy'], tick_label=range(1, n_pilots+1))

    plt.title("Accuracy on test dataset for each pilot - {}".format(model_name))
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.xlabel("Pilot")
    plt.xlim([0, n_pilots+1])
    plt.grid(True, axis='y')
    return plt

def plot_loss(loss_history, pilot_idx): 
    plt.figure()
    plt.plot(loss_history.history['loss'])
    plt.plot(loss_history.history['val_loss'])
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Training & validation loss - Pilot {}".format(pilot_idx))
    plt.legend(['Training loss','Validation loss']);
    return plt

def compute_cm(y_true, y_pred, classes, normalize=False, title=None, fig=None, ax_idx=0):
    ''' TODO '''
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()
    
    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_xticks(ticks=np.arange(cm.shape[1]))
    ax.set_yticks(ticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(labels=classes)
    ax.set_yticklabels(labels=classes)
    ax.tick_params(labelrotation=45)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


def plot_cm(cms, pilot_idx, title='Confusion matrix', class_names=['Left','Right','Foot','Tongue'], fig=None, ax_idx=0):
    '''  '''
    
    np.set_printoptions(precision=2)

    fig = compute_cm(y_true=cms['y_true{}'.format(pilot_idx)],
                     y_pred=cms['y_pred{}'.format(pilot_idx)],
                     classes=class_names,
                     title=title, normalize=False,
                     fig=fig, ax_idx=ax_idx);
    return fig


def compute_online_metrics(model, X_test, y_test, crop_samples=125, stride=1, dl_input=False):
    ''' Compute accuracy and kappa score for each consecutive time window (with stride).
        Input:
            - trained_model:
            - X_test, y_test: Test data comprising EEG trials and labels (not seen by model during training).
            - crop_samples: Temporal window size of the data fed to the model.
            - stride: Number of samples between two consecutive time windows.
        Output:
            - scores: Dictionnary comprising two arrays of scores for each time window.
    '''
    scores = {'accuracy' : [], 'kappa' : []}
    inference_times = []
    t = crop_samples
    t_max = X_test.shape[-1] + crop_samples
    
    # Padding input signal
    if dl_input:
        X_test = np.pad(X_test, ((0,0), (0,0), (0,0), (crop_samples,0)), mode='minimum')
    else:
        X_test = np.pad(X_test, ((0,0), (0,0), (crop_samples,0)), mode='minimum')
    
    while t < t_max:
        # Extract query temporal window (causal -> look at the past signal)
        X_query = X_test[..., t - crop_samples : t]
    
        # Predict labels using the trained model
        now = time.time()
        y_pred = np.argmax(model.predict(X_query), axis=1) if dl_input else model.predict(X_query)
        inference_times.append(time.time() - now)
        
        # Compute scores
        acc = accuracy_score(y_test, y_pred)
        kap = cohen_kappa_score(y_test, y_pred)
        
        # Save average scores
        scores['accuracy'].append(acc)
        scores['kappa'].append(kap)
        
        # Update next time window interval
        t += stride

    scores['accuracy'] = np.array(scores['accuracy'], dtype=float)
    scores['kappa'] = np.array(scores['kappa'], dtype=float)
    return scores, np.mean(inference_times)


def plot_online_metrics(scores, len_tot=1500, fs=250, stride=10, n_folds=1, start_mi=500, title='Online metrics over full trial', fig=None, ax_idx=0):
    if fig:
        ax = fig.get_axes()[ax_idx]
    else:
        fig, ax = plt.subplots()

    ax.axvline(start_mi//stride, ls='--', c='gray')
    ax.plot(scores['accuracy']/n_folds)
    ax.plot(scores['kappa']/n_folds)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Performance')
    tick_locs = np.arange(0, (len_tot + fs)//stride, fs//stride)
    tick_lbls = np.arange(len_tot//fs + 2)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels(tick_lbls)
    ax.set_ylim([0,1])
    ax.legend(['Start of MI', 'Accuracy score', 'Kappa score'])
    ax.set_title(title)
    return fig    
