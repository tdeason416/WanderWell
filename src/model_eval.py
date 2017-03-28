import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
plt.style.use('ggplot')
import datetime
import os

def roc_curve(json_location, save_file=True):
    '''
    Generates ROC curve given a json object with keys 'probability' and 'label'
    --------
    Parameters
    json_location = location of two column dataset
    save_file = place to save file 
    --------
    Returns
    saves file at save file location, or at ../images/roc_cirve(ordinal_time)
    '''
    if save_file:
        save_file = 'images/roc_curve-{}'.format(datetime.datetime.now().toordinal())
    model = pd.read_json(json_location)
    model.probability = model.probability.apply(lambda x: x['array'][1])
    roc_values = []
    for thres in np.linspace(.01,.99,300):
        model['pred'] = model['probability'].apply(lambda x: 1 if x > thres else 0)
        rowdict = {}
        is_pos = model[model['label'] == 1]
        is_neg = model[model['label'] == 0]
        rowdict['tpr'] = float(is_pos[is_pos['pred'] == 1].shape[0])/is_pos.shape[0]
        rowdict['fpr'] = float(is_neg[is_neg['pred'] == 1].shape[0])/is_neg.shape[0]
        roc_values.append(rowdict)
    roc_values = pd.DataFrame(roc_values)
    fig, ax = plt.subplots(1,1,figsize=(13,13))
    ax.scatter(roc_values['fpr'], roc_values['tpr'], color ='red')
    ax.plot(np.linspace(0,1,20), np.linspace(0,1,20), color='black')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.savefig('../images/roc_curve-{}'.format(datetime.datetime.now().toordinal()))


def roc_curve_many(json_folder, save_file=True):
    '''
    does the same as roc curve but for many items
    '''
    if json_folder[-1] != '/':
        json_folder += '/'
    colors  = itertools.cycle(['red','cyan', 'green', 'orange', 'violet', 'blue', 
                                'pink', 'yellow', 'purple', 'black'])
    if save_file:
        save_file = '../images/roc_curve-{}'.format(datetime.datetime.now().toordinal())
    list_of_pd_frames = os.listdir(json_folder)
    fig, ax = plt.subplots(1,1,figsize=(13,13))
    for frame in list_of_pd_frames:
        model_probs = pd.read_json(json_folder+frame)
        print model_probs.columns
        probability = model_probs.columns[0]
        label = model_probs.columns[1]
        model_probs[probability] = model_probs[probability].apply(lambda x: x['array'][1])
        roc_values = []
        for thres in np.linspace(.01, .99, 500):
            model_probs['pred'] = model_probs[probability].apply(lambda x: 1 if x > thres else 0)
            rowdict = {}
            is_pos = model_probs[model_probs[label] == 1]
            is_neg = model_probs[model_probs[label] == 0]
            rowdict['tpr'] = float(is_pos[is_pos['pred'] == 1].shape[0])/is_pos.shape[0]
            rowdict['fpr'] = float(is_neg[is_neg['pred'] == 1].shape[0])/is_neg.shape[0]
            roc_values.append(rowdict)
        roc_values = pd.DataFrame(roc_values)
        ax.scatter(roc_values['fpr'], roc_values['tpr'], color =colors.next(),label=frame)
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), color='black')
    ax.legend()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.savefig(save_file)
