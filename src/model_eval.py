import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
plt.style.use('ggplot')
import datetime
import os
import re

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


def roc_curve_many(json_folder, filename, title):
    '''
    does the same as roc curve but for many items
    '''

    if json_folder[-1] != '/':
        json_folder += '/'
    bdir = json_folder
    colors  = itertools.cycle(['red', 'cyan', 'green', 'orange', 'violet', 'blue', 
                                        'pink', 'yellow', 'purple', 'black'])
    fig, ax = plt.subplots(1,1,figsize=(15,15), facecolor='white')
    for fname in os.listdir(bdir):
        tree = pd.read_json(bdir+fname)
        if 'rf' in fname:
            mtype = 'rf'
        elif 'gbt' in fname or 'gbr' in fname:
            mtype = 'gbr'
        else:
            mtype = 'nb'
        label = re.sub('[^0-9]', '', fname) + mtype
        if tree.columns[0] == 'label':
            proba = tree.columns[1]
            lab = tree.columns[0]
        else:
            proba = tree.columns[0]
            lab = tree.columns[1]
        vals = []
        print fname
        print lab
        print label
        print '--------'
        npr_last = 0
        for n in np.linspace(.01,.99,200):
            tdic = {}
            treepos = tree[tree[lab] == 1]
            treepos['pred'] = treepos[proba].apply(lambda x: 1 if x >= n else 0)
            vals.append(tdic)
            tdic['tpr'] = float(treepos[treepos['pred'] == treepos[lab]].shape[0])/treepos.shape[0]
            treeneg = tree[tree[lab] == 0]
            treeneg['pred'] = treeneg[proba].apply(lambda x: 1 if x >= n else 0)
            tdic['npr'] = float(treeneg[treeneg['pred'] != treeneg[lab]].shape[0])/treeneg.shape[0]
            tdic['width'] = tdic['npr'] - npr_last
            npr_last = tdic['npr']
            vals.append(tdic)
        roc_values = pd.DataFrame(vals)
        ax.plot(roc_values['npr'], roc_values['tpr'], color =colors.next(),label= \
                    "{}".format(label, sum([row['width'] * row['tpr'] for row in vals])))
    ax.plot(np.linspace(0, 1, 20), np.linspace(0, 1, 20), color='black')
    ax.legend(loc='lower right', fontsize=24)
    ax.set_xlabel('False Positive Rate', fontsize=24)
    ax.set_ylabel('True Positive Rate', fontsize=24)
    ax.set_title(title, fontsize=24)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    fig.savefig('../images/{}'.format(filename))