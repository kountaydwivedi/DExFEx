from IPython.core.display import display, HTML
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

import time
import torch
import torch.nn.functional as F

import os
import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix



##===================================================================================##
#
# Set all seeds
#
##===================================================================================##

def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




##===================================================================================##
#
# Dataset preprocessing
#
##===================================================================================##

def prepare_data_for_ranksum_test(df_luad, df_lusu):
    
    ## set index as 'sample'
    df_luad.set_index('sample', inplace=True)
    df_lusu.set_index('sample', inplace=True)
    
    ## remove all the rows that have only 0s as entries
    df_luad = df_luad[(df_luad.T != 0).any()]
    df_lusu = df_lusu[(df_lusu.T != 0).any()]
    df_luad.reset_index(inplace=True)
    df_lusu.reset_index(inplace=True)
    text = HTML("Rows containing all 0s removed.")
    display(text)
    
    ## save the genesymbols as a list
    ## transpose the datasets and make GeneSymbols as  column headers
    ## remove the 1st  row (i.e., the genesymbols )
    ## make "columns" as columns of the dataframes
    columns = list(df_luad["sample"])
    text = HTML("Gene names saved in list 'columns'.")
    display(text)
    df_luad = df_luad.T
    df_lusu = df_lusu.T
    text = HTML("Dataset transposed (rows to column and columns to rows).")
    display(text)
    df_luad = df_luad.iloc[1:]
    df_lusu = df_lusu.iloc[1:]
    df_luad.columns = columns
    df_lusu.columns = columns
    text = HTML("Gene name column removed.")
    display(text)
    
#     label_1 = np.ones(len(df_luad), dtype=int)
#     label_0 = np.zeros(len(df_lusu), dtype=int)
    
    ## insert label columns
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.insert.html
    df_luad.insert(0, "label", 'LUAD')
    df_lusu.insert(0, "label", 'LUSC')
    text = HTML("Samples' ground truth values (LUAD, LUSC) inserted.")
    display(text)
    
    ## merge both dataframes
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    df_final = pd.concat([df_luad, df_lusu])
    text = HTML("DF_LUAD and DF_LUSU dataframes combined to one.")
    display(text)
    
    ## remove columns (genes) that have same value across all rows
    nunique = df_final.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df_final = df_final.drop(cols_to_drop, axis=1)
    text = HTML("Columns containing same value across all samples removed.")
    display(text)
    
    text = HTML("Dataframe with labels ====>")
    display(text)
    display(df_final)
    
    return df_final
    
    
def dataset_preprocess(df_luad, df_lusu):
    
    
    ## set index as 'sample'
    df_luad.set_index('sample', inplace=True)
    df_lusu.set_index('sample', inplace=True)
    
    ## remove all the rows that have only 0s as entries
    df_luad = df_luad[(df_luad.T != 0).any()]
    df_lusu = df_lusu[(df_lusu.T != 0).any()]
    df_luad.reset_index(inplace=True)
    df_lusu.reset_index(inplace=True)
    text = HTML("Rows containing all 0s removed.")
    display(text)
    
    ## save the genesymbols as a list
    ## transpose the datasets and make GeneSymbols as  column headers
    ## remove the 1st  row (i.e., the genesymbols )
    ## make "columns" as columns of the dataframes
    columns = list(df_luad["sample"])
    text = HTML("Gene names saved in list 'columns'.")
    display(text)
    df_luad = df_luad.T
    df_lusu = df_lusu.T
    text = HTML("Dataset transposed (rows to column and columns to rows).")
    display(text)
    df_luad = df_luad.iloc[1:]
    df_lusu = df_lusu.iloc[1:]
    df_luad.columns = columns
    df_lusu.columns = columns
    text = HTML("Gene name column removed.")
    display(text)
    
    label_1 = np.ones(len(df_luad), dtype=int)
    label_0 = np.zeros(len(df_lusu), dtype=int)
    
    ## insert label columns
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.insert.html
    df_luad.insert(0, "label", label_1)
    df_lusu.insert(0, "label", label_0)
    text = HTML("Samples' ground truth values (0, 1) inserted.")
    display(text)
    
    ## merge both dataframes
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
    df_final = pd.concat([df_luad, df_lusu])
    text = HTML("DF_LUAD and DF_LUSU dataframes combined to one.")
    display(text)
    
    ## remove columns (genes) that have same value across all rows
    nunique = df_final.nunique()
    cols_to_drop = nunique[nunique == 1].index
    df_final = df_final.drop(cols_to_drop, axis=1)
    text = HTML("Columns containing same value across all samples removed.")
    display(text)
    
    text = HTML("Dataframe with labels ====>")
    display(text)
    display(df_final)
    
    ## take the label column as a list
    # https://www.kite.com/python/answers/how-to-return-a-column-of-a-pandas-dataframe-as-a-list-in-python
    labels = df_final['label'].tolist()
    
    ## drop column 'label'
    df_final.drop(df_final.columns[0], inplace=True, axis=1)
    
    ## finally, set 'columns' to df_final.columns, and then reset index
    columns = df_final.columns
    df_final.reset_index(drop=True, inplace=True)
    
    text = HTML("Data preprocessing done. Returning processed dataframe as 'df_final', sample labels as 'labels', gene names as 'columns'.")
    display(text)
    text = HTML("Done.")
    display(text)
    return df_final, labels, columns
    





##===================================================================================##
#
# Training encoder
#
##===================================================================================##

def train_encoder(
    num_epochs,
    model,
    optimizer,
    device, 
    train_loader,
    valid_loader=None,
    loss_fn=None,
    logging_interval=100,
    skip_epoch_stats=False
):
        
    mean_loss_enc = 0
    log_dict = {'train_loss_per_batch': [], 'avg_loss':0., 'final_loss':0.}

    if loss_fn is None:
        loss_fn = F.mse_loss
        
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        mean_loss_batch = 0
        for batch_idx, (features, _) in enumerate(tqdm(train_loader)):
            features = features.to(device)
            features = features.float()
            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, features)
            optimizer.zero_grad()
            loss.backward()
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            mean_loss_batch+=loss.item()
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
            if not batch_idx % logging_interval:
                print('Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f'% (epoch+1, num_epochs, batch_idx,len(train_loader), loss))
        mean_loss_enc += (mean_loss_batch/len(train_loader))
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    print("Overall avg loss = %.2f" %(mean_loss_enc/num_epochs))
    log_dict['avg_loss']=mean_loss_enc/num_epochs
    log_dict['final_loss']=loss
    return log_dict



##===================================================================================##
#
# Compute pearson correlation 
#
##===================================================================================##
## https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
def pearson_correlation(model, data_loader, device):
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (features, _) in enumerate(data_loader):
            features = features.to(device)
            features = features.float()
            logits = model(features)
            true_weights = features.detach().cpu().numpy()  ## assume it true_X
            autoencoder_weights = logits.detach().cpu().numpy()  ## assume it pred_X
            
            ## now start computing pearson correlation
            true_X = true_weights
            pred_X = autoencoder_weights

            pearson_corr = ncc(true_X, pred_X)
            return pearson_corr
            
## https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))





##===================================================================================##
#
# Plot encoder loss
#
##===================================================================================##

def plot_training_loss(minibatch_losses, num_epochs, averaging_iterations=100, custom_label=''):

    iter_per_epoch = len(minibatch_losses) // num_epochs
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_losses)),(minibatch_losses), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_losses) < 1000:
        num_losses = len(minibatch_losses) // 2
    else:
        num_losses = 1000
    ax1.set_ylim([0, np.max(minibatch_losses[num_losses:])*1.5])
    ax1.plot(np.convolve(minibatch_losses,
                         np.ones(averaging_iterations,)/averaging_iterations,mode='valid'),label=f'Running Average{custom_label}')
    
    ax1.legend()
    # Set scond x-axis
    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))
    newpos = [e*iter_per_epoch for e in newlabel]
    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())
    
    plt.tight_layout()
    
    
    
    
    
##===================================================================================##
#
# Training Classifier
#
##===================================================================================##    

def train_classifier(
    num_epochs, 
    model,
    optimizer,
    scheduler,
    device,
    train_loader,
    valid_loader=None,
    loss_fn=None,
    logging_interval=100, 
    skip_epoch_stats=False
):
    
    log_dict = {'train_loss_per_batch':[], 'train_loss_per_epoch':[], 'train_acc':[], 'valid_acc':[]}

    if loss_fn is None:
        loss_fn = F.cross_entropy
        
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):


        # ----- model training ------

        cumulative_loss = 0
        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.float()
            features = features.to(device)
#             targets = targets.to(device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)
            
            # FORWARD AND BACK PROP
            logits = model(features)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            
            # UPDATE MODEL PARAMETERS
            optimizer.step()
            
            # LOGGING

            log_dict['train_loss_per_batch'].append(loss.item())
            cumulative_loss+=loss.item()
        
        # ----- model evaluation ------

        log_dict['train_loss_per_epoch'].append(np.mean(cumulative_loss))
        if skip_epoch_stats:
            
            model.eval()
            with torch.no_grad():  # save memory during inference
                train_acc = compute_accuracy(model, train_loader, device)
                log_dict['train_acc'].append(train_acc.item())
                print('Train acc = ', train_acc.item())
                
                valid_acc = compute_accuracy(model, valid_loader, device)
                log_dict['valid_acc'].append(valid_acc.item())
                print('Valid acc = ', valid_acc.item())
                print("***************")
                
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
    return log_dict






##===================================================================================##
#
# Compute Accuracy
#
##===================================================================================##

def compute_accuracy(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for i, (features, targets) in enumerate(data_loader):
            features = features.float()
            features = features.to(device)
            targets = torch.tensor(targets, dtype=torch.long, device=device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
        
    return (correct_pred.float()/num_examples * 100)






##===================================================================================##
#
# Plot accuracy
#
##===================================================================================##

def plot_accuracy(train_acc, train_loss, valid_acc=None):
    
    num_epochs = len(train_acc)
    # plt.figure(figsize=(10,10))
    plt.plot(np.arange(1, num_epochs+1), train_acc, label='Training')
    if valid_acc is not None:
        plt.plot(np.arange(1, num_epochs+1), valid_acc, label='Validation')
    plt.plot(np.arange(1, num_epochs+1),train_loss, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    

    
    
##===================================================================================##
#
# Read/Write Genes
#
##===================================================================================##

def read_essential_genes(PATH):
    essential_genes=[]
    gene_list_path = PATH+"essential_genes_set/essential_genes.kd"
    with open(gene_list_path, "r") as file:
        for gene in file:
            gene=gene.strip()
            essential_genes.append(gene)
    return essential_genes


def read_promo_genes(PATH):
    essential_genes=[]
    gene_list_path = PATH+"essential_genes_set/PROMO Analysis/df_intersect_analysis/promo_genes.txt"
    df_promo_genes = pd.read_csv(gene_list_path, sep='\t')
    promo_genes = list(df_promo_genes['Probeset Id'])
    return promo_genes


def write_seed_genes(PATH, seed, common_genes):
        gene_list_path = PATH+"project_summary_seed_wise/seed="+str(seed)+"/deg.kd"
        with open(gene_list_path, "w") as file:
            for gene in list(common_genes):
                file.write("%s\n" % gene)
        display(HTML("Common Genes written successfully !"))

        
        
        
        
        
##===================================================================================##
#
# Plot Train/Test/K-FOLD Accuracy
#
##===================================================================================##

def plot_train_test_k_fold_accuracy(
    val1, 
    val2,
    N,
    width,
    width_mult,
    fig_size,
    title,
    x_ticks,
    legends,
    file_path
):
    
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    
    bar1 = val1
    bar2 = val2

    ind = np.arange(N)  # the x locations for the groups
    width = width       # the width of the bars

    fig, ax = plt.subplots(figsize = fig_size)
    rects1 = ax.bar(ind, bar1, width, color='r', alpha=0.55)
    rects2 = ax.bar(ind + width*width_mult, bar2, width, color=color, alpha=0.55)

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Accuracy',fontsize='large', fontweight='bold')
    ax.set_title(title,fontsize='xx-large', fontweight='bold')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x_ticks, fontsize='large', fontweight='bold')

    ax.legend((rects1[0], rects2[0]), legends)


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.5*height,
                    '%.4f' % float(height),
                    ha='center', va='bottom', fontsize='large', fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
#     plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    