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
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import seaborn as sea
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix



##===================================================================================##
#
# Set all seeds
#
##===================================================================================##

# kindly manually change seed here only !!
## seeds to be used from 315-324
init_seed = 324

def get_seed():
    return init_seed

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




def meth_data_preprocess(df_luad, df_lusu):
    
    ## transpose the dataset
    df_lusu = df_lusu.T
    df_luad = df_luad.T
    
    ## insert respective labels to the samples
    label_luad = np.ones(len(df_luad), dtype=int) ## 1 : LUAD
    label_lusu = np.zeros(len(df_lusu), dtype=int) ## 0 : LUSU
    df_luad.insert(len(df_luad.columns), 'label', label_luad)
    df_lusu.insert(len(df_lusu.columns), 'label', label_lusu)
    
    ## save gene symbols as columns for future use
    columns = list(df_luad.columns)
    
    ## fill nan with mean of the respective columns for luad and lusc using SimpleImputer
    impute_mean_luad = SimpleImputer(missing_values=np.nan, strategy='mean')
    impute_mean_lusu = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_luad_new = impute_mean_luad.fit_transform(df_luad)
    df_lusu_new = impute_mean_lusu.fit_transform(df_lusu)
    
    ## as SimpleImputer converts dataframe to numpy arrays, reconvert them to dataframe
    df_luad_new = pd.DataFrame(df_luad_new, columns=columns)
    df_lusu_new = pd.DataFrame(df_lusu_new, columns=columns)
    df_luad_new = df_luad_new.apply(pd.to_numeric)
    df_lusu_new = df_lusu_new.apply(pd.to_numeric)
    
    ## append LUAD and LUSC datasets
    df_final = df_luad_new.append(df_lusu_new)
    df_final.reset_index(drop=True, inplace=True)
    df_final = df_final.astype({'label':int})
    
    ## return df_final
    return df_final



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
    skip_epoch_stats=False,
    patience=5
):
    
    count_patience = 0
    init_loss = 1000
    mean_loss_enc = 0
    log_dict = {'train_loss_per_batch': [],'final_loss':0.}

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
            
            # LOGGING
            log_dict['train_loss_per_batch'].append(loss.item())
        print('Epoch: %03d/%03d | Loss: %.4f'% (epoch+1, num_epochs, loss.item()))
        if(init_loss > loss.item()):
            init_loss = loss.item()
            count_patience = 0
        else:
            count_patience = count_patience+1
            if count_patience >= patience:
                print("Early stopping (patience = {}). Lowest loss achieved: {}".format(patience, init_loss))
                log_dict['final_loss'] = init_loss
                break
        
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    log_dict['final_loss']=loss.item()
    return log_dict



##===================================================================================##
#
# Compute pearson correlation 
#
##===================================================================================##

## https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))    


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
    device,
    train_loader,
    valid_loader,
    patience = 5
):
    
    log_dict = {'train_loss_per_batch':[], 'train_loss_per_epoch':[], 'train_acc':[], 'valid_acc':[]}

    loss_fn = F.cross_entropy
    
    count_patience = 0
    init_loss = 1000
    model_copy = []
    epoch_num = 0
    init_acc_train = 0.
    init_acc_valid = 0.
    start_time = time.time()
    for epoch in tqdm(range(num_epochs)):


        # ----- model training ------

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.float()
            features = features.to(device)
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
        
        # ----- model evaluation ------
        
        
        train_acc = compute_accuracy(model, train_loader, device)
        log_dict['train_acc'].append(train_acc.item())

        valid_acc = compute_accuracy(model, valid_loader, device)
        log_dict['valid_acc'].append(valid_acc.item())
        
        print('Epoch: %03d/%03d | Loss: %.4f | Acc_train: %.4f | Acc_valid: %.4f'%
              (epoch+1, num_epochs, loss.item(),train_acc.item(),valid_acc.item()))
        
        ## https://pythonguides.com/pytorch-early-stopping/
        
        if(init_loss > loss.item()):
            init_loss = loss.item()
            init_acc_valid = valid_acc.item()
            init_acc_train = train_acc.item()
            count_patience = 0
            epoch_num = epoch+1
            model_copy = []
            model_copy = copy.deepcopy(model)
        else:
            count_patience = count_patience+1
            if count_patience >= patience:
                print('==========================================================================================')
                print("Early stopping @ epoch: %03d. Min Loss: %.4f | Noted Train Acc: %.4f | Noted Valid Acc: %.4f"%
                      (epoch_num, init_loss, init_acc_train, init_acc_valid))
                print('==========================================================================================')
                model = model_copy
                break

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
# Read/Write Genes
#
##===================================================================================##

def read_essential_genes(PATH):
    essential_genes=[]
    gene_list_path = PATH+"project_summary_seed_wise_meth/essential_genes_1000.kd"
    with open(gene_list_path, "r") as file:
        for gene in file:
            gene=gene.strip()
            essential_genes.append(gene)
    return essential_genes


def write_seed_genes(PATH, seed, common_genes, quant):
    if quant == 300:
        gene_list_path = PATH+"project_summary_seed_wise_meth/seed="+str(seed)+"/deg_300.kd"
    elif quant == 500:
        gene_list_path = PATH+"project_summary_seed_wise_meth/seed="+str(seed)+"/deg_500.kd"
    else:
        gene_list_path = PATH+"project_summary_seed_wise_meth/seed="+str(seed)+"/deg_1000.kd"
    
    with open(gene_list_path, "w") as file:
        for gene in list(common_genes):
            file.write("%s\n" % gene)
    display(HTML("Common Genes written successfully !"))
        
        
def write_luad_lusu_seed_genes(PATH, seed, luad_seed_genes, lusu_seed_genes):
    gene_list_path_luad = PATH+"project_summary_seed_wise/seed="+str(seed)+"/deg_LUAD.kd"
    gene_list_path_lusu = PATH+"project_summary_seed_wise/seed="+str(seed)+"/deg_LUSU.kd"
    
    with open(gene_list_path_luad, "w") as file:
        for gene in list(luad_seed_genes):
            file.write("%s\n" % gene)
    display(HTML("LUAD Genes written successfully !"))
    with open(gene_list_path_lusu, "w") as file:
        for gene in list(lusu_seed_genes):
            file.write("%s\n" % gene)
    display(HTML("LUSU Genes written successfully !"))
    


        
        
        
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
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    