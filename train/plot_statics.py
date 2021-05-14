
import time
import matplotlib.pyplot as plt
import seaborn as sns

def plot_Statistics(history, conf_train, conf_valid, name):

    currTime = time.asctime( time.localtime(time.time()))[4:-5]
    currTime = currTime.split(' ')
    currTime = currTime[0]+'_'+currTime[1]+'_'+currTime[2]

    train_acc  = []
    train_loss = []
    val_acc    = []
    val_loss   = []
    for train_item, val_item in zip(history['train'],history['valid']):
        
        val_loss.append(val_item.__getitem__(0))
        val_acc.append(val_item.__getitem__(1).cpu().detach().numpy())
        
        train_loss.append(train_item.__getitem__(0))
        train_acc.append(train_item.__getitem__(1).cpu().detach().numpy())
    
    params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)


    num_epochs=range(1000)
    plt.figure(figsize=(12, 6))
    plt.plot(num_epochs, train_acc, label='Training Accuracy')
    plt.plot(num_epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy', fontsize=18)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/acc'+name+currTime+'.png')
    plt.show()

    num_epochs=range(1000)
    plt.figure(figsize=(12, 6))
    plt.plot(num_epochs, train_loss, label='Training Loss')
    plt.plot(num_epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig('plots/loss'+name+currTime+'.png')
    plt.show()

    #set the size of figure 542129345
    plt.figure(figsize=(8,8))
    #normalize each column (class) with total datapoints in that column  
    conf_train = conf_train.astype('float')/conf_train.sum(axis=1)*100
    #plot confusion matrix 
    p=sns.heatmap(conf_train, xticklabels=['Fall','Stand','Tie'], yticklabels=['Fall','Stand','Tie'],
                cbar=False, annot=True, cmap='coolwarm',robust=True, fmt='.1f',annot_kws={'size':20})
    plt.title('Training matrix: Actual labels Vs Predicted labels')
    plt.savefig('plots/train_cf'+name+currTime+'.png')

    #set the size of figure 
    plt.figure(figsize=(8,8))
    #normalize each column (class) with total datapoints in that column  
    conf_valid = conf_valid.astype('float')/conf_valid.sum(axis=1)*100
    #plot confusion matrix 
    p=sns.heatmap(conf_valid, xticklabels=['Fall','Stand','Tie'], yticklabels=['Fall','Stand','Tie'],
                cbar=False, annot=True, cmap='coolwarm',robust=True, fmt='.1f',annot_kws={'size':20})
    plt.title('Validation matrix: Actual labels vs Predicted labels')
    plt.savefig('plots/valcf'+name+currTime+'.png')