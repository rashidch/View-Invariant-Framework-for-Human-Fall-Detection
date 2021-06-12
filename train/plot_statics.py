import time
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_Statistics(history, conf_train, conf_valid, name, epochs=500):
    currTime = time.asctime(time.localtime(time.time()))[4:-5]
    currTime = currTime.split(' ')
    currTime = currTime[0] + '_' + currTime[1] + '_' + currTime[2]

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for train_item, val_item in zip(history['train'], history['valid']):
        val_loss.append(val_item.__getitem__(0))
        val_acc.append(val_item.__getitem__(1).cpu().detach().numpy())

        train_loss.append(train_item.__getitem__(0))
        train_acc.append(train_item.__getitem__(1).cpu().detach().numpy())

    params = {'legend.fontsize': 'x-large',
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large'}
    plt.rcParams.update(params)

    acc_path = os.path.join('plots', 'acc', name)
    if not os.path.exists(acc_path):
        os.makedirs(acc_path)

    num_epochs = range(epochs)
    plt.figure(figsize=(12, 6))
    plt.plot(num_epochs, train_acc, label='Training Accuracy')
    plt.plot(num_epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy', fontsize=18)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(acc_path + '/acc_' + currTime + '.png')
    plt.show()

    loss_path = os.path.join('plots', 'loss', name)
    if not os.path.exists(loss_path):
        os.makedirs(loss_path)

    num_epochs = range(epochs)
    plt.figure(figsize=(12, 6))
    plt.plot(num_epochs, train_loss, label='Training Loss')
    plt.plot(num_epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(loss_path + '/loss_' + currTime + '.png')
    plt.show()

    conf_path = os.path.join('plots', 'train_conf', name)
    if not os.path.exists(conf_path):
        os.makedirs(conf_path)

    # set the size of figure 542129345
    plt.figure(figsize=(8, 8))
    # normalize each column (class) with total datapoints in that column
    conf_train = conf_train.astype('float') / conf_train.sum(axis=1) * 100
    # plot confusion matrix
    p = sns.heatmap(conf_train, xticklabels=['Fall', 'Stand'], yticklabels=['Fall', 'Stand'],
                    cbar=False, annot=True, cmap='coolwarm', robust=True, fmt='.1f', annot_kws={'size': 20})
    plt.title('Training matrix: Actual labels Vs Predicted labels')
    plt.savefig(conf_path + '/conf_' + currTime + '.png')
    plt.show()

    conf_path = os.path.join('plots', 'val_conf', name)
    if not os.path.exists(conf_path):
        os.makedirs(conf_path)

    # set the size of figure
    plt.figure(figsize=(8, 8))
    # normalize each column (class) with total datapoints in that column
    conf_valid = conf_valid.astype('float') / conf_valid.sum(axis=1) * 100
    # plot confusion matrix
    p = sns.heatmap(conf_valid, xticklabels=['Fall', 'Stand'], yticklabels=['Fall', 'Stand'],
                    cbar=False, annot=True, cmap='coolwarm', robust=True, fmt='.1f', annot_kws={'size': 20})
    plt.title('Validation matrix: Actual labels vs Predicted labels')
    plt.savefig(conf_path + '/conf_' + currTime + '.png')
    plt.show()