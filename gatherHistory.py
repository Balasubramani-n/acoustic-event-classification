import pickle
import seaborn as sns
import numpy as np

with open("training_histories/history_1", "rb") as fp:   
    history1 = pickle.load(fp)

with open("training_histories/history_2", "rb") as fp:   
    history2 = pickle.load(fp)

with open("training_histories/history_3", "rb") as fp:   
    history3 = pickle.load(fp)

with open("training_histories/history_4", "rb") as fp:   
    history4 = pickle.load(fp)

with open("training_histories/history_5", "rb") as fp:   
    history5 = pickle.load(fp)

with open("training_histories/history_6", "rb") as fp:   
    history6 = pickle.load(fp)

with open("training_histories/history_7", "rb") as fp:   
    history7 = pickle.load(fp)

with open("training_histories/history_8", "rb") as fp:   
    history8 = pickle.load(fp)

with open("training_histories/history_9", "rb") as fp:   
    history9 = pickle.load(fp)

with open("training_histories/history_10", "rb") as fp:   
    history10 = pickle.load(fp)

tot_history = [history1, history2, history3, history4, history5, history6, history7, history8, history9, history10]

def compute_min_loss():
    avg_train_loss_per_fold = []
    avg_val_loss_per_fold = []
    std_train_loss_per_fold = []
    std_val_loss_per_fold = []

    tot_min_train_loss = 0.0
    tot_min_val_loss = 0.0

    # iterate over all folds
    for fold_histories in tot_history:
        fold_min_train_losses = []
        fold_min_val_losses = []

        # collect min losses of all fold's histories
        # for history in fold_histories:
        fold_min_train_losses.append(np.min(fold_histories['loss']))
        fold_min_val_losses.append(np.min(fold_histories['val_loss']))

        # avg min loss
        avg_train_loss_per_fold.append(np.mean(fold_min_train_losses))
        avg_val_loss_per_fold.append(np.mean(fold_min_val_losses))
        # std of min loss
        std_train_loss_per_fold.append(np.std(fold_min_train_losses))
        std_val_loss_per_fold.append(np.std(fold_min_val_losses))
        
        # add fold's avg min loss to sum of total loss 
        tot_min_train_loss += np.mean(fold_min_train_losses)
        tot_min_val_loss += np.mean(fold_min_val_losses)

    avg_min_train_loss = tot_min_train_loss / len(tot_history)
    avg_min_val_loss = tot_min_val_loss / len(tot_history)

    print("10-Fold Cross Validation\n")
    print("Average train min loss: %.4f" % avg_min_train_loss)
    print("Average validation min loss: %.4f" % avg_min_val_loss)

def compute_max_accuracy():

    avg_train_acc_per_fold = []
    avg_val_acc_per_fold = []
    std_train_acc_per_fold = []
    std_val_acc_per_fold = []

    tot_max_train_acc = 0.0
    tot_max_val_acc = 0.0

    # iterate over all folds
    for fold_histories in tot_history:
        fold_max_train_accs = []
        fold_max_val_accs = []

        # collect max accuracies of all fold's histories
        fold_max_train_accs.append(np.max(fold_histories['accuracy']))
        fold_max_val_accs.append(np.max(fold_histories['val_accuracy']))
        
        # avg max accuracy
        avg_train_acc_per_fold.append(np.mean(fold_max_train_accs))
        avg_val_acc_per_fold.append(np.mean(fold_max_val_accs))
        # std of max accuracy
        std_train_acc_per_fold.append(np.std(fold_max_train_accs))
        std_val_acc_per_fold.append(np.std(fold_max_val_accs))
            
        # add fold's avg max accuracy to sum of total accuracy 
        tot_max_train_acc += np.mean(fold_max_train_accs)
        tot_max_val_acc += np.mean(fold_max_val_accs)
    
    avg_max_train_acc = tot_max_train_acc / len(tot_history)
    avg_max_val_acc = tot_max_val_acc / len(tot_history)

    print("10-Fold Cross Validation\n")
    print("Average train max accuracy: %.4f %%" % (avg_max_train_acc * 100))
    print("Average validation max accuracy: %.4f %%" % (avg_max_val_acc * 100))


compute_min_loss()
compute_max_accuracy()
