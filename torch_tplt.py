# pytorch template

# The names used in this tplt need to be changed accordingly.

from tqdm import tqdm
import torch
from torch import nn
from copy import deepcopy
from torchkeras.metrics import Accuracy
import logging
import sys


model = nn.Sequential()

loss_fn= nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
metrics_dict = {"acc": Accuracy()}

epochs = 5
ckpt_path = "checkpoint.pt"

#early_stopping相关设置
monitor="val_acc"
patience=5
mode="max"

history = {}

for epoch in range(1, epochs+1):
    logger.info(("Epoch {0} / {1}".format(epoch, epochs)))

    ####################
    # 1. Train
    ####################
    model.train()
    total_loss, step = 0, 0
    loop = tqdm(enumerate(train_data_after_dataloader), bar_format='\033[42m{bar}\033[0m {n}/{total}', total =len(train_data_after_dataloader),file = sys.stdout)
    train_metrics_dict = deepcopy(metrics_dict)

    for i, batch in loop:
        # 1.1 conduct training in each batch
        features, labels = batch
        
        ###
        # All can be sent to accelerate()
        ###

        # 1.2 Forward
        preds = model(features)
        loss = loss_fn(preds, labels)

        # 1.3 Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 1.4 Metrics
        # This metrics related to torchkeras.
        # Pytorch might not use this style.
        step_metrics = {"train_"+name:metric_fn(preds, labels).item()
                        for name, metric_fn in train_metrics_dict.items()}

        step_log = dict({"train_loss":loss.item()},**step_metrics)

        total_loss += loss.item()

        step+=1
        if i!=len(dl_train)-1:
            loop.set_postfix(**step_log)
        else:
            epoch_loss = total_loss/step
            epoch_metrics = {"train_"+name:metric_fn.compute().item()
                             for name,metric_fn in train_metrics_dict.items()}
            epoch_log = dict({"train_loss":epoch_loss},**epoch_metrics)
            loop.set_postfix(**epoch_log)

            for name,metric_fn in train_metrics_dict.items():
                metric_fn.reset()
    
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]

    ####################
    # 2. validate
    ####################
    model.eval()

    total_loss,step = 0,0
    loop = tqdm(enumerate(dl_val), total =len(dl_val),file = sys.stdout)

    val_metrics_dict = deepcopy(metrics_dict)

    with torch.no_grad():
        for i, batch in loop:

            features,labels = batch

            #forward
            preds = net(features)
            loss = loss_fn(preds,labels)

            #metrics
            step_metrics = {"val_"+name:metric_fn(preds, labels).item()
                            for name,metric_fn in val_metrics_dict.items()}

            step_log = dict({"val_loss":loss.item()},**step_metrics)

            total_loss += loss.item()
            step+=1
            if i!=len(dl_val)-1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = (total_loss/step)
                epoch_metrics = {"val_"+name:metric_fn.compute().item()
                                 for name,metric_fn in val_metrics_dict.items()}
                epoch_log = dict({"val_loss":epoch_loss},**epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name,metric_fn in val_metrics_dict.items():
                    metric_fn.reset()

    epoch_log["epoch"] = epoch
    for name, metric in epoch_log.items():
        history[name] = history.get(name, []) + [metric]

    ####################
    # 2.2 early stopping
    ####################
    arr_scores = history[monitor]
    best_score_idx = np.argmax(arr_scores) if mode=="max" else np.argmin(arr_scores)
    if best_score_idx==len(arr_scores)-1:
        torch.save(net.state_dict(),ckpt_path)
        print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
             arr_scores[best_score_idx]),file=sys.stderr)
    if len(arr_scores)-best_score_idx>patience:
        print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
            monitor,patience),file=sys.stderr)
        break
    net.load_state_dict(torch.load(ckpt_path))

dfhistory = pd.DataFrame(history)

############################################
# 3. Evaluate model on train and validation
############################################
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt

def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_"+metric]
    val_metrics = dfhistory['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(dfhistory,"loss")
plot_metric(dfhistory,"acc")

############################################
# 4. Evaluate model on test
############################################
y_pred_probs = torch.sigmoid(net(torch.tensor(x_test[0:10]).float())).data
y_pred_probs

y_pred = torch.where(y_pred_probs>0.5,
        torch.ones_like(y_pred_probs),torch.zeros_like(y_pred_probs))
y_pred

############################################
# 5. Save the model
############################################

# 5.1 Save the model parameters (recommend)
torch.save(net.state_dict(), "./data/net_parameter.pt")

net_clone = create_net()
net_clone.load_state_dict(torch.load("./data/net_parameter.pt"))

torch.sigmoid(net_clone.forward(torch.tensor(x_test[0:10]).float())).data

# 5.2 Save the whole model (not recommend)
torch.save(net, './data/net_model.pt')
net_loaded = torch.load('./data/net_model.pt')
torch.sigmoid(net_loaded(torch.tensor(x_test[0:10]).float())).data