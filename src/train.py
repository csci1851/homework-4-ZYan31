import torch
import torch.nn as nn
import torch.optim as optim
import os
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

MODEL_DIR = "results"

criterion = nn.BCEWithLogitsLoss()

def evaluate_model(model, dataloader, device):
    # TODO: set model.eval(), disable grad, iterate batches, move x/y to device
    model.eval()
    metrix = {"accuracy": [], "auroc": [], "confusion_matrix": []}
    allprobs = []
    alltargets = []
    with torch.no_grad():
        for x, y in iter(dataloader):
            # move to GPU
            x, y = x.to(device), y.to(device).squeeze().float()
            # TODO: run forward to logits, apply sigmoid to get probabilities
            out = model.forward(x).squeeze()
            pred = torch.sigmoid(out)
            # TODO: collect probs and targets on CPU lists for metric computation
            allprobs.extend(pred.to("cpu").tolist())
            alltargets.extend(y.to("cpu").tolist())
    # TODO: threshold probs at 0.5 for class predictions
    #print(type(probs))
    # print(sorted(set(alltargets)))
    # probs = torch.as_tensor(allprobs)
    # preds = (probs > 0.5).long()
    # targets = torch.as_tensor(alltargets).long()
    # print("target classes:", torch.unique(targets, return_counts=True))
    # print("pred classes:", torch.unique(preds, return_counts=True))
    # print("first 20 probs:", probs[:20])
    # print("first 20 preds:", preds[:20])
    # print("first 20 targets:", targets[:20])
    preds = (torch.tensor(allprobs) > 0.5)
    alltargets = torch.tensor(alltargets)
    # TODO: compute accuracy_score, roc_auc_score, and confusion_matrix
    metrix["accuracy"] = accuracy_score(alltargets, preds)
    metrix["auroc"] = roc_auc_score(alltargets, allprobs)
    metrix["confusion_matrix"] = confusion_matrix(alltargets, preds)
    # TODO: return a dict with keys: "accuracy", "auroc", "confusion_matrix"
    return metrix

def train_cnn_model(model, train_loader, val_loader, epochs, device, lr=1e-3, save_path=f"{MODEL_DIR}/cnn_best_model.pt"):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    best_val_auroc = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_auroc": []}
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        train_samples = 0
        val_samples = 0
        for Xtrain, ytrain in iter(train_loader):
            optimizer.zero_grad()
            Xtrain, ytrain = Xtrain.to(device), ytrain.to(device).squeeze().float()
            out = model(Xtrain).squeeze()
            loss = criterion(out, ytrain)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()* Xtrain.size(0)
            train_samples += Xtrain.size(0)
        avg_train_loss = train_loss/train_samples
        metrix = evaluate_model(model, val_loader, device)
        with torch.no_grad():
            for Xval, yval in iter(val_loader):
                Xval, yval = Xval.to(device), yval.to(device).float().squeeze()
                out = model(Xval).squeeze()
                loss = criterion(out, yval)
                val_loss += loss.item()* Xval.size(0)
                val_samples += Xval.size(0)
        if best_val_auroc < metrix["auroc"]:
            best_val_auroc = metrix["auroc"]
            torch.save(model.state_dict(), save_path)
        avg_val_loss = val_loss/val_samples
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(metrix["accuracy"])
        history["val_auroc"].append(metrix["auroc"])
    print("Best AUROC: ", best_val_auroc)
    return history

def eval_cnn_model(model, test_loader, device, model_path=f"{MODEL_DIR}/cnn_best_model.pt"):
    # TODO: load state_dict from model_path, move to device, set eval
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    # TODO: call evaluate_model on test_loader
    metrix = evaluate_model(model, test_loader, device)
    # TODO: print accuracy and AUROC, return metrics dict
    print("Accuracy: ", metrix["accuracy"], "\nAUROC: ", metrix["auroc"])
    return metrix

def train_fcn_model(model, train_loader, val_loader, epochs, device, lr=1e-3, save_path=f"{MODEL_DIR}/fcn_best_model.pt"):
    # TODO: move model to device; create Adam optimizer with lr
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    # TODO: initialize best_val_auroc and history dict with lists:
    #       train_loss, val_loss, val_accuracy, val_auroc
    best_val_auroc = 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_auroc": []}
    # TODO: for each epoch:
    #       - set model.train()
    #       - loop over train_loader: forward, compute BCEWithLogitsLoss,
    #         backward, optimizer.step(), accumulate running_loss
    #       - compute avg train loss
    #       - run evaluate_model on val_loader for accuracy/auroc
    #       - compute val loss with no_grad (separate pass over val_loader)
    #       - append metrics/losses to history
    #       - print epoch summary
    #       - if val AUROC improves, save model state_dict
    #for (Xtrain, ytrain, Xval, yval) in zip(iter(train_loader), iter(val_loader)):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for i in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        train_samples = 0
        val_samples = 0
        for Xtrain, ytrain in iter(train_loader):
            # w0 = next(model.parameters()).detach().clone()
            optimizer.zero_grad()
            Xtrain, ytrain = Xtrain.to(device), ytrain.to(device).float().view(-1)
            # print(Xtrain.min().item(), Xtrain.max().item(), Xtrain.mean().item(), Xtrain.std().item())
            # print(torch.unique(ytrain))
            out = model(Xtrain).view(-1)
            loss = criterion(out, ytrain)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()* Xtrain.size(0)
            train_samples += Xtrain.size(0)
            # w1 = next(model.parameters()).detach().clone()
            # print("weight changed:", not torch.equal(w0, w1))
            # print("grad mean:", next(model.parameters()).grad.abs().mean().item())
            # print("loss:", loss.item())
        avg_train_loss = train_loss/train_samples
        metrix = evaluate_model(model, val_loader, device)
        with torch.no_grad():
            for Xval, yval in iter(val_loader):
                Xval, yval = Xval.to(device), yval.to(device).float().view(-1)
                out = model(Xval).view(-1)
                loss = criterion(out, yval)
                val_loss += loss.item()* Xval.size(0)
                val_samples += Xval.size(0)
        if best_val_auroc < metrix["auroc"]:
            best_val_auroc = metrix["auroc"]
            torch.save(model.state_dict(), save_path)
        avg_val_loss = val_loss/val_samples
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(metrix["accuracy"])
        history["val_auroc"].append(metrix["auroc"])
    # TODO: print best AUROC and return history
    print("Best AUROC: ", best_val_auroc)
    return history

def eval_fcn_model(model, test_loader, device, model_path=f"{MODEL_DIR}/fcn_best_model.pt"):
    # TODO: load FCN state_dict, move to device, set eval
    # TODO: call evaluate_model, print accuracy/AUROC, return metrics
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    metrix = evaluate_model(model, test_loader, device)
    print("Accuracy: ", metrix["accuracy"], "\nAUROC: ", metrix["auroc"])
    return metrix
