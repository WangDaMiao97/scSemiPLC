# -*- coding: utf-8 -*-
'''
@Time    : 2025/10/20 15:00
@Author  : Linjie Wang
@FileName: train_Kidney.py
@Software: PyCharm
'''

import random
import torch.utils.data
import torch.nn.parallel
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from model import actinn, clustering
from model.utilities import *
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os
import time
import warnings
warnings.filterwarnings("ignore")

def train(X_label, Y_label, X_unlabel, Y_unlabel, clustering, model, estimator, opt_est,
          Pre_epochs=100, Supervised_epochs=200, SemiSupervised_epochs=150):

    # unlabeled data
    dataset = CustomDataset(data=X_unlabel, transform_args=transformation_list)
    pretrain_loader = DataLoader(dataset, batch_size=512, shuffle=True, sampler=None,
                                 batch_sampler=None, collate_fn=None, pin_memory=True)
    # Pretraining models using contrastive learning with unlabeled data
    opt_model = torch.optim.Adam(params=model_cla.parameters(), lr=5e-4, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0.005, amsgrad=False)
    for _ in tqdm(range(Pre_epochs), desc="Pre-Training"):
        model.train()
        for batch_idx, (_, _, inputs_u_w, inputs_u_s) in enumerate(pretrain_loader):
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)
            N = inputs_u_w.shape[0]
            opt_model.zero_grad()
            _, logits_u_w = model(inputs_u_w)
            _, logits_u_s = model(inputs_u_s)
            Lc = ContrastiveLoss(logits_u_w, logits_u_s, 0.1)
            Lc.backward()
            opt_model.step()

    # Establish a mapping relationship between the clusters obtained using the Hungarian algorithm and the cell types.
    _, logits = model(torch.tensor(X_label).to(device))
    pred_labels = logits.argmax(1)
    gt_lables = torch.tensor(Y_label).to(device)
    num_classes = max(pred_labels.max().item(), gt_lables.max().item()) + 1
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long, device=pred_labels.device)
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = torch.sum((pred_labels == i) & (gt_lables == j))
    matrix_np = matrix.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-matrix_np)
    gt2pred_mapping = {col_ind[i]: row_ind[i] for i in range(len(row_ind))}
    pred2gt_mapping = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    aligned_gt_labels = gt_lables.clone()
    for old_label, new_label in gt2pred_mapping.items():
        aligned_gt_labels[gt_lables == old_label] = new_label # Application label mapping

    # labeled data
    label_data = [[torch.tensor(feat), label, aligned_label] for feat, label, aligned_label in zip(X_label, Y_label, aligned_gt_labels.cpu())]
    label_loader = DataLoader(label_data, batch_size=32, shuffle=True, sampler=None, batch_sampler=None, collate_fn=None, pin_memory=True)
    #  Supervised training stage
    opt_model = torch.optim.Adam(params=model_cla.parameters(), lr=1e-4, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0.005, amsgrad=False)
    for epoch in range(Supervised_epochs):
        model.train()
        for batch_idx, (inputs, _, aligned_targets) in enumerate(label_loader):
            opt_model.zero_grad()
            _, logits_x = model(inputs.to(device))
            Ls = F.cross_entropy(logits_x, aligned_targets.to(device), reduction='mean')
            Ls.backward()
            opt_model.step()
        if epoch % 20 == 0:
            print(f"Supervised epoch {epoch}, Loss:{Ls:.3f}")

    # Semi-supervised training stage with consistency regularization
    update_interval = 10
    inputs_x = torch.tensor(X_label).to(device)
    opt_model = torch.optim.Adam(params=model_cla.parameters(), lr=5e-5, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=0.005, amsgrad=False)
    for epoch in tqdm(range(SemiSupervised_epochs), desc="SemiSupervised-Training"):
        if epoch % update_interval ==0:
            with torch.no_grad():
                # update_labels
                model.eval()
                init_target_centers = get_centers(net=model, data=inputs_x, labels=aligned_gt_labels,
                                                  num_classes=NUM_CLASS)
                clustering.set_init_centers(init_target_centers)
                clustering.feature_clustering(model, data=torch.tensor(X_unlabel))
                targets_sel = clustering.samples['p_label']

        model.train()
        estimator.train()
        # Estimator optimization with model fixed
        for batch_idx, (idx, input_u, inputs_u_w, inputs_u_s) in enumerate(pretrain_loader):
            input_u = input_u.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)

            opt_est.zero_grad()
            with torch.no_grad():
                fea_sel, logits_sel = model(input_u)
                _, logits_sel_s = model(inputs_u_s)
                _, logits_sel_w = model(inputs_u_w)
            input_s = torch.cat([fea_sel, logits_sel], dim=1)
            input_s.requires_grad_(True)
            sim_s = estimator(input_s)
            # Consistency loss
            Lcw = (F.cross_entropy(logits_sel_w, targets_sel[idx], reduction='none') * sim_s).mean()
            Lcs = (F.cross_entropy(logits_sel_s, targets_sel[idx], reduction='none') * sim_s).mean()
            Lu_est = Lcs + Lcw
            Lu_est.backward()
            opt_est.step()

        # Model optimization with estimator fixed
        for (idx, input_u, inputs_u_w, inputs_u_s) in pretrain_loader:
            opt_model.zero_grad()
            _, logits_x = model(inputs_x)
            Ls = F.cross_entropy(logits_x, aligned_gt_labels, reduction='mean')  # 分类任务训练模型
            # Ls.backward()
            # opt_model.step()

            input_u = input_u.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)
            # opt_model.zero_grad()
            fea_sel, logits_sel = model(input_u)
            _, logits_sel_s = model(inputs_u_s)
            _, logits_sel_w = model(inputs_u_w)
            with torch.no_grad():
                input_s = torch.cat([fea_sel.detach(), logits_sel.detach()], dim=1)
                sim_s = estimator(input_s).detach()
            Lcw = (F.cross_entropy(logits_sel_w, targets_sel[idx], reduction='none') * sim_s).mean()
            Lcs = (F.cross_entropy(logits_sel_s, targets_sel[idx], reduction='none') * sim_s).mean()
            Lu_model = Ls + 0.5 * (Lcs + Lcw)
            Lu_model.backward()
            opt_model.step()

    model.eval()
    y_true = np.array([])
    y_pred = np.array([])
    eval_fea = np.array([])
    with torch.no_grad():
        fea_x, output_x = model(torch.tensor(X_label).to(device))
        _, predict_x = torch.max(output_x.squeeze(), 1)
        y_true = np.append(y_true, Y_label)
        y_pred = np.append(y_pred, predict_x.detach().cpu().numpy())
        eval_fea = np.append(eval_fea, fea_x.detach().cpu().numpy())

        fea_u, output_u = model(torch.tensor(X_unlabel).to(device))
        _, predict_u = torch.max(output_u.squeeze(), 1)
        y_true = np.append(y_true, Y_unlabel)
        y_pred = np.append(y_pred, predict_u.detach().cpu().numpy())
        eval_fea = np.append(eval_fea, fea_u.detach().cpu().numpy())

    aligned_pred_labels = y_pred.copy()
    # Mapping pred labels to ground truth's original numbers
    for pred_label_idx, target_gt_idx in pred2gt_mapping.items():
        aligned_pred_labels[y_pred == pred_label_idx] = target_gt_idx

    accuracy = accuracy_score(y_true, aligned_pred_labels)
    precision = precision_score(y_true, aligned_pred_labels, average="macro")
    recall = recall_score(y_true, aligned_pred_labels, average="macro")
    f1 = f1_score(y_true, aligned_pred_labels, average="macro")

    eval_fea = eval_fea.reshape(-1, 50)

    return y_true, y_pred, eval_fea, 100.*accuracy, 100. * precision, 100. * recall, 100. * f1

if __name__ == "__main__":
    # Record the results of the metrics
    valid_f1_sum, valid_acc_sum = 0, 0
    valid_pre_sum, valid_rec_sum = 0, 0
    out_pred = pd.DataFrame()
    out_true = pd.DataFrame()
    out_batch = pd.DataFrame()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    clustering = clustering.Clustering(0.005, 128 * 9, device=device)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Data loading...
    dataset_name = "Kidney"
    if not os.path.exists(f"data/{dataset_name}/scSemiAnno"):
        os.makedirs(f"data/{dataset_name}/scSemiAnno")
    type_to_label_dict = {'kidney capillary endothelial cell': 0, 'kidney cell': 1,
                          'kidney collecting duct epithelial cell': 2,
                          'kidney loop of Henle ascending limb epithelial cell': 3,
                          'kidney proximal straight tubule epithelial cell': 4, 'leukocyte': 5, 'macrophage': 6,
                          'mesangial cell': 7}

    # Bladder
    data_set_path = f"data/{dataset_name}/data_set.csv"
    label_set_path = f"data/{dataset_name}/label_set.csv"

    data_set = pd.read_csv(data_set_path)
    label_set = pd.read_csv(label_set_path)
    data_set.set_index("Unnamed: 0", inplace=True)
    data_set.rename_axis("", inplace=True)

    feature_size = data_set.shape[1]
    NUM_CLASS = len(type_to_label_dict)

    # Data Augmentation Settings
    transformation_list = [{  # weak
        'mask_percentage': 0.5, 'apply_mask_prob': 0.8,
        'noise_percentage': 0.5, 'sigma': 0.5, 'apply_noise_prob': 0.0
    }, {  # strong
        'mask_percentage': 0.5, 'apply_mask_prob': 0.0,
        'noise_percentage': 0.5, 'sigma': 0.5, 'apply_noise_prob': 0.8
    }]

    X = np.array(data_set).astype(np.float32)
    Y = convert_type2label(label_set.iloc[:, 1], type_to_label_dict)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    index = 0
    label_num = list(range(NUM_CLASS))
    for unlabel_index, label_index in skf.split(X, Y):
        X_label, X_unlabel = X[label_index], X[unlabel_index]
        Y_label, Y_unlabel = Y[label_index], Y[unlabel_index]

        # Model definition
        model_cla = actinn.ACTINN(output_dim=NUM_CLASS, input_size=feature_size).to(device)
        model_cla.apply(init_weights)
        # Confidence estimator
        model_est = actinn.Con_estimator(NUM_CLASS=NUM_CLASS).to(device)
        model_est.apply(init_weights)
        optimizer_est = torch.optim.Adam(params=model_est.parameters(), lr=5e-4, betas=(0.9, 0.999),
                                         eps=1e-08, weight_decay=0.005, amsgrad=False)
        # Model Training
        label_true, pred, fea, val_acc, val_pre, val_rec, val_f1 = \
            train(X_label, Y_label, X_unlabel, Y_unlabel, clustering=clustering,
                  model=model_cla, estimator=model_est, opt_est=optimizer_est)
        end_time = time.time()
        print('valid F1:{:.3f}%, valid_acc:{:.3f}%'.format(val_f1, val_acc))
        print('valid pre:{:.3f}%, valid_rec:{:.3f}%\n'.format(val_pre, val_rec))
        # Results recording
        valid_f1_sum += val_f1
        valid_acc_sum += val_acc
        valid_pre_sum += val_pre
        valid_rec_sum += val_rec

        colname = 'out' + str(index + 1)
        out_pred[colname] = pred
        out_true[colname] = label_true
        out_latent = pd.DataFrame(fea)
        out_latent.to_csv(f"data/{dataset_name}/scSemiAnno/out_latent" + str(index + 1) + ".csv", index=False)

        index += 1

    print('average accuracy:{:.3f}%, average F1:{:.3f}%'.format(valid_acc_sum / 10, valid_f1_sum / 10))
    print('average precision:{:.3f}%, average recall:{:.3f}%\n'.format(valid_pre_sum / 10, valid_rec_sum / 10))

    out_true.to_csv(f"data/{dataset_name}/scSemiAnno/out_true.csv", index=False)
    out_pred.to_csv(f"data/{dataset_name}/scSemiAnno/out_pred.csv", index=False)
    out_batch.to_csv(f"data/{dataset_name}/scSemiAnno/out_batch.csv", index=False)
