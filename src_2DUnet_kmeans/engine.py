import torch
import torch.optim as optim
import os
from tqdm import tqdm
import torch.nn as nn
from evaluation import *
from losses import *
from utils import *

class Engine:
    def __init__(
        self, 
        args,
        gpu_id,
        train_loader,
        val_loader,
        test_loader, 
        model,
        logger,
        save_dir
        ):
        self.model = model
        self.args = args
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.save_dir = save_dir

        if len(gpu_id) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_id)
        # pretrain_model_path = os.path.join(project_path, record_path, "weights", "embedder.pth")
        self.model = model.to('cuda')
        self.lr = args.lr
        # self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=0.000005)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        # self.loss = nn.BCEWithLogitsLoss()
        self.logger = logger
        self.sminloss = SimMinLoss()
        self.smaxloss = SimMaxLoss()
        self.save_model_path = os.path.join(save_dir, "model")
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
    
    def run(self):
        train_record = {'Dice':[], 'loss':[]}
        val_record = {'Dice':[], 'loss':[]}
        best_score = 0.0
        for epoch in range(1, self.epochs+1):
            train_loss, train_results = self.train(epoch)
            train_record['loss'].append(train_loss)
            train_record['Dice'].append(train_results['Dice'][0])
            self.logger.log(f"Epoch {epoch:4d}/{self.epochs:4d} | Train Loss: {train_loss:.3f}, Train Dice: {train_results['Dice'][0]:.3f}+-{train_results['Dice'][1]:.3f}, IOU: {train_results['IoU'][0]:.3f}+-{train_results['IoU'][1]:.3f}")

            val_loss, val_results = self.val(epoch)
            val_record['loss'].append(val_loss)
            val_record['Dice'].append(val_results['Dice'][0])
            self.logger.log(f"Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss:.3f}, Val Dice: {val_results['Dice'][0]:.3f}+-{val_results['Dice'][1]:.3f}, IOU: {val_results['IoU'][0]:.3f}+-{val_results['IoU'][1]:.3f}")

            cur_score = val_results['Dice'][0]
            if cur_score > best_score:
                best_score = cur_score
                self.logger.log(f"Save model at Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss:.3f}, Val Dice: {val_results['Dice'][0]:.3f}+-{val_results['Dice'][1]:.3f}, IOU: {val_results['IoU'][0]:.3f}+-{val_results['IoU'][1]:.3f}")
                model_path = os.path.join(self.save_model_path, "encoder.pth")
                torch.save(self.model.state_dict(), model_path)
        test_results = self.test(self.save_dir, model_path)
        self.logger.log(f"Test Dice: {test_results['Dice'][0]:.3f}+-{test_results['Dice'][1]:.3f}, IOU: {test_results['IoU'][0]:.3f}+-{test_results['IoU'][1]:.3f}, HD95: {test_results['HD95'][0]:.3f}+-{test_results['HD95'][1]:.3f}")    

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        train_labels = []
        pred_results = []
        out_results = []
        for data, label, chosen_idx, _ in train_bar:
            data = data.squeeze(0)
            label = label.squeeze(0)
            self.optimizer.zero_grad()
            loss, pred_batch, out_batch = self.step(data, label, chosen_idx)
            loss.backward()
            self.optimizer.step()
            total_num += data.shape[0]
            total_loss += loss.item() * data.shape[0]
            mean_loss = total_loss / total_num
            train_bar.set_description(f'Train Epoch: [{epoch}/{self.epochs}] Loss: {mean_loss:.4f}')
            pred_results.append(pred_batch)
            train_labels.append(label)
            out_results.append(out_batch)

        pred_results = torch.cat(pred_results, dim=0).numpy()
        train_labels = torch.cat(train_labels, dim=0).numpy()
        out_results = torch.cat(out_results, dim=0).numpy()
        results = self.evaluate(train_labels, out_results)
        train_results = {}
        train_results['Dice'] = (np.mean(results['Dice']), np.std(results['Dice']))
        train_results['IoU'] = (np.mean(results['IoU']), np.std(results['IoU']))

        return total_loss / total_num, train_results

    def step(self, data_batch, label_batch, chosen_idx):   
        logit = self.model(data_batch.cuda())
        c0_idx = torch.LongTensor(chosen_idx[0])
        c1_idx = torch.LongTensor(chosen_idx[1])
        c2_idx = torch.LongTensor(chosen_idx[2])
        # print(c0_idx, c1_idx, c2_idx)
        supervised_idx = torch.LongTensor([c0_idx[0], c1_idx[0], c2_idx[0]])
        # loss = self.loss(logit, label_batch.cuda())
        pred = torch.sigmoid(logit)

        foreground = (data_batch.cuda()) * pred
        cluster0_fore = torch.index_select(foreground, 0, c0_idx.cuda())
        cluster1_fore = torch.index_select(foreground, 0, c1_idx.cuda())
        cluster2_fore = torch.index_select(foreground, 0, c2_idx.cuda())
        cluster0_fore_flatten = torch.flatten(cluster0_fore, start_dim=1)
        cluster1_fore_flatten = torch.flatten(cluster1_fore, start_dim=1)
        cluster2_fore_flatten = torch.flatten(cluster2_fore, start_dim=1)


        supervised_pred = torch.index_select(pred, 0, supervised_idx.cuda())
        supervised_label = torch.index_select(label_batch, 0, supervised_idx)

        # print(supervised_logit.shape, supervised_label.shape)

        
        if len(chosen_idx[0]) != label_batch.shape[0]:
            loss_collect = {}
            loss = 0
            if len(cluster0_fore) > 1:
                loss_collect["SimMax_Foreground0"] = (self.smaxloss(cluster0_fore_flatten))
            if len(cluster1_fore) > 1:
                loss_collect["SimMax_Foreground1"] = (self.smaxloss(cluster1_fore_flatten))
            if len(cluster2_fore) > 1:
                loss_collect["SimMax_Foreground2"] = (self.smaxloss(cluster2_fore_flatten))
            # loss_collect["SimMin"] = (self.sminloss(foreground, background))
            # loss_collect["SimMax_Background"] = (self.smaxloss(background))
            loss_collect["Supervised_Dice"] = dice_loss(supervised_pred, supervised_label.cuda())
            for k, l in loss_collect.items():
                if k == "Supervised_Dice":
                    loss += 10* l
                else:
                    loss += l
                # print(k, l)
        else:
            loss = dice_loss(pred, label_batch.cuda())
        pred = pred.detach().cpu()
        out = torch.where(pred > 0.5, 1, 0)
        return loss, pred, out

    def inference(self, data_batch, label_batch):   
        logit = self.model(data_batch.cuda())
        pred = torch.sigmoid(logit)
        pred = pred.detach().cpu()
        out = torch.where(pred > 0.5, 1, 0)
        return pred, out

    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        val_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for data, label, chosen_idx, _ in val_bar:
                data = data.squeeze(0)
                label = label.squeeze(0)
                loss, pred_batch, out_batch = self.step(data, label, chosen_idx)
                total_num += data.shape[0]
                total_loss += loss.item() * data.shape[0]
                pred_results.append(pred_batch)
                val_labels.append(label)
                out_results.append(out_batch)
                mean_loss = total_loss / total_num
                val_bar.set_description(f'Val Epoch: [{epoch}/{self.epochs}] Loss: {mean_loss:.4f}')

        pred_results = torch.cat(pred_results, dim=0).numpy()
        val_labels = torch.cat(val_labels, dim=0).numpy()
        out_results = torch.cat(out_results, dim=0).numpy()
        results = self.evaluate(val_labels, out_results)
        val_results = {}
        val_results['Dice'] = (np.mean(results['Dice']), np.std(results['Dice']))
        val_results['IoU'] = (np.mean(results['IoU']), np.std(results['IoU']))

        return total_loss / total_num, val_results

    def test(self, save_dir, load_model_path):
        state_dict_weight = torch.load(load_model_path)
        self.model.load_state_dict(state_dict_weight, strict=False)
        
        test_image_path = os.path.join(save_dir, "test_image")
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)

        self.model.eval()
        test_bar = tqdm(self.test_loader, desc="Test stage")
        test_labels = []
        pred_results = []
        out_results = []
        with torch.no_grad():
            for data, label, case_id in test_bar:
                data = data.squeeze(0)
                label = label.squeeze(0)
                pred_batch, out_batch = self.inference(data, label)
                pred_results.append(pred_batch)
                test_labels.append(label)
                out_results.append(out_batch)
                draw_image(test_image_path, case_id[0], data, label, out_batch)

        pred_results = torch.cat(pred_results, dim=0).numpy()
        test_labels = torch.cat(test_labels, dim=0).numpy()
        out_results = torch.cat(out_results, dim=0).numpy()
        # print(out_results.shape, test_labels.shape)
        results = self.evaluate(test_labels, out_results)
        test_results = {}
        test_results['Dice'] = (np.mean(results['Dice']), np.std(results['Dice']))
        test_results['IoU'] = (np.mean(results['IoU']), np.std(results['IoU']))
        test_results['HD95'] = (np.mean(results['HD95']), np.std(results['HD95']))
        return test_results


    def evaluate(self, labels, pred):
        labels = labels.squeeze(1)
        pred = pred.squeeze(1)
        # print(labels.shape, pred.shape)
        result_metric = {'Dice':[], 'IoU':[], 'HD95':[]}
        for i in range(labels.shape[0]):
            result = compute_seg_metrics(labels[i], pred[i])
            for k in result_metric.keys():
                result_metric[k].append(result[k])
        return result_metric