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
            self.scheduler.step()
            self.logger.log(f"Epoch {epoch:4d}/{self.epochs:4d} | Current learning rate: {self.scheduler.get_last_lr()}")

            cur_score = val_results['Dice'][0]
            if cur_score > best_score:
                best_score = cur_score
                self.logger.log(f"Save model at Epoch {epoch:4d}/{self.epochs:4d} | Val Loss: {val_loss:.3f}, Val Dice: {val_results['Dice'][0]:.3f}+-{val_results['Dice'][1]:.3f}, IOU: {val_results['IoU'][0]:.3f}+-{val_results['IoU'][1]:.3f}")
                model_path = os.path.join(self.save_model_path, "encoder.pth")
                torch.save(self.model.state_dict(), model_path)
        test_results = self.ESV_EDV_test(self.save_dir, model_path)
        self.logger.log(f"Test Dice: {test_results['Dice'][0]:.3f}+-{test_results['Dice'][1]:.3f}, IOU: {test_results['IoU'][0]:.3f}+-{test_results['IoU'][1]:.3f}, HD95: {test_results['HD95'][0]:.3f}+-{test_results['HD95'][1]:.3f}")    

    def train(self, epoch):
        self.model.train()
        train_bar = tqdm(self.train_loader)
        total_loss, total_num = 0.0, 0
        result_epoch_dice = []
        result_epoch_iou = []
        for (data, (large_label, small_label), chosen_id, _) in train_bar:
            label = torch.cat([small_label, large_label]).unsqueeze(1)
            data = data.squeeze(0)
            self.optimizer.zero_grad()
            loss, pred_batch, out_batch = self.step(data, label, chosen_id)
            loss.backward()
            self.optimizer.step()
            total_num += data.shape[0]
            total_loss += loss.item() * data.shape[0]
            mean_loss = total_loss / total_num
            train_bar.set_description(f'Train Epoch: [{epoch}/{self.epochs}] Loss: {mean_loss:.4f}')
            results = self.evaluate(label.numpy(), out_batch.numpy())
            result_epoch_dice.append(results['Dice'])
            result_epoch_iou.append(results['IoU'])

        train_results = {}
        train_results['Dice'] = (np.mean(result_epoch_dice), np.std(result_epoch_dice))
        train_results['IoU'] = (np.mean(result_epoch_iou), np.std(result_epoch_iou))

        return total_loss / total_num, train_results

    def step(self, data_batch, label_batch, chosen_idx):   
        logit = self.model(data_batch.cuda())
        c0_idx = torch.LongTensor(chosen_idx[0])
        c1_idx = torch.LongTensor(chosen_idx[1])

        supervised_idx = torch.LongTensor([c0_idx[0], c1_idx[0]])
        # loss = self.loss(logit, label_batch.cuda())
        pred = torch.sigmoid(logit)

        foreground = (data_batch.cuda()) * pred
        cluster0_fore = torch.index_select(foreground, 0, c0_idx.cuda())
        cluster1_fore = torch.index_select(foreground, 0, c1_idx.cuda())
        cluster0_fore_flatten = torch.flatten(cluster0_fore, start_dim=1)
        cluster1_fore_flatten = torch.flatten(cluster1_fore, start_dim=1)


        supervised_pred = torch.index_select(pred, 0, supervised_idx.cuda())
        supervised_label = label_batch
        

        # print(supervised_logit.shape, supervised_label.shape)

        
        if len(chosen_idx[0]) != label_batch.shape[0]:
            loss_collect = {}
            loss = 0
            if len(cluster0_fore) > 1:
                loss_collect["SimMax_Foreground0"] = (self.smaxloss(cluster0_fore_flatten))
            if len(cluster1_fore) > 1:
                loss_collect["SimMax_Foreground1"] = (self.smaxloss(cluster1_fore_flatten))
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
        supervised_pred = supervised_pred.detach().cpu()
        out = torch.where(supervised_pred > 0.5, 1, 0)
        return loss, supervised_pred, out

    def inference(self, data_batch):   
        logit = self.model(data_batch.cuda())
        pred = torch.sigmoid(logit)
        pred = pred.detach().cpu()
        out = torch.where(pred > 0.5, 1, 0)
        return pred, out

    def val(self, epoch):
        self.model.eval()
        val_bar = tqdm(self.val_loader)
        total_loss, total_num = 0.0, 0
        result_dice = []
        result_iou = []
        with torch.no_grad():
            for (data, (large_label, small_label), chosen_id, _) in val_bar:
                data = data.squeeze(0)
                label = torch.cat([small_label, large_label]).unsqueeze(1)
                loss, _, out_batch = self.step(data, label, chosen_id)
                total_num += data.shape[0]
                total_loss += loss.item() * data.shape[0]
                mean_loss = total_loss / total_num
                val_bar.set_description(f'Val Epoch: [{epoch}/{self.epochs}] Loss: {mean_loss:.4f}')

                results = self.evaluate(label.numpy(), out_batch.numpy())
                result_dice.append(results['Dice'])
                result_iou.append(results['IoU'])

        val_results = {}
        val_results['Dice'] = (np.mean(result_dice), np.std(result_dice))
        val_results['IoU'] = (np.mean(result_iou), np.std(result_iou))

        return total_loss / total_num, val_results

    def ESV_EDV_test(self, save_dir, load_model_path):
        state_dict_weight = torch.load(load_model_path)
        self.model.load_state_dict(state_dict_weight, strict=False)
        
        test_image_path = os.path.join(save_dir, "test_image")
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)

        self.model.eval()
        test_bar = tqdm(self.test_loader, desc="Test stage")
        result_dice = []
        result_iou = []
        result_HD95 = []
        with torch.no_grad():
            for (data, (large_label, small_label, large_index, small_index), video_id) in test_bar:
                data = data.squeeze(0)
                label = torch.cat([large_label, small_label]).unsqueeze(1)
                _, out_batch = self.inference(data)
                chosen_id = torch.LongTensor([large_index, small_index, (large_index+small_index)//2])
                data = torch.index_select(data, 0, chosen_id)
                out_batch_sel = torch.index_select(out_batch, 0, chosen_id)

                draw_image_index(test_image_path, video_id[0], data, label, out_batch_sel, chosen_id)
                results = self.evaluate(label.numpy(), out_batch_sel.numpy())
                result_dice.append(results['Dice'])
                result_iou.append(results['IoU'])
                result_HD95.append(results['HD95'])

        test_results = {}
        test_results['Dice'] = (np.mean(result_dice), np.std(result_dice))
        test_results['IoU'] = (np.mean(result_iou), np.std(result_iou))
        test_results['HD95'] = (np.mean(result_HD95), np.std(result_HD95))

        return test_results

    def test(self, save_dir, load_model_path):
        state_dict_weight = torch.load(load_model_path)
        self.model.load_state_dict(state_dict_weight, strict=False)
        
        test_image_path = os.path.join(save_dir, "test_image_extra")
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)

        self.model.eval()
        test_bar = tqdm(self.test_loader, desc="Test stage")
        result_dice = []
        result_iou = []
        result_HD95 = []
        with torch.no_grad():
            for image, label, case_id in test_bar:
                # print(label.shape)
                image = image.squeeze(0)
                label = label.squeeze(0).unsqueeze(1)
                _, out_batch = self.inference(image)
                # print(out_batch.shape)
                draw_image(test_image_path, case_id[0], image, label, out_batch)
                results = self.evaluate(label.numpy(), out_batch.numpy())
                result_dice.append(results['Dice'])
                result_iou.append(results['IoU'])
                result_HD95.append(results['HD95'])
        result_dice = [i for k in result_dice for i in k]
        result_iou = [i for k in result_iou for i in k]
        result_HD95 = [i for k in result_HD95 for i in k]
        test_results = {}
        test_results['Dice'] = (np.mean(result_dice), np.std(result_dice))
        test_results['IoU'] = (np.mean(result_iou), np.std(result_iou))
        test_results['HD95'] = (np.mean(result_HD95), np.std(result_HD95))

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