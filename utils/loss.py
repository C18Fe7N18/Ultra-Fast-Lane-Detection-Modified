import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        # loss0 : n,c,w
        loss = torch.cat(loss_all)
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()

    def forward(self, x, labels):
        n, dim, num_rows, num_cols = x.shape
        griding_num = dim - 1
        l = labels.data.cpu().numpy()
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)
        # 获得标签中车道线的上下限
        lane_label_all = []
        for k in range(labels.shape[0]):
            lane_label_img = []
            for i in range(labels.shape[2]):
                lane_label_top_perlane = []
                lane_label_down_perlane = []
                for j in range(labels.shape[1] - 1):
                    if l[k, 0, i] < griding_num and lane_label_top_perlane == []:
                        lane_label_top_perlane.append(j)
                    if l[k, j, i] == griding_num and l[k, j + 1, i] < griding_num and lane_label_top_perlane == []:
                        lane_label_top_perlane.append(j + 1)

                    if l[k, j, i] < griding_num and l[k, j + 1, i] == griding_num and lane_label_down_perlane == []:
                        lane_label_down_perlane.append(j)
                        break
                    if l[k, labels.shape[1] - 1, i] < griding_num and lane_label_down_perlane == []:
                        lane_label_down_perlane.append(labels.shape[1] - 1)
                if lane_label_top_perlane == [] or lane_label_down_perlane == []:
                    lane_label_top_perlane.append(0)
                    lane_label_down_perlane.append(0)
                lane_label_perlane = [int(lane_label_top_perlane[0]), int(lane_label_down_perlane[0])]
                lane_label_img.append(lane_label_perlane)
            lane_label_all.append(lane_label_img)
        flag = 0
        loss = 0
        for k in range(labels.shape[0]):
            loss_k = 0
            label_top = []
            label_down = []
            for l in lane_label_all[k]:
                label_top.append(l[0])
                label_down.append(l[1])
            for j in range(4):
                # 全局约束
                diff_list1 = []
                for i in range(label_top[j], label_down[j]):
                    diff_list1.append(pos[k, i, j] - pos[k, i + 1, j])
                loss1 = 0
                for i in range(len(diff_list1) - 1):
                    loss1 += self.l1(diff_list1[i], diff_list1[i + 1])
                if len(diff_list1) <= 1:
                    loss1 = 0
                else:
                    loss1 /= len(diff_list1) - 1
                loss_k += loss1
            loss += loss_k / 8
            if loss_k:
                flag += 1
        if flag:
            loss = loss / flag
        else:
            loss = 0
        return loss


class ParsingRelationLines(nn.Module):
    def __init__(self):
        super(ParsingRelationLines, self).__init__()
        self.l1 = torch.nn.L1Loss()

    def forward(self, x, labels):
        n, dim, num_rows, num_cols = x.shape
        griding_num = dim - 1
        l = labels.data.cpu().numpy()
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        pos = torch.sum(x * embedding, dim=1)
        # 获得标签中车道线的上下限
        lane_label_all = []
        for k in range(labels.shape[0]):
            lane_label_img = []
            for i in range(labels.shape[2]):
                lane_label_top_perlane = []
                lane_label_down_perlane = []
                for j in range(labels.shape[1] - 1):
                    if l[k, 0, i] < griding_num and lane_label_top_perlane == []:
                        lane_label_top_perlane.append(j)
                    if l[k, j, i] == griding_num and l[k, j + 1, i] < griding_num and lane_label_top_perlane == []:
                        lane_label_top_perlane.append(j + 1)

                    if l[k, j, i] < griding_num and l[k, j + 1, i] == griding_num and lane_label_down_perlane == []:
                        lane_label_down_perlane.append(j)
                        break
                    if l[k, labels.shape[1] - 1, i] < griding_num and lane_label_down_perlane == []:
                        lane_label_down_perlane.append(labels.shape[1] - 1)
                if lane_label_top_perlane == [] or lane_label_down_perlane == []:
                    lane_label_top_perlane.append(0)
                    lane_label_down_perlane.append(0)
                lane_label_perlane = [int(lane_label_top_perlane[0]), int(lane_label_down_perlane[0])]
                lane_label_img.append(lane_label_perlane)
            lane_label_all.append(lane_label_img)
        # 计算相邻三条车道线关系
        flag = 0
        loss = 0
        for k in range(labels.shape[0]):
            loss_l = 0
            loss_r = 0
            weight_l = 0
            weight_r = 0
            left = 0
            right = 0
            diff_list_l = []
            diff_list_r = []
            label_top = []
            label_down = []
            for l in lane_label_all[k]:
                label_top.append(l[0])
                label_down.append(l[1])
            # 左边三条线
            top = max(label_top[0:3])
            down = min(label_down[0:3])
            if top < down:
                left = 1
                weight_l = down - top + 1
                for i in range(0, num_cols - 2):
                    diff_list_l.append(pos[k, top:down + 1, i] - pos[k, top:down + 1, i + 1])
                for i in range(len(diff_list_l) - 1):
                    loss_l += self.l1(diff_list_l[i], diff_list_l[i + 1]) / (down + 1 - top)
                loss_l /= len(diff_list_l) - 1
            # 右边三条线
            top = max(label_top[1:4])
            down = min(label_down[1:4])
            if top < down:
                right = 1
                weight_r = down - top + 1
                for i in range(1, num_cols - 1):
                    diff_list_r.append(pos[k, top:down + 1, i] - pos[k, top:down + 1, i + 1])
                for i in range(len(diff_list_r) - 1):
                    loss_r += self.l1(diff_list_r[i], diff_list_r[i + 1]) / (down + 1 - top)
                loss_r /= len(diff_list_r) - 1
            # 综合俩结果
            if left and right:
                loss_k = (loss_l * weight_l + loss_r * weight_r) / (2 * num_rows)
            else:
                loss_k = (loss_l * weight_l + loss_r * weight_r) / num_rows
            if loss_k:
                flag += 1
            loss += loss_k
        if flag != 0:
            loss = loss / flag
        else:
            loss = 0
        return loss
