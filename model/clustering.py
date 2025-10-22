import torch
from torch.nn import functional as F
from .utilities import to_onehot
from scipy.optimize import linear_sum_assignment
from math import ceil

class DIST(object):
    def __init__(self, dist_type='cos'):
        self.dist_type = dist_type

    def get_dist(self, pointA, pointB, cross=False):
        return getattr(self, self.dist_type)(pointA, pointB, cross)

    def cos(self, pointA, pointB, cross):
        pointA = F.normalize(pointA, dim=1)
        pointB = F.normalize(pointB, dim=1)

        if not cross:
            return 0.5 * (1.0 - torch.sum(pointA * pointB, dim=1))
        else:
            assert (pointA.size(1) == pointB.size(1))
            return 0.5 * (1.0 - torch.matmul(pointA, pointB.transpose(0, 1)))


class Clustering(object):
    def __init__(self, eps, max_len, dist_type='cos', device="cpu"):
        self.eps = eps
        self.Dist = DIST(dist_type)
        self.samples = {}
        self.path2label = {}
        self.center_change = None
        self.stop = False
        self.max_len = max_len
        self.device = device
        self.centers = None
        self.max_iters = 100 # Maximum iterations for clustering centers

    def set_init_centers(self, init_centers=None):
        '''
        Set the initial cluster centers
        '''
        self.centers = init_centers
        self.init_centers = init_centers
        self.num_classes = self.centers.size(0)

    def random_init_centers(self, num_classes):
        """
        Randomly initialize cluster centers
        """
        self.num_classes = num_classes

        features = self.samples['feature']
        if len(features) >= num_classes:
            # Randomly select samples as the center
            indices = torch.randperm(len(features))[:num_classes]
            init_centers = features[indices]
        else:
            # If the samples are insufficient, use a random normal distribution
            init_centers = torch.randn(num_classes, features.size(1), device=self.device)
        self.set_init_centers(init_centers)

    def clustering_stop(self, centers):
        if centers is None:
            self.stop = False
        else:
            dist = self.Dist.get_dist(centers, self.centers)
            dist = torch.mean(dist, dim=0)
            self.stop = dist.item() < self.eps

    def assign_labels(self, feats):
        dists = self.Dist.get_dist(feats, self.centers, cross=True)
        _, labels = torch.min(dists, dim=1)
        return dists, labels

    def align_centers(self):
        cost = self.Dist.get_dist(self.centers, self.init_centers, cross=True)
        cost = cost.data.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def collect_samples(self, net, loader):
        device = next(net.parameters()).device
        data_feat, data_gt, data_u, data_w, data_s = [], [], [], [], []
        for batch_idx, (inputs_u, inputs_u_w, inputs_u_s, label_u) in enumerate(loader):
            inputs_u = inputs_u.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)
            label_u = label_u.to(device)

            data_u += [inputs_u.to(device)]
            data_w += [inputs_u_w.to(device)]
            data_s += [inputs_u_s.to(device)]
            data_gt += [label_u.to(device)]

            feature, output = net(inputs_u)
            data_feat += [feature]

        self.samples['data'] = torch.cat(data_u, dim=0)
        self.samples['data_w'] = torch.cat(data_w, dim=0)
        self.samples['data_s'] = torch.cat(data_s, dim=0)
        self.samples['gt'] = torch.cat(data_gt, dim=0) if len(data_gt) > 0 else None
        self.samples['feature'] = torch.cat(data_feat, dim=0)

    def feature_clustering(self, net=None, data=None, loader=None, num_classes=0):
        centers = None
        self.stop = False

        if loader is None:
            device = next(net.parameters()).device
            self.samples['feature'], _ = net(data.to(device))
        else:
            self.collect_samples(net, loader)
        feature = self.samples['feature']
        if self.centers is None:
            self.random_init_centers(num_classes)

        refs = torch.LongTensor(range(self.num_classes)).unsqueeze(1).to(self.device)
        num_samples = feature.size(0)
        num_split = ceil(1.0 * num_samples / self.max_len)

        # K-means iterative clustering continues until convergence or the maximum number of iterations is reached.
        epoch = 0
        while True:
            self.clustering_stop(centers)
            if centers is not None:
                self.centers = centers
            if self.stop or epoch > self.max_iters:
                break
            epoch += 1

            centers = 0
            count = 0
            start = 0
            for N in range(num_split):
                cur_len = min(self.max_len, num_samples - start)
                cur_feature = feature.narrow(0, start, cur_len)
                dist2center, labels = self.assign_labels(cur_feature)
                labels_onehot = to_onehot(labels, self.num_classes)
                count += torch.sum(labels_onehot, dim=0)
                labels = labels.unsqueeze(0)
                mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
                reshaped_feature = cur_feature.unsqueeze(0)
                # update centers
                centers += torch.sum(reshaped_feature * mask, dim=1)
                start += cur_len

            mask = (count.unsqueeze(1) > 0).type(torch.cuda.FloatTensor)
            centers = mask * centers + (1 - mask) * self.init_centers

        # After clustering is complete, assign final labels to samples
        dist2center, labels = [], []
        start = 0
        count = 0
        for N in range(num_split):
            cur_len = min(self.max_len, num_samples - start)
            cur_feature = feature.narrow(0, start, cur_len)
            cur_dist2center, cur_labels = self.assign_labels(cur_feature)

            labels_onehot = to_onehot(cur_labels, self.num_classes)
            count += torch.sum(labels_onehot, dim=0)

            dist2center += [cur_dist2center]
            labels += [cur_labels]
            start += cur_len

        self.samples['p_label'] = torch.cat(labels, dim=0)
        self.samples['dist2center'] = torch.cat(dist2center, dim=0)

        cluster2label = self.align_centers()
        # reorder the centers
        self.centers = self.centers[cluster2label, :]
        # re-label the data according to the index
        num_samples = len(self.samples['feature'])
        for k in range(num_samples):
            self.samples['p_label'][k] = cluster2label[self.samples['p_label'][k]].item()

        del self.samples['feature']
