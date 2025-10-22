from copy import deepcopy
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import numpy as np
import scanpy as sc
import scipy as sp


class transformation():
    def __init__(self,
                 cell_profile):
        self.cell_profile = deepcopy(cell_profile)
        self.gene_num = len(self.cell_profile)

    def build_mask(self, masked_percentage):
        return np.random.rand(self.gene_num) < masked_percentage

    def random_mask(self, mask_percentage, apply_mask_prob):

        s = np.random.uniform(0, 1)
        if s < apply_mask_prob:
            # create the mask for mutation
            mask = self.build_mask(mask_percentage)

            # do the mutation with prob
            self.cell_profile[mask] = 0.0

    def random_gaussian_noise(self, noise_percentage, sigma, apply_noise_prob):

        s = np.random.uniform(0, 1)
        if s < apply_noise_prob:
            # create the mask for mutation
            mask = self.build_mask(noise_percentage)

            # create the noise
            noise = np.random.normal(0, sigma, mask.sum())

            # do the mutation
            self.cell_profile[mask] += noise

    def ToTensor(self):
        self.cell_profile = torch.from_numpy(self.cell_profile)


def RandomTransform(sample, args_transformation):

    tr = transformation(sample)
    # Mask
    tr.random_mask(args_transformation['mask_percentage'], args_transformation['apply_mask_prob'])
    # (Add) Gaussian noise
    tr.random_gaussian_noise(args_transformation['noise_percentage'], args_transformation['sigma'],
                             args_transformation['apply_noise_prob'])
    tr.ToTensor()

    return tr.cell_profile


class CustomDataset(Dataset):
    def __init__(self, data, transform_args=None):
        """
        Args:
            data: Raw data
            label: Data labels
            transform_args: Parameters required for transformation
        """
        self.data = data
        self.transform_args = transform_args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Apply data augmentation
        sample_w = RandomTransform(sample, self.transform_args[0])
        sample_s = RandomTransform(sample, self.transform_args[1])

        return [idx, sample, sample_w, sample_s]

def convert_type2label(types, type_to_label_dict):
    """
    Convert types to labels
    INPUTS:
        types-> list of types
        type_to_label dictionary-> dictionary of cell types mapped to numerical labels

    RETURN:
        labels-> list of labels
    """
    types_array = np.array(types)
    vectorized_lookup = np.vectorize(type_to_label_dict.get)
    labels = vectorized_lookup(types_array).astype(np.int64)
    return labels


def init_weights(model):
    """
    Initializing the weights of a model with Xavier uniform
    INPUTS:
        model -> a pytorch model which will be initilized with xavier weights

    RETURN:
        the updated weights of the model
    """

    if isinstance(model, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(model.weight)
        model.bias.data.fill_(0.02)

def to_onehot(label, num_classes):
    identity = torch.eye(num_classes).to(label.device)
    onehot = torch.index_select(identity, 0, label)
    return onehot


def normalize(adata, copy=True, highly_genes=None, filter_min_counts=True, size_factors=True, normalize_input=True,
              logtrans_input=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6:  # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes is not None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes,
                                    subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

def ContrastiveLoss(x1, x2, temperature):
    device = x1.device
    batch_size = x1.shape[0]
    z_i = F.normalize(x1.float(), dim=1)
    z_j = F.normalize(x2.float(), dim=1)

    representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)  # simi_mat: (2*bs, 2*bs)

    sim_ij = torch.diag(similarity_matrix, batch_size)
    sim_ji = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).to(device).float()
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    return torch.sum(loss_partial) / (2 * batch_size)


def get_centers(net=None, data=None, labels=None, dataloader=None, num_classes=0):
    centers = 0
    device = next(net.parameters()).device
    refs = torch.LongTensor(range(num_classes)).unsqueeze(1).to(device)
    if dataloader is None:
        feature, output = net(data)

        label = labels.unsqueeze(0).expand(num_classes, -1)
        mask = (label == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
        feature = feature.unsqueeze(0)
        # update centers
        centers += torch.sum(feature * mask, dim=1)

    else:
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            data = inputs.to(device)
            labels = labels.to(device)

            feature, output = net(data)

            labels = labels.unsqueeze(0).expand(num_classes, -1)
            mask = (labels == refs).unsqueeze(2).type(torch.cuda.FloatTensor)
            feature = feature.unsqueeze(0)
            # update centers
            centers += torch.sum(feature * mask, dim=1)

    return centers
