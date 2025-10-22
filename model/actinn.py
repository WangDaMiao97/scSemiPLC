import torch
import torch.nn as nn


class ACTINN(nn.Module):
    def __init__(self, output_dim=None, input_size=None):
        # The Classifer class: We are developing a model similar to ACTINN for good accuracy
        if output_dim is None or input_size is None:
            raise ValueError('Must explicitly declare input dim (num features) and output dim (number of classes)')
        super(ACTINN, self).__init__()
        self.inp_dim = input_size
        self.out_dim = output_dim

        # feed forward layers
        self.feat_encoder = nn.Sequential(
            nn.Linear(self.inp_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, output_dim)
        )

    def forward(self, x):
        # Forward pass of the classifier
        fea = self.feat_encoder(x)
        out = self.classifier(fea)
        return fea, out

class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)


class Con_estimator(nn.Module): # 置信度评估器
    def __init__(self, NUM_CLASS):
        super(Con_estimator, self).__init__()
        self.l2norm = FeatureL2Norm()
        self.num_class = NUM_CLASS

        self.con_estimator = nn.Sequential(nn.Linear(50+self.num_class, 32),
                                           nn.ReLU(),
                                           nn.Linear(32, 8),
                                           nn.ReLU())  # sigmoid
        self.last_layer = nn.Linear(8, 1)

    def forward(self, est_input):
        con_output = self.con_estimator(est_input)
        con_output = self.last_layer(con_output)
        con_output = torch.sigmoid(con_output)

        return con_output.squeeze(1)
