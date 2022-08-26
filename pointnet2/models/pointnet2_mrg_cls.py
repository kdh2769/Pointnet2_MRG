import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops.pointnet2_modules import PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

class PointNet2ClassificationMRG(PointNet2ClassificationSSG):
    def _build_model(self):

        self.SA_branch1 = nn.ModuleList()
        self.SA_branch1.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[0,64,64,128],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_branch1.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128,128,256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_branch2 = nn.ModuleList()
        self.SA_branch2.append(
            PointnetSAModule(
                npoint=512,
                radius=0.4,
                nsample=128,
                mlp=[0,64,128,256],
                use_xyz=self.hparams["model.use_xyz"],
            )
        )
        self.SA_branch3 = nn.ModuleList() # SA(64, 128, 256, 512) 
        self.SA_branch3.append(
            PointnetSAModule(
                mlp=[0, 64, 128, 256, 512], 
                use_xyz=self.hparams["model.use_xyz"]
            )
        )
        self.SA_branch4 = nn.ModuleList()
        self.SA_branch4.append(
            PointnetSAModule(
                mlp=[256, 512, 1024], 
                use_xyz=self.hparams["model.use_xyz"]
            ))
        
        self.fc_layer = nn.Sequential(
            nn.Linear(1024 + 512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

    def forward(self, pointcloud):  
        xyz, features = self._break_up_pc(pointcloud)
        features = None # without normal
        xyz_branch12_list=[]
        features_branch12_list = []
        features_all_list = []

        xyz_branch1, features_branch1 = xyz, features
       
        for module in self.SA_branch1:  #input all original points 
            xyz_branch1, features_branch1 = module(xyz_branch1, features_branch1)
        features_branch12_list.append(features_branch1)
        xyz_branch12_list.append(xyz_branch1)

        xyz_branch2, features_branch2 = xyz, features
        for module in self.SA_branch2: # input all original points 
            xyz_branch2, features_branch2 = module(xyz_branch2, features_branch2)
        features_branch12_list.append(features_branch2)
        xyz_branch12_list.append(xyz_branch2)

        features_branch12_concat = torch.cat(features_branch12_list, dim=2)
        xyz_branch12_concat = torch.cat(xyz_branch12_list, dim=1)

        xyz_branch3, features_branch3 = xyz, features
        for module in self.SA_branch3: #all original points
            xyz_branch3, features_branch3 = module(xyz_branch3, features_branch3) # complete 
        features_all_list.append(features_branch3)

        features_branch4 = features_branch12_concat
        for module in self.SA_branch4: # branch1 + branch2 
            xyz_branch12_concat , features_branch4 = module(xyz_branch12_concat,features_branch4)
        features_all_list.append(features_branch4)

        features_all_concat = torch.cat(features_all_list, dim=1)

        return self.fc_layer(features_all_concat.squeeze(-1))