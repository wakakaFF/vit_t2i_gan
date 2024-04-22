import torch
import torch.nn as nn
from model.layers import *


class Encoder(nn.Module):
    def __init__(self, modelConfig):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(modelConfig['img_channel'], channels[0], 3, 1, 1)]
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i + 1]))
                resolution //= 2
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(Swish())
        layers.append(nn.Conv2d(channels[-1], modelConfig['latent_channel'], 3, 1, 1))
        self.model = nn.Sequential(*layers).cuda()

    def forward(self, x):
        return self.model(x)


# torch.Size([B, 64, 16, 16])

# modelConfig = {
#     "state": "train",  # train or test
#     # train_set
#     "epoch": 200,
#     "batch_size": 32,
#     "dataset": 'xray',
#     # img
#     "img_size": 256,
#     "img_channel": 3,
#     # report
#     "report_channel": 1,
#     "report_seq_length": 64,
#     "report_embedding_dim": 768,
#     # img_latent
#     "latent_channel": 4,
#     "latent_dim": 16,
#     # time_steps
#     "num_time_steps": 4,
#     # optimizer
#     "lr_g": 1e-3,
#     "lr_d": 1e-4,
#     "beta_min": 0.1,
#     "beta_max": 20,
#     "use_geometric": False,
#     # codebook
#     'num_codebook_vectors': 1024,
#     'beta': 0.1,
#     # gpt2
#     # vit
#     'work_channel': 330,  # 320+10
#     'vit_dim': 512,
#     'vit_channel': 64,
#     'vit_depth': 8,
#     'vit_transf_qkv_dim': 2048,
#     'vit_transf_dim': 2048,
#
#     # file
#     'dataset_root': 'D:/Code/data/Xray/Xray',
#     "save_weight_dir": "./Checkpoints_xray/",
#     "save_img": "./save_images",
#     "device": "cuda",
#     "training_load_weight": None,
#     "labels": ("persistent stable right basilar atelectasis . low lung volumes and patient rotation . given "
#                "differences in technique , heart size within normal limits . persistent right basilar opacity "
#                ", atelectasis . no suspicious pulmonary opacity , pneumothorax or definite pleural effusion . "
#                "mild degenerative change of the thoracic spine ."
#                "CXR562_IM-2163-1001.png	no acute or active cardiac , pulmonary or pleural disease . "
#                "frontal and lateral views of the chest show normal size and configuration of the cardiac "
#                "silhouette . normal mediastinal contour , pulmonary and vasculature , central airways and "
#                "lung volumes . no pleural effusion ."),
#     "test_load_weight": "ckpt_180_.pt", }
#
# img = torch.randn(5, 3, 256, 256)
#
# encoder = Encoder(modelConfig)
# x = encoder(img)
# print(x.shape)
