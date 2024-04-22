import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 文本特征提取层
        self.text_conv1 = nn.Conv1d(in_channels=768, out_channels=64, kernel_size=3, padding=1)
        self.text_conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)

        # 图像特征提取层
        self.img_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.img_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # 全连接层
        self.fc1 = nn.Linear(32 * 128 * 128, 64)  # 修改全连接层的输入维度和输出维度
        self.fc2 = nn.Linear(64, 1)

    def forward(self, report, img):
        # 处理文本信息
        # report = report.permute(0, 2, 1)  # 调整维度
        # text_out = F.relu(self.text_conv1(report))
        # text_out = F.relu(self.text_conv2(text_out))
        # text_out = torch.mean(text_out, dim=2)  # 对文本特征进行池化

        # 处理图像信息
        img_out = F.relu(self.img_conv1(img))
        img_out = F.relu(self.img_conv2(img_out))
        img_out = F.max_pool2d(img_out, 2)  # 降低图像特征维度
        img_out = img_out.view(img_out.size(0), -1)  # 展平图像特征
        # 合并文本和图像特征
        # combined_features = torch.cat((text_out, img_out), dim=1)

        # 判别器输出
        discriminator_out = F.relu(self.fc1(img_out))
        discriminator_out = torch.sigmoid(self.fc2(discriminator_out))

        # out = torch.zeros_like(discriminator_out)
        #
        # out = discriminator_out.masked_fill(discriminator_out[:] > 0.7, float(1.0))
        # out = out.masked_fill(discriminator_out[:] <= 0.7, float(0.0))
        return discriminator_out
        # return discriminator_out

# latent_img = torch.randn((1, 3, 256, 256))
# report = torch.randn((1, 64, 768))
#
# A = TextImageDiscriminator()
# a = A(report, latent_img)
# print(a)
