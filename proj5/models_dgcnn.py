import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    # x: (B, F, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # broadcasting (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx

def get_graph_feature(x, k=20):
    # x: (B, F, N)
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k)  # (B, N, k)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points #(B, 1, 1)

    idx = idx + idx_base # broadcasted to (B, N, k), values in [0, B*N)
    idx = idx.view(-1) #(B*N*k, )

    x = x.transpose(2, 1).contiguous()  # (B, N, F)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (B*N*k, F)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (B, N, k, F)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (B, N, k, F)

    edge_feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()  # (B, 2F, N, k)
    return edge_feature

class DGCNN_cls(nn.Module):
    def __init__(self, num_classes=3, k=20):
        super(DGCNN_cls, self).__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))

        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(128)
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))

        self.bn4 = nn.BatchNorm2d(256)
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))

        self.bn5 = nn.BatchNorm1d(1024)
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(2048, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1)  #(B, 3, N)

        x1 = get_graph_feature(x, k=self.k)  #(B, 6, N, k)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=-1, keepdim=False)[0]  #(B, 64, N)

        x2 = get_graph_feature(x1, k=self.k) #(B, 128, N, k)
        x2 = self.conv2(x2) #(B, 64, N, k)
        x2 = x2.max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=-1, keepdim=False)[0]  # (B, 256, N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 512, N)
        x = self.conv5(x_cat)  # (B, 1024, N)

        x1 = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)  # (B, 1024)
        x2 = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)  # (B, 1024)
        x = torch.cat((x1, x2), dim=1)  # (B, 2048)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x) # (B, num_classes)

        return F.log_softmax(x, dim=1) #(B, num_classes)

class DGCNN_seg(nn.Module):
    def __init__(self, num_seg_classes=6, k=20):
        super(DGCNN_seg, self).__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))

        self.conv5 = nn.Sequential(nn.Conv1d(320, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(2048 + 320, 512, 1, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2))
        self.dp1 = nn.Dropout(0.5)
        self.conv7 = nn.Sequential(nn.Conv1d(512, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.dp2 = nn.Dropout(0.5)
        self.conv8 = nn.Conv1d(256, num_seg_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (B, 3, N)

        x1 = get_graph_feature(x, k=self.k)
        x1 = self.conv1(x1).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x2 = get_graph_feature(x1, k=self.k)
        x2 = self.conv2(x2).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x3 = get_graph_feature(x2, k=self.k)
        x3 = self.conv3(x3).max(dim=-1, keepdim=False)[0]  # (B, 64, N)

        x4 = get_graph_feature(x3, k=self.k)
        x4 = self.conv4(x4).max(dim=-1, keepdim=False)[0]  # (B, 128, N)

        x_cat = torch.cat((x1, x2, x3, x4), dim=1)  # (B, 320, N)
        x_global = self.conv5(x_cat)  # (B, 1024, N)

        x_max = F.adaptive_max_pool1d(x_global, 1).repeat(1, 1, x.size(2)) #(B, 1024, N)
        x_avg = F.adaptive_avg_pool1d(x_global, 1).repeat(1, 1, x.size(2)) #(B, 1024, N)
        x_global = torch.cat([x_max, x_avg], dim=1)  # (B, 2048, N)

        x = torch.cat([x_global, x_cat], dim=1)  # (B, 2048+320, N)
        x = self.conv6(x)
        x = self.dp1(x)
        x = self.conv7(x)
        x = self.dp2(x)
        x = self.conv8(x)  # (B, num_seg_classes, N)

        return x.permute(0, 2, 1)  # (B, N, num_seg_classes)        