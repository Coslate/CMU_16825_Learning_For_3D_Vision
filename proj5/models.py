import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        #self.conv0 = nn.Conv1d(3, 64, 1)
        #self.bn0 = nn.BatchNorm1d(64)

        #self.conv1 = nn.Conv1d(64, 64, 1)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(256, num_classes)        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = points.permute(0, 2, 1)  # (B, 3, N)

        #x = F.relu(self.bn0(self.conv0(x)))  # (B, 64, N)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)
        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = F.relu(self.bn3(self.conv3(x)))  # (B, 1024, N)

        x = torch.max(x, 2)[0]  # Global max pooling: (B, 1024)

        x = F.relu(self.bn4(self.fc1(x)))    # (B, 512)
        x = F.relu(self.bn5(self.fc2(x)))    # (B, 256)
        x = self.dropout(x)
        x = self.fc3(x)                      # (B, num_classes)
        return F.log_softmax(x, dim=1)        


# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        self.conv0 = nn.Conv1d(3, 64, 1)
        self.bn0 = nn.BatchNorm1d(64)
        self.conv1 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv4 = nn.Conv1d(1088, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)

        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)

        self.conv6 = nn.Conv1d(256, 128, 1)
        self.bn6 = nn.BatchNorm1d(128)

        self.conv7 = nn.Conv1d(128, num_seg_classes, 1)        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        x = points.permute(0, 2, 1)  # (B, 3, N)

        x0 = F.relu(self.bn0(self.conv0(x)))  # (B, 64, N)
        x1 = F.relu(self.bn1(self.conv1(x0)))  # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))  # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))  # (B, 1024, N)

        x_global = torch.max(x3, 2, keepdim=True)[0]  # (B, 1024, 1)
        x_global = x_global.repeat(1, 1, x.size(2))   # (B, 1024, N)

        x_concat = torch.cat([x1, x_global], dim=1)  # (B, 64 + 1024 = 1088, N)

        x = F.relu(self.bn4(self.conv4(x_concat)))  # (B, 512, N)
        x = F.relu(self.bn5(self.conv5(x)))         # (B, 256, N)
        x = F.relu(self.bn6(self.conv6(x)))         # (B, 128, N)
        x = self.conv7(x)                           # (B, num_seg_classes, N)

        return x.permute(0, 2, 1)  # (B, N, num_seg_classes)        



