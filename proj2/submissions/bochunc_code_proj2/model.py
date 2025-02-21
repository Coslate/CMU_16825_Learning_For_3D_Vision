from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        self.alpha = 0.1
        self.dropout_p = 0.15
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # TODO:
            self.layer_project = torch.nn.Sequential(
                nn.Linear(512, 512*4*4*4),
                nn.LeakyReLU(negative_slope=self.alpha, inplace=True)
                #torch.nn.ReLU(inplace=True)
            )

            # I' = (I-1)*S - 2P + F
            self.layer_decoder = nn.Sequential(
                nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False), #8
                nn.BatchNorm3d(256),
                nn.LeakyReLU(negative_slope=self.alpha, inplace=False),

                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False), #16
                nn.BatchNorm3d(128),
                nn.LeakyReLU(negative_slope=self.alpha, inplace=False),

                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False), #32
                nn.BatchNorm3d(64),
                nn.LeakyReLU(negative_slope=self.alpha, inplace=False),

                nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1, bias=False), #32
                nn.BatchNorm3d(32),
                nn.LeakyReLU(negative_slope=self.alpha, inplace=False),

                # Added Dropout to prevent overfitting
                nn.Dropout3d(self.dropout_p),

                # Additional Conv3D for sharper final output
                nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(16),
                nn.LeakyReLU(negative_slope=self.alpha, inplace=False),                

                nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(8),
                nn.LeakyReLU(negative_slope=0.1, inplace=False),                

                # Final prediction layer with larger kernel size for better voxel refinement
                nn.ConvTranspose3d(8, 1, kernel_size=3, stride=1, padding=1, bias=False), #32
                nn.Sigmoid()  # Output is probability for voxel presence
            )

        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            self.fc_layers = torch.nn.Sequential(
                torch.nn.Linear(512, 1024),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(1024, self.n_point),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.n_point, self.n_point*2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(self.n_point*2, self.n_point*3),
                torch.nn.Tanh()
            )
            # self.decoder =             
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)

            self.mesh_vert_size = mesh_pred.verts_packed().shape[0]
            # TODO:
            # self.decoder =             
            self.decoder = nn.Sequential(
                nn.Linear(512, 1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 2048),
                nn.LeakyReLU(),
                nn.Linear(2048, self.mesh_vert_size * 3)
            ) 

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # Expand feature vector to 3D shape
            x = self.layer_project(encoded_feat)
            x = x.view(B, 512, 4, 4, 4)  # Reshape to a small 3D volume

            # Decode to 3D voxel grid
            voxels_pred = self.layer_decoder(x)
            return voxels_pred
        elif args.type == "point":
            # TODO:
            x = self.fc_layers(encoded_feat)
            pointclouds_pred = x.view(B, self.n_point, 3)  # Reshape into (B, n_point, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat) #(B, V*3)
            deform_vertices_pred = deform_vertices_pred.view(-1, self.mesh_vert_size*3) #(B, V*3)
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          

