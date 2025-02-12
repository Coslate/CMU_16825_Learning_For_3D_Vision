import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	bce_loss = torch.nn.BCEWithLogitsLoss()  # Handles sigmoid + BCE	
	loss = bce_loss(voxel_src, voxel_tgt.float())
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
    # Find the nearest neighbor in the target for each source point (k=1)
	dist_src2tgt, dist_src2tgt_idx, _ = knn_points(point_cloud_src, point_cloud_tgt, K=1)
	dist_src2tgt.squeeze(0)
	dist_src2tgt.squeeze(2)

    # Find the nearest neighbor in the source for each target point (k=1)
	dist_tgt2src, dist_tgt2src_idx, _ = knn_points(point_cloud_tgt, point_cloud_src, K=1)
	dist_tgt2src.squeeze(0)
	dist_tgt2src.squeeze(2)

    # Compute Chamfer Loss (mean of squared distances)
	loss_chamfer = dist_src2tgt.mean() + dist_tgt2src.mean()
	return loss_chamfer	

'''Nearly same result as mesh_laplacian_smoothing'''
'''But the formula of the Loss function is different'''
'''
def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	verts = mesh_src.verts_packed()            
	edges = mesh_src.edges_packed()            

    # Get the vertices for both ends of each edge
	v_start = verts[edges[:, 0]]            
	v_end = verts[edges[:, 1]]              

	# Smoothness loss: Mean squared difference between connected vertices
	loss_laplacian = ((v_start - v_end) ** 2).sum(dim=1).mean()
	return loss_laplacian
'''
def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	return loss_laplacian
