import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss import chamfer_distance

# define losses
def occ_loss(src, tgt, use_logit=True):
	# voxel_src: b*num_samples x 1
	# voxel_tgt: b*num_samples x 1
	# loss = 
	# implement some loss for binary voxel grids
	tgt=tgt.float()
	if use_logit:
		#bce_loss = torch.nn.BCEWithLogitsLoss()  # Handles sigmoid + BCE	
		#loss = bce_loss(src, tgt) #used by model_occnet.py

		bce_loss = torch.nn.BCEWithLogitsLoss()  
		loss = bce_loss(torch.clamp(src, min=-10, max=10), tgt)  # Clamp logits, used by model_occnet2.py	
	else:
		loss = torch.nn.functional.binary_cross_entropy(src, tgt)
	return loss

def voxel_loss(voxel_src,voxel_tgt, use_logit=True):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	voxel_tgt=voxel_tgt.float()
	if use_logit:
		bce_loss = torch.nn.BCEWithLogitsLoss()  # Handles sigmoid + BCE	
		loss = bce_loss(voxel_src, voxel_tgt)
	else:
		loss = torch.nn.functional.binary_cross_entropy(voxel_src, voxel_tgt)
	return loss

def chamfer_loss_official(point_cloud_src,point_cloud_tgt):
	#print(f"use chamfer_loss_official.")
	loss, _ = chamfer_distance(point_cloud_src, point_cloud_tgt)
	return loss

def chamfer_loss_test2(point_cloud_src, point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	p1_dists, _, _ = knn_points(point_cloud_src, point_cloud_tgt)
	p2_dists, _, _ = knn_points(point_cloud_tgt, point_cloud_src)
	# implement chamfer loss from scratch
	loss_chamfer = torch.sum((p1_dists + p2_dists)) / (point_cloud_src.shape[0] * point_cloud_src.shape[1])
	return loss_chamfer

def chamfer_loss_test(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	
	point_cloud_src_cpy, point_cloud_tgt_cpy = point_cloud_src, point_cloud_tgt
	src = knn_points(point_cloud_src, point_cloud_tgt)
	tgt = knn_points(point_cloud_tgt, point_cloud_src)
	#loss_chamfer = torch.mean(torch.sum(src.dists[..., 0].sum(1) + tgt.dists[..., 0].sum(1)))
	loss_chamfer = torch.mean(src.dists[..., 0].sum(1) + tgt.dists[..., 0].sum(1))
	return loss_chamfer

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
    # Find the nearest neighbor in the target for each source point (k=1)
	dist_src2tgt, dist_src2tgt_idx, _ = knn_points(point_cloud_src, point_cloud_tgt, K=1)

    # Find the nearest neighbor in the source for each target point (k=1)
	dist_tgt2src, dist_tgt2src_idx, _ = knn_points(point_cloud_tgt, point_cloud_src, K=1)

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
