#! /bin/csh -f

#Q1.1
python fit_data.py --type 'vox'

#Q1.2
python fit_data.py --type 'point' --max_iter 200000

#Q1.3
python fit_data.py --type 'mesh'

#Q2.1
#Vox Train
python3 train_model.py --type 'vox' --max_iter 30001 --save_freq 1000 --batch_size 96 --num_workers 14  --lr 1.8e-3  --use_cas 1 --cas_warmup_steps 700 --cas_min_lr 3e-4 --cas_final_lr 8e-5 --use_step_update 1 --output_path ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk

#Vox Eval
 python eval_model.py --type 'vox' --load_checkpoint --eval_chk_file ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk/checkpoint_vox.pth --output_eval_path ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk/

#Q2.2
python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 5000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

python eval_model.py --type 'point' --load_checkpoint --n_points 5000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

#Q2.3
python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

python eval_model.py --type 'mesh' --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

#Q2.5
#Point Cloud Train
python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 1000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_1000_bk

python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 2500 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_2500_bk

python train_model.py --type 'point' --save_freq 600 --lr_sch_on 1 --n_point 5000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 601 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

python train_model.py --type 'point' --save_freq 500 --lr_sch_on 1 --n_point 10000 --max_iter 8001 --lr 9e-5 --batch_size 32 --early_stop_iter 501 --output_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_10000_bk

#Point Cloud Eval
python eval_model.py --type 'point' --load_checkpoint --n_points 1000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_1000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_1000_bk/

python eval_model.py --type 'point' --load_checkpoint --n_points 2500 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_2500_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_2500_bk/

python eval_model.py --type 'point' --load_checkpoint --n_points 5000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk

python eval_model.py --type 'point' --load_checkpoint --n_points 10000 --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_10000_bk/checkpoint_point.pth --output_eval_path ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_10000_bk


#Mesh Train
python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --w_chamfer 0.5 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_0.5_lr_9e-5_lr_sch_on_1_loss_orig_bk

python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --w_chamfer 2 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_2_lr_9e-5_lr_sch_on_1_loss_orig_bk

python ./train_model.py --type 'mesh' --max_iter 4001 --batch_size 64 --num_workers 8 --save_freq 700 --early_stop_iter 701 --lr 9e-5 --lr_sch_on 1 --w_chamfer 5 --output_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_5_lr_9e-5_lr_sch_on_1_loss_orig_bk

#Mesh Eval
python eval_model.py --type 'mesh' --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_0.5_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_0.5_lr_9e-5_lr_sch_on_1_loss_orig_bk

python eval_model.py --type 'mesh' --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk

python eval_model.py --type 'mesh' --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_2_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_2_lr_9e-5_lr_sch_on_1_loss_orig_bk

python eval_model.py --type 'mesh'  --num_workers 8 --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_5_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_wchamfer_5_lr_9e-5_lr_sch_on_1_loss_orig_bk


#Q2.6
python ./vis_activation.py --type 'vox' --load_checkpoint --eval_chk_file ./outputs_max_iter10001_bz96_num_workers_14_lr_1.8e-3_use_cas_1_cas_warmup_steps_700_cas_min_lr_3e-4_cas_final_lr_8e_-5_use_step_update_1_bk/checkpoint_vox.pth --output_eval_path ./vis_saliency_map_vox
python ./vis_activation.py --type 'point' --load_checkpoint --eval_chk_file ./outputs_point_model4_max_iter8001_bz32_num_workers_8_lr_9e-5_lr_sch_on_0_n_points_5000_bk/checkpoint_point.pth --output_eval_path ./vis_saliency_map_point
python ./vis_activation.py --type 'mesh' --load_checkpoint --eval_chk_file ./outputs_mesh_model4_max_iter4001_bz64_num_workers_8_lr_9e-5_lr_sch_on_1_loss_orig_bk/checkpoint_mesh.pth --output_eval_path ./vis_saliency_map_mesh

#Q3.1
python3 train_model_occnet.py --type 'occ' --max_iter 10001 --early_stop_iter 3001 --save_freq 1000 --batch_size 96 --num_workers 12 --lr 2e-3  --use_cas 1 --cas_warmup_steps 1000 --cas_min_lr 3e-5 --cas_final_lr 2e-5 --use_step_update 1 --output_path ./outputs_occ_model1_max_iter10001_bz96_num_workers_12_lr_2e-3_use_cas_1_nsamplept_8192