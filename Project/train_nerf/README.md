##  How to Run
#For Materials_Highres Scene:
#Use stratified sampling strategy
step1. In ./configs/nerf_materials_highres.yaml, set output_gif_file: ./images/part_4_1.nerf_materials_highres.use_views.gif
step2. In ./configs/nerf_materials_highres.yaml, set checkpoint_path: ./checkpoints.q4_1.nerf_materials_highres.use_views
step3. In ./configs/nerf_materials_highres.yaml, set use_views: True
step4. In ./configs/nerf_materials_highres.yaml, set num_epochs: 261
step5. Run: python volume_rendering_main.py --config-name=nerf_materials_highres
step6. Output: ./images/part_4_1.nerf_materials_highres.use_views.gif

#Use coarse_fine sampling strategy
#For Materials_Highres Scene:
step1. In ./configs/nerf_materials_highres.yaml, set output_gif_file: ./images/part_4_2.nerf_materials_highres.use_views.use_coarse_fine.gif
step2. In ./configs/nerf_materials_highres.yaml, set checkpoint_path: ./checkpoints.q4_2.nerf_materials_highres.use_views.use_coarse_fine
step3. In ./configs/nerf_materials_highres.yaml, set use_views: True
step4. In ./configs/nerf_materials_highres.yaml, set sampler.type: coarse_fine
step5. In ./configs/nerf_materials_highres.yaml, set sampler.use_fine_sampling: True
step6. In ./configs/nerf_materials_highres.yaml, set num_epochs: 201
step7. Run: python volume_rendering_main.py --config-name=nerf_materials_highres
step8. Output: ./images/part_4_2.nerf_materials_highres.use_views.use_coarse_fine.gif
