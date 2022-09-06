##################################################################
##################### EXP: fuxun-mmrazor #########################
##################################################################


# id: 
python tools/azureml/aml_submit.py \
--workspace /home/frank/.azureml/config-v100.json \
--exp_name fuxun-mmrazor  \
--environment /home/frank/.azureml/environment.json-mmrazor \
--datastore /home/frank/.azureml/ds_coco.json \
--compute_target v100x4 \
--input_dir yue \
--output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_nodrop \
--cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_nodrop.py --launcher pytorch" \
--num_nodes 1


# id: 
python tools/azureml/aml_submit.py \
--workspace /home/frank/.azureml/config-v100.json \
--exp_name fuxun-mmrazor  \
--environment /home/frank/.azureml/environment.json-mmrazor \
--datastore /home/frank/.azureml/ds_coco.json \
--compute_target v100x4 \
--input_dir yue \
--output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-t_1x_adamw_nodrop \
--cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-l-t_1x_coco_adamw_nodrop.py --launcher pytorch" \
--num_nodes 1
