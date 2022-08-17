##################################################################
##################### EXP: fuxun-mmrazor #########################
##################################################################


## id: silver_star, 0.422
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_gfl_r101_fpn_gfl_r50_fpn_1x_coco.py --launcher pytorch" \
# --num_nodes 1



## id: epic_map, 0.374
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco.py --launcher pytorch" \
# --num_nodes 1



## id: joyful_net, 0.381
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-r50_1x \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-r50_1x_coco.py --launcher pytorch" \
# --num_nodes 1



## id: clever_door, 0.434
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw.py --launcher pytorch" \
# --num_nodes 1


## id: mighty_crayon
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-r50_1x_allfpn \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-r50_1x_coco_allfpn.py --launcher pytorch" \
# --num_nodes 1



## id: boring_plane
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn.py --launcher pytorch" \
# --num_nodes 1




# # id: khaki_shelf, 0.373
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-r50_1x_allfpn_iclr \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-r50_1x_coco_allfpn_iclr.py --launcher pytorch" \
# --num_nodes 1



# # id: orange_skin
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclr \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclr.py --launcher pytorch" \
# --num_nodes 1


# # id: affable_kiwi, 0.379
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclr1e2 \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclr1e2.py --launcher pytorch" \
# --num_nodes 1


# # id: icy_diamond
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclr1e3 \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclr1e3.py --launcher pytorch" \
# --num_nodes 1


# # id: quirky_picture
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclr_nofeatloss \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclr.py --launcher pytorch" \
# --num_nodes 1


# # id: quirky_rabbit
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclr_nochannelloss \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclr.py --launcher pytorch" \
# --num_nodes 1



# # id: sincere_ring
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclr_nospatialloss \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclr.py --launcher pytorch" \
# --num_nodes 1


# # id: happy_chain
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclr_noadaptationlayer \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclr.py --launcher pytorch" \
# --num_nodes 1



# # id: calm_car
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclrdyhead \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclrdyhead.py --launcher pytorch" \
# --num_nodes 1



# # id: icy_oxygen
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-t_1x_adamw \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-l-t_1x_coco_adamw.py --launcher pytorch" \
# --num_nodes 1


# # id: ashy_vulture
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-t_1x_adamw_allfpn \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-l-t_1x_coco_adamw_allfpn.py --launcher pytorch" \
# --num_nodes 1


# # id: tender_parcel
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-t_1x_adamw_2xbs \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-l-t_1x_coco_adamw_2xbs.py --launcher pytorch" \
# --num_nodes 1


# # id: serene_square
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s_1x_adamw \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-l-s_1x_coco_adamw.py --launcher pytorch" \
# --num_nodes 1


# # id: plucky_shark
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclrdyhead_channelonly \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclrdyhead.py --launcher pytorch" \
# --num_nodes 1


# # id: green_neck
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_pcc \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_pcc.py --launcher pytorch" \
# --num_nodes 1


# # id: kind_nut
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-s-t_1x_adamw_allfpn_iclrdyhead_channelonly_x10 \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_swin-s-t_1x_coco_adamw_allfpn_iclrdyhead.py --launcher pytorch" \
# --num_nodes 1



# # id: stoic_vinegar
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s-t_1x_adamw_mt \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_l_s_t_05x_coco_adamw_mt.py --launcher pytorch" \
# --num_nodes 1



# # id: lemon_ballon
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s-t_1x_adamw_st1 \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_l_s_t_05x_coco_adamw_st1.py --launcher pytorch" \
# --num_nodes 1



# # id: nice_river
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s-t_1x_adamw_st2 \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_l_s_t_05x_coco_adamw_st2.py --launcher pytorch" \
# --num_nodes 1



# # id: patient_rhythm
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s-t_1x_adamw_mt_1bs \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_l_s_t_05x_coco_adamw_mt.py --launcher pytorch" \
# --num_nodes 1


# # id: silver_cord
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s-t_1x_adamw_mt_1bs_dropout \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_l_s_t_05x_coco_adamw_mt_dropout.py --launcher pytorch" \
# --num_nodes 1


# # id: frosty_soursop
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s-t_1x_adamw_mt_1bs_mask \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_l_s_t_05x_coco_adamw_mt_mask.py --launcher pytorch" \
# --num_nodes 1



# # id: plucky_insect
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-swin-l-s-t_1x_adamw_mt_2bs_mask \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_l_s_t_05x_coco_adamw_mt_mask.py --launcher pytorch" \
# --num_nodes 1







# # id: musing_bridge
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-dino-s-r50_1x \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_dino-s-r50_1x_coco.py --launcher pytorch" \
# --num_nodes 1




# # id: tidy_celery
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-dinos-swint_1x \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_dinos-swint_1x_coco.py --launcher pytorch" \
# --num_nodes 1



# # id: busy_button
# python tools/azureml/aml_submit.py \
# --workspace /home/frank/.azureml/config-v100.json \
# --exp_name fuxun-mmrazor  \
# --environment /home/frank/.azureml/environment.json-mmrazor \
# --datastore /home/frank/.azureml/ds_coco.json \
# --compute_target v100x4 \
# --input_dir yue \
# --output_dir fuxun/output/fuxun-mmrazor-cwd-dinos-swins_1x_05bs \
# --cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_dinos-swins_1x_coco.py --launcher pytorch" \
# --num_nodes 1



# id: 
python tools/azureml/aml_submit.py \
--workspace /home/frank/.azureml/config-v100.json \
--exp_name fuxun-mmrazor  \
--environment /home/frank/.azureml/environment.json-mmrazor \
--datastore /home/frank/.azureml/ds_coco.json \
--compute_target v100x4 \
--input_dir yue \
--output_dir fuxun/output/fuxun-mmrazor-cwd-dino-s-r50_1x_05weight \
--cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_dino-s-r50_1x_coco.py --launcher pytorch" \
--num_nodes 1




# id: 
python tools/azureml/aml_submit.py \
--workspace /home/frank/.azureml/config-v100.json \
--exp_name fuxun-mmrazor  \
--environment /home/frank/.azureml/environment.json-mmrazor \
--datastore /home/frank/.azureml/ds_coco.json \
--compute_target v100x4 \
--input_dir yue \
--output_dir fuxun/output/fuxun-mmrazor-cwd-dinos-swint_1x_05weight \
--cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_dinos-swint_1x_coco.py --launcher pytorch" \
--num_nodes 1



# id: 
python tools/azureml/aml_submit.py \
--workspace /home/frank/.azureml/config-v100.json \
--exp_name fuxun-mmrazor  \
--environment /home/frank/.azureml/environment.json-mmrazor \
--datastore /home/frank/.azureml/ds_coco.json \
--compute_target v100x4 \
--input_dir yue \
--output_dir fuxun/output/fuxun-mmrazor-cwd-dinos-swins_1x_05bs_05weight \
--cmd "tools/mmdet/train_mmdet.py ./configs/distill/cwd/cwd_cls_head_dinos-swins_1x_coco.py --launcher pytorch" \
--num_nodes 1