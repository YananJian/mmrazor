_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_1x.py',
    '../../_base_/mmdet_runtime.py'
]

student = dict(
    type='mmdet.MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        ),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5))
    )

teacher_ckpt = 'https://github.com/Mrxiaoyuer/coco_checkpoints/releases/download/v0.0.3/dinos_536.pth'  # noqa

teacher = dict(
    type='mmdet.DINO',
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt),
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DINOHead',
        num_query=900,
        num_classes=80,
        num_feature_levels=4,
        in_channels=2048,  # TODO
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(
            type='CdnQueryGenerator',
            noise_scale=dict(label=0.5, box=1.0),  # 0.5, 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        transformer=dict(
            type='DinoTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_levels=4,
                        dropout=0.0),  # 0.1 for DeformDETR
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DinoTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0),  # 0.1 for DeformDETR
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=4,
                            dropout=0.0),  # 0.1 for DeformDETR
                    ],
                    feedforward_channels=2048,  # 1024 for DeformDETR
                    ffn_dropout=0.0,  # 0.1 for DeformDETR
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            temperature=20,
            normalize=True),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=300))


# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                assist_module='transformer.encoder.layers.0.attentions.0.attention_weights',
                teacher_module='bbox_head.transformer.encoder.layers.0.attentions.0.attention_weights',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_encoder_0_attention_weights',
                        tau=1,
                        reduction='sum',
                        loss_weight=1,
                    )
                ]),

            dict(
                assist_module='transformer.encoder.layers.1.attentions.0.attention_weights',
                teacher_module='bbox_head.transformer.encoder.layers.1.attentions.0.attention_weights',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_encoder_1_attention_weights',
                        tau=1,
                        reduction='sum',
                        loss_weight=1,
                    )
                ]),

            dict(
                assist_module='transformer.encoder.layers.2.attentions.0.attention_weights',
                teacher_module='bbox_head.transformer.encoder.layers.2.attentions.0.attention_weights',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_encoder_2_attention_weights',
                        tau=1,
                        reduction='sum',
                        loss_weight=1,
                    )
                ]),
            dict(
                assist_module='transformer.encoder.layers.3.attentions.0.attention_weights',
                teacher_module='bbox_head.transformer.encoder.layers.3.attentions.0.attention_weights',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_encoder_3_attention_weights',
                        tau=1,
                        reduction='sum',
                        loss_weight=1,
                    )
                ]),
            dict(
                assist_module='transformer.encoder.layers.4.attentions.0.attention_weights',
                teacher_module='bbox_head.transformer.encoder.layers.4.attentions.0.attention_weights',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_encoder_4_attention_weights',
                        tau=1,
                        reduction='sum',
                        loss_weight=1,
                    )
                ]),
            dict(
                assist_module='transformer.encoder.layers.5.attentions.0.attention_weights',
                teacher_module='bbox_head.transformer.encoder.layers.5.attentions.0.attention_weights',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_encoder_5_attention_weights',
                        tau=1,
                        reduction='sum',
                        loss_weight=1,
                    )
                ]),
        ],
        assist = True,
        assist_loss_mul = 0.1,
        assist_module=dict(
            student_module='rpn_head',  # assist head will take inputs as student rpn_head inputs, which is student fpn outputs
            teacher_module='bbox_head', # assist head will copy from bbox_head from teacher module
        ),
     ),
)

find_unused_parameters = True

optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=2.5, norm_type=2))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)

# log_config = dict(interval=1, hooks=[ dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook') ])
