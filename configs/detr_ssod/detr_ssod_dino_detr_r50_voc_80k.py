_base_ = "base_dino_detr_ssod_voc.py"


classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
data = dict(
    samples_per_gpu=5,
    workers_per_gpu=5,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/voc07_trainval.json",
            img_prefix="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/",
            classes=classes,

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/voc12_trainval.json",
            img_prefix="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/",
            classes=classes,
        ),
    ),
    test=dict(
            type="CocoDataset",
            ann_file="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/voc07_test.json",
            img_prefix="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/",
            classes=classes,
    ),
    val=dict(
            type="CocoDataset",
            ann_file="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/voc07_test.json",
            img_prefix="/root/paddlejob/workspace/env_run/output/temp/data/VOCdevkit/",
            classes=classes,

    ),
    
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)


semi_wrapper = dict(
    type="DinoDetrSSOD",
    model="${model}",
    train_cfg=dict(
        use_teacher_proposal=False,
        pseudo_label_initial_score_thr=0.4,
        min_pseduo_box_size=0,
        unsup_weight=4.0,
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=80000)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
