
_base_ = "base_dino_detr_ssod_coco_full.py"


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="/root/paddlejob/workspace/env_run/output/temp/data/coco/annotations/instances_train2017.json",
            img_prefix="/root/paddlejob/workspace/env_run/output/temp/data/coco/train2017/",

        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="/root/paddlejob/workspace/env_run/output/temp/data/coco/annotations/instances_unlabeled2017.json",
            img_prefix="/root/paddlejob/workspace/env_run/output/temp/data/coco/unlabeled2017/"

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
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
        unsup_weight=2.0,
    ),
    test_cfg=dict(inference_on="student"),
)

custom_hooks = [
    dict(type="NumClassCheckHook"),
    dict(type="MeanTeacher", momentum=0.999, interval=1, warm_up=0),
    dict(type='StepRecord', normalize=False),
]

runner = dict(_delete_=True, type="IterBasedRunner", max_iters=240000)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/coco_full"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type='TensorboardLoggerHook')
    ],
)
