from sahi.slicing import slice_coco
'''
coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="../dataset/VisDrone_coco/annotations/instances_val2017.json",
    image_dir="../dataset/VisDrone_coco/val2017/",
    output_coco_annotation_file_name="annotations/instances_val2017.json",

    ignore_negative_samples=False,
    output_dir="../dataset/VisDrone_coco_sliced/val2017",
    slice_height=448,
    slice_width=448,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True
)
'''

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="../dataset/VisDrone_coco/annotations/instances_train2017.json",
    image_dir="../dataset/VisDrone_coco/train2017/",
    output_coco_annotation_file_name="annotations/instances_train2017.json",

    ignore_negative_samples=False,
    output_dir="../dataset/VisDrone_coco_sliced/train2017",
    slice_height=448,
    slice_width=448,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.1,
    verbose=True
)