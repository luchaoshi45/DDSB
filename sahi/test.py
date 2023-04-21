from sahi.slicing import *
'''
	480x360
	540x960
	765x1360
	788x1400
	1050x1400
	1078x1916
    1080x1920
'''

if __name__ == "__main__":
    image_height, image_width = 1080, 1920
    # image_height, image_width = 480, 360
    image_height, image_width = 540, 960
    slice_height, slice_width = 448, 448
    overlap_height_ratio, overlap_width_ratio = 0.2, 0.2
    re = get_slice_bboxes(image_height, image_width,
                          slice_height, slice_width,
                          overlap_height_ratio, overlap_width_ratio)
    print(re)