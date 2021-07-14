import glob
import os
import shutil

'''
    Kitti Depth Annotated
    ---------------------

    File Struct
    -----------
    tgt_img_dir ref_img_0_dir ref_img_2_dir annotated_depth_img_dir
    tgt_img_dir ref_img_0_dir ref_img_2_dir annotated_depth_img_dir
    tgt_img_dir ref_img_0_dir ref_img_2_dir annotated_depth_img_dir
    ....
    tgt_img_dir ref_img_0_dir ref_img_2_dir annotated_depth_img_dir
'''

def kitti_depth_annotated_prepare(kitti_dir, Test=False):

    # get annotated images
    depth_dir = kitti_dir + 'data_depth_annotated/train/*_sync'
    folders = sorted(glob.glob(depth_dir))
    
    file = open("./splits/kitti_depth_annotated_train_files.txt","w")

    for folder in folders:
        date  = folder[-26:-16]
        drive = folder[-26: -1] + 'c'
        
        annotated_dir = folder + '/proj_depth/groundtruth/image_02/*.png'
        annotated_depth_img_dir = sorted(glob.glob(annotated_dir))

        img_dir = sorted(glob.glob(kitti_dir + date + '/' + drive + '/image_02/data/*.png'))

        if img_dir:
            for ann_img in annotated_depth_img_dir:

                img_indx       = int(ann_img[-14:-4])
                tgt_img_indx   = int(img_dir[img_indx][-14:-4])
                ref_img_0_indx = int(img_dir[img_indx - 1][-14:-4])
                ref_img_2_indx = int(img_dir[img_indx + 1][-14:-4])
                
                if img_indx == tgt_img_indx and (img_indx - 1) == ref_img_0_indx  \
                                            and (img_indx + 1) == ref_img_2_indx:
                    file.write(img_dir[img_indx] + ' ')
                    file.write(img_dir[img_indx - 1] + ' ')
                    file.write(img_dir[img_indx + 1] + ' ')
                    file.write(ann_img + '\n')  
            break


if __name__ == "__main__":

    kitti_depth_annotated_prepare('../KITTI/')
    




