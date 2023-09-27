import cv2 
import numpy as np 
import os 
import argparse 
import pdb 
import json 

## mean iou computed only on the classes in classes_list
def mean_iou(preds, gt, classes_list=[]):
    if len(classes_list) > 0:
        ious = []
        for j, motif_class in enumerate(classes_list):
            motif_count = np.sum((gt == motif_class))
            if motif_count > 0:
                intersection_mc = np.sum((gt == motif_class) * (preds == motif_class))
                union_mc = np.sum(((gt == motif_class) + (preds == motif_class))>0)
                iou_mc = intersection_mc / (union_mc + 10**(-6))
                ious.append(iou_mc)
        if np.isclose(np.sum(ious), 0):
            mean_iou = 0
        else:
            mean_iou = np.mean(ious)
        return mean_iou
    else:
        return 0


## mean pixel accuracy computed only on the classes in classes_list
def mean_pixel_accuracy(preds, gt, classes_list=[]):
    if len(classes_list) > 0:
        pas = []
        for j, motif_class in enumerate(classes_list):
            motif_count = np.sum((gt == motif_class))
            if motif_count > 0:
                pa_mc = np.sum((preds==gt) * (gt == motif_class))
                pas.append(pa_mc / motif_count) 
        if np.isclose(np.sum(pas), 0):
            mean_pa = 0
        else:
            mean_pa = np.mean(pas)
        return mean_pa
    else:
        return 0

def remap_labels(image, num_classes):
    if num_classes == 14:
        return image  # no remapping needed for 14 classes
    elif num_classes == 13:
        remapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
             8: 7, 9: 8, 10: 9, 11:10, 12:11, 13:12} 
    elif num_classes == 12:
        remapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6,
             8: 7, 9: 8, 10: 9, 11:10, 12:11, 13:12}
    elif num_classes == 3:
        remapping = {
            0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2,
            8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2
        }
    else:
        raise ValueError("Invalid number of classes")

    for k, v in remapping.items():
        image[image == k] = v
    return image

def main(args):

    # predictions
    assert len(args.preds) > 1, 'Please specify the prediction folder (it should contain the images to evaluate them)'
    preds_folder = args.preds

    if preds_folder[0] != '/':
        preds_folder = os.path.join(os.getcwd(), preds_folder)

    # dataset (gt, test split)
    assert len(args.dataset) > 1, 'Please specify the dataset folder'
    assert os.path.exists(args.dataset), f'{args.dataset} does not exist!'
    
    # the test.txt file
    if args.testset == '':
        test_path = os.path.join(args.dataset, 'test.txt')
    else:
        test_path = args.t

    # the groundtruth masks
    if args.groundtruth == '':
        #groundtruth_dir = os.path.join(args.dataset, f'segmap{args.classes}c')
        groundtruth_dir = args.dataset
    else:
        groundtruth_dir = args.gt

    imgs_names = np.loadtxt(test_path, dtype=str)
    IoU = np.zeros(len(imgs_names))
    PA = np.zeros(len(imgs_names))
    if args.classes == 13 or args.classes == 12:
        IoU_motif = np.zeros(len(imgs_names))
        PA_motif = np.zeros(len(imgs_names))
    
    # evaluate on these classes
    classes_list = np.arange(args.classes)

    for j, img_name in enumerate(imgs_names):

        gt_image = cv2.imread(os.path.join(groundtruth_dir, img_name))
        gt_image = remap_labels(gt_image, args.classes)
        gt_resized = cv2.resize(gt_image, (args.size, args.size), interpolation=cv2.INTER_NEAREST)

        pred_image = cv2.imread(os.path.join(preds_folder, img_name))
        pred_image = remap_labels(pred_image, args.classes)
        pred_resized = cv2.resize(pred_image, (args.size, args.size), interpolation=cv2.INTER_NEAREST)
        
        IoU[j] = mean_iou(pred_resized, gt_resized, classes_list=classes_list)
        PA[j] = mean_pixel_accuracy(pred_resized, gt_resized, classes_list=classes_list)
        if args.classes == 13:
            IoU_motif[j] = mean_iou(pred_resized, gt_resized, classes_list=classes_list)
            PA_motif[j] = mean_pixel_accuracy(pred_resized, gt_resized, classes_list=classes_list)
        elif args.classes == 12:
            IoU_motif[j] = mean_iou(pred_resized, gt_resized, classes_list=classes_list[1:])
            PA_motif[j] = mean_pixel_accuracy(pred_resized, gt_resized, classes_list=classes_list[1:])            
    
    print('')
    print("#" * 30)
    print(f"Performances on {args.classes} classes")
    print(f"IoU (avg): {np.mean(IoU):.3f}")
    print(f"PA  (avg): {np.mean(PA):.3f}")
    if args.classes == 13:
        print("-" * 30)
        print("Performances on motif only")
        print(f"IoU (motif): {np.mean(IoU_motif):.3f}")
        print(f"PA  (motif): {np.mean(PA_motif):.3f}") 
    print("#" * 30)
    perf = {
        'IoU_avg': np.mean(IoU),
        'PA_avg': np.mean(PA)
    }
    if args.classes == 13:
        perf['IoU_motif'] = np.mean(IoU_motif)
        perf['PA_motif'] = np.mean(PA_motif)

    if args.output == '':
        output_dir = os.getcwd()
        output_path = os.path.join(output_dir, f'performances_{args.classes}classes.json')
    else:
        if os.path.isdir(args.output):
            output_dir = args.output
            output_path = os.path.join(output_dir, f'performances_{args.classes}classes.json')
        else:
            output_path = args.output
    
    with open(output_path, 'w') as opj: 
        json.dump(perf, opj, indent=3)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate the results on the MoFF dataset')
    parser.add_argument('-p', '--preds', type=str, default='', help='folder with predictions')
    parser.add_argument('-c', '--classes', type=int, default=3, help='number of classes (3 for scenario 1 and 13 for scenario 2)')
    parser.add_argument('-d', '--dataset', type=str, default='', help='MoFF folder')
    parser.add_argument('-t', '--testset', type=str, default='', help='the full path to the test.txt file (use only if not in the dataset folder)')
    parser.add_argument('-gt', '--groundtruth', type=str, default='', help='the full path to the folder with the groundtruth images (use only if not in the dataset folder)')
    parser.add_argument('-s', '--size', type=int, default=512, help='size of the images (the evaluation is done on the resized images)')
    parser.add_argument('-o', '--output', type=str, default='', help='output path (a folder where we save the file or a file path (ends with json))')
    args = parser.parse_args()
    main(args)
