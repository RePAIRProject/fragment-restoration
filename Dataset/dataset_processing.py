import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import shutil
from segments import SegmentsClient, SegmentsDataset
from segments.utils import export_dataset


def collect_unique_pixel_values(folder_path):
    unique_values_set = set()
    all_files = os.listdir(folder_path)
    ground_truth_files = [file for file in all_files if file.endswith("ground-truth_semantic.png")]

    for file in ground_truth_files:
        file_path = os.path.join(folder_path, file)
        img = Image.open(file_path)
        bitmap_array = np.array(img)

        unique_values = np.unique(bitmap_array)
        unique_values_set.update(unique_values)

    unique_values_array = np.array(list(unique_values_set)).reshape(1, -1)
    return unique_values_array

def visualize_segmentation_mask(mask,n_classes):
    mask = np.array(cv2.imread(mask, 0))
    cmap = plt.cm.get_cmap('tab20',n_classes)
    plt.imshow(mask, cmap=cmap, vmin=0, vmax=n_classes)
    plt.colorbar(ticks=range(n_classes))
    plt.show()


def remap_mask(mask, mapping):
    remapped_mask = np.zeros_like(mask)
    for old_class, new_class in mapping.items():
        remapped_mask[mask == old_class] = new_class
    return remapped_mask

def export_segments_ai():

    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient('9dfbd86d1236dcc33e99909b304919f5b1580698')
    release = client.get_release('sinemaslan/repair_fragments_patterns-clone', 'v2') # Alternatively: release = 'flowers-v1.0.json'
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

    # Export to semantic format
    export_dataset(dataset, export_format='semantic')

    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient('9dfbd86d1236dcc33e99909b304919f5b1580698')
    release = client.get_release('lucap/repair_fragments_patterns', 'v2') # Alternatively: release = 'flowers-v1.0.json'
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

    # Export to semantic format
    export_dataset(dataset, export_format='semantic')


    # Initialize a SegmentsDataset from the release file
    client = SegmentsClient('9dfbd86d1236dcc33e99909b304919f5b1580698')
    release = client.get_release('UNIVE/decor2', 'v2') # Alternatively: release = 'flowers-v1.0.json'
    dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

    # Export to semantic format
    export_dataset(dataset, export_format='semantic')

def merge_segments_folders(images_output_path,masks_output_path):
    # Merge all masks from 3 different folders into the same folder
    folder_paths = ['/home/sinem/PycharmProjects/fragment-restoration/Dataset/segments/UNIVE_decor2/v2/',
                    '/home/sinem/PycharmProjects/fragment-restoration/Dataset/segments/lucap_repair_fragments_patterns/v2/',
                    '/home/sinem/PycharmProjects/fragment-restoration/Dataset/segments/sinemaslan_repair_fragments_patterns-clone/v2/']


    os.makedirs(images_output_path, exist_ok=True)
    os.makedirs(masks_output_path, exist_ok=True)

    for folder_path in folder_paths:
        all_files = os.listdir(folder_path)

        image_files = [file for file in all_files if
                       not (file.endswith("ground-truth_semantic.png") or file.endswith("ground-truth.png"))]
        mask_files = [file for file in all_files if file.endswith("ground-truth_semantic.png")]
        prefix = "Decor1_" if "lucap_repair_fragments_patterns" in folder_path or "sinemaslan_repair_fragments_patterns-clone" in folder_path else "Decor2_"

        for file in image_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(images_output_path, prefix + file)
            shutil.copy(src_path, dst_path)

        for file in mask_files:
            src_path = os.path.join(folder_path, file)
            dst_path = os.path.join(masks_output_path, prefix + file)
            shutil.copy(src_path, dst_path)

def relabel_decor2(folder_path):


    all_files = os.listdir(folder_path)
    semantic_files = [file for file in all_files if file.endswith("ground-truth_semantic.png")]

    for file in semantic_files:
        file_path = os.path.join(folder_path, file)
        img = Image.open(file_path)
        bitmap_array = np.array(img)

        temp_array = np.copy(bitmap_array)

        temp_array[bitmap_array == 1] = 4
        temp_array[bitmap_array == 4] = 12
        temp_array[bitmap_array == 2] = 13
        temp_array[bitmap_array == 3] = 14

        modified_img = Image.fromarray(temp_array)
        modified_img.save(file_path)

def create_combined_mask(foreground_mask, mask, new_class_label=1):
    white_pixels_in_mask = (mask > 0)
    combined_mask = np.where(white_pixels_in_mask, 0, foreground_mask)
    combined_mask = np.where((mask == 0) & (foreground_mask > 0), new_class_label, combined_mask)
    combined_mask = np.where(mask > 0, mask, combined_mask)

    return combined_mask

def check_images_are_rgb(folder_path):
    image_files = os.listdir(folder_path)

    for file in image_files:
        file_path = os.path.join(folder_path, file)
        img = Image.open(file_path)

        if img.mode != 'RGB':
            print(f"{file} is not a 3-channel RGB image.")
            return False
    print("All images are 3-channel RGB images.")
    return True

# Create the foreground masks
def create_fg(input_images,output_images,fg_folder):

    image_files = os.listdir(input_images)

    # Create foreground masks
    os.makedirs(fg_folder, exist_ok=True)
    os.makedirs(output_images, exist_ok=True)

    for file in image_files:
        print(file)
        file_path = os.path.join(input_images, file)
        #print(file_path)
        img2 = plt.imread(file_path)
        # Get the size of the RGBA image
        width, height, rd = np.shape(img2)

        # Create a numpy ones matrix with the same size
        mask = np.ones((height, width), dtype=np.uint8)*255

        if img2.shape[2] == 4:
            fg = np.bitwise_and(img2[:,:,3] > 0, mask)
            logical_fg = img2[:,:,3] > 0
        else:
            fg = np.bitwise_and(img2[:,:,0] > 0, mask)
            logical_fg = img2[:,:,0] > 0

        img2 = np.array(img2[:, :, :3])
        mask = np.repeat(logical_fg[:, :, np.newaxis], 3, axis=2)
        masked_image = mask * img2

        plt.imsave(os.path.join(output_images, f"{file[:-4]}.png"), masked_image)
        plt.imsave(os.path.join(fg_folder, f"{file[:-4]}_fg.png"), fg, cmap='gray')

        """
        img = Image.open(file_path)
        if img.mode == 'RGBA':
            background = Image.new('RGB', img.size, (0, 0, 0))
            background.paste(img, mask=img.split()[3])
    
            background.save(file_path, 'PNG')
        """

def refine_masks(input_masks,output_masks,fg_masks):

    '''0 background
    1 fragment_background (this is the fragment region)
    2 bluebird, (animal_blue_bird)
    3 yellowbird, (animal_yellow_bird)
    4 redgriffon
    5 red flower, (flower_red)
    6 blue flower,  (flower_blue)
    7 red_circle, (red_dot)
    8 red spiral, (red_spiral_pattern)
    9 curved green stripe (curved_green_line)
    10 thin red stripe, (thin_straight_red_line)
    11 thick red stripe,   (thick_straight_red_line)
    12 thin floral stripe, (yd_small_flower)
    13 thick floral stripe (yd_big_flower)'''

    mapping = {
        0: 0,  # Background
        1: 8,  # red spiral
        2: 15,  # dummy number
        3: 6,  # blue flower
        4: 10,  # thin red stripe
        5: 2,  # bluebird
        6: 4,  # red griffon
        7: 9,  # curved green stripe
        8: 7,  # red_circle
        9: 15,  # dummy number
        10: 5,  # red flower
        11: 3,  # yellow bird
        12: 11,  # thick red stripe
        13: 13,  # thick floral stripe
        14: 12,  # thin floral stripe
    }

    mask_files = os.listdir(input_masks)
    fg_files = os.listdir(fg_masks)

    # Create output masks folder
    os.makedirs(output_masks, exist_ok=True)

    for mfile in mask_files:
        print(mfile)
        m_file_path = os.path.join(input_masks, mfile)
        # print(file_path)
        bitmap_array = np.array(cv2.imread(m_file_path, 0))
        remapped_mask = remap_mask(bitmap_array, mapping)

        fg_file_path = os.path.join(fg_masks, f"{mfile[:-32]}_fg.png")
        print(fg_file_path)
        # print(file_path)
        fg_mask = np.array(cv2.imread(fg_file_path, 0))
        combined_mask = create_combined_mask(fg_mask, remapped_mask)
        m_file_path = os.path.join(output_masks, mfile)
        cv2.imwrite(m_file_path, combined_mask)

def plot_ds_statistics():
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Pie chart for number of images starting with "Decor1" and "Decor2"
    labels = ['Decor1', 'Decor2']
    sizes = [decor1_images, decor2_images]

    fig1, ax1 = plt.subplots()
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels,
                                       autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, (p / 100) * sum(sizes)),
                                       startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Images from "Decor1" and "Decor2"')
    plt.setp(autotexts, size=8, weight='bold')
    plt.savefig('images_decor1_decor2_pie_chart.png')
    plt.show()

    # Bar plot for number of images containing each pixel value
    pixel_values = list(pixel_value_counts.keys())
    counts = list(pixel_value_counts.values())

    fig2, ax2 = plt.subplots()
    colors = sns.color_palette("husl", len(pixel_values))
    ax2.bar(pixel_values, counts, color=colors)
    plt.title('Distribution of image count over the different pixel classes')
    plt.ylabel('Image count per class')
    plt.xlabel('Pixel Class Labels')
    plt.xticks(pixel_values)

    # Add the legend
    legend_labels = {
        1: "(1) red_spiral_pattern",
        2: "(2) flower_blue",
        3: "(3) thin_straight_red_lins",
        4: "(4) animal_blue_bird",
        5: "(5) curved_green_line",
        6: "(6) red_dot",
        7: "(7) horse",
        8: "(8) flower_red",
        9: "(9) animal_yellow_bird",
        10: "(10) thick_straight_red_line",
        11: "(11) yd_big_flower",
        12: "(12) yd_small_flower"
    }

    # Create custom handles for the legend, skipping pixel value 0
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[np.where(np.array(pixel_values) == pv)[0][0]], label=legend_labels[pv])
        for pv in pixel_values if pv in legend_labels
    ]
    ax2.legend(handles=handles, title='Motif categories', loc='best')

    plt.savefig('images_pixel_values_bar_plot_with_colored_legend.png')
    plt.show()
#----------------------------------------------------
# Export repositories from segments.ai
#export_segments_ai()

#----------------------------------------------------
# Change mask labels of Decor 2 repository, overwrite by changed masks in decor2 repository in segments folder
#folder_path = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/segments/UNIVE_decor2/v2/'
#relabel_decor2(folder_path)

#----------------------------------------------------
# Merge exported repositories into images_from_segments and masks_from_segments folders
segments_images_path = '/Dataset/segments_images'
segments_masks_path = '/Dataset/segments_masks'
merge_segments_folders(segments_images_path,segments_masks_path)

#----------------------------------------------------
# Create foreground masks for the fragment region, and save them into fg folder. Assign 0 to the background of all images
#images_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/images/'
fg_folder = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/fg/'
#create_fg(segments_images_path,images_folder,fg_folder)

#----------------------------------------------------
# Refine masks
output_masks = '/home/sinem/PycharmProjects/fragment-restoration/Dataset/masks/'
refine_masks(segments_masks_path,output_masks,fg_folder)