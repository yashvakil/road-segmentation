import numpy as np
import math
import pandas

from matplotlib import pyplot as plt
from config import args
from os.path import join
from os import environ
environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


def plot_log(file_dir, file_name, save=False, verbose=False):
    data = pandas.read_csv(join(file_dir, file_name))

    if verbose:
        print("="*20 + "DATA KEYS" + "="*20)
        for key in data.keys():
            print(key)
        print("="*49)

    fig = plt.figure(figsize=(24, 12))
    fig.subplots_adjust(top=0.95, right=0.95, hspace=0.5, wspace=0.5)
    
    data["val_out_seg_dice_hard"] = data["val_out_seg_dice_hard_intersection"].values/data["val_out_seg_dice_hard_union"].values
    data["out_seg_dice_hard"] = data["out_seg_dice_hard_intersection"].values/data["out_seg_dice_hard_union"].values

    elements = ["epoch", "out_seg_dice_hard_intersection", "val_out_seg_dice_hard_intersection", "out_seg_dice_hard_union", "val_out_seg_dice_hard_union"]

    grid = "24"
    count = 1
    for key in data.keys():
        if key not in elements:
            fig.add_subplot(grid+str(count))
            plt.plot(data['epoch'].values, data[key].values, label=key)
            plt.legend()
            plt.title(key)
            count = count + 1
    plt.show()

    if save:
        fig.savefig(join(file_dir, 'logs_graph.jpg'))
        if verbose:
            print("="*51)
            print("Logs Graph saved at " + str(join(file_dir, "logs_graph.jpg")))
            print("="*51)


def combine_images(generated_images, height=None, width=None, save=False, save_at="combined_images.jpg", verbose=False):
    num = generated_images.shape[0]
    if width is None and height is None:
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
    elif width is not None and height is None:
        height = int(math.ceil(float(num) / width))
    elif height is not None and width is None:
        width = int(math.ceil(float(num) / height))

    shape = generated_images.shape[1:3]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i*shape[0]:(i + 1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[:, :, 0]

    if save:
        plt.imsave(str(save_at), image)
        if verbose:
            print("="*51)
            print("Combined image saved at " + str(save_at))
            print("="*51)

    return image


if __name__ == "__main__":
    plot_log(args.logs_dir, "epoch_logs_19_01_25-18_10_09.csv", save=True, verbose=args.debug)