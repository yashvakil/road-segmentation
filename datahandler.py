import tifffile as tif
import numpy as np

from os.path import join, exists
from os import makedirs
from config import args
from skimage.transform import resize
from typing import Union, Tuple, List


def load_satellite_images(path: str, save: bool = False, verbose: bool = False) -> Union[np.ndarray, np.ndarray]:
    """ Load the images form tif file

    :param path:
    :param save:
    :param verbose:

    :return:  An ndarray of images that were loaded
    """
    if verbose:
        print("=" * 19 + "LOADING IMAGES" + "=" * 19)

    satellite_image_whole = tif.imread(join(path, "satellite.tif"))
    satellite_image_whole = satellite_image_whole[:,:,0:3]

    road_image_whole = tif.imread(join(path, "road.tif"))

    if road_image_whole.ndim <= 2:
        road_image_whole = np.expand_dims(road_image_whole, axis=-1)

    if verbose:
        print("=" * 19 + "IMAGES LOADED" + "=" * 19)
    # assert satellite_image_whole.shape[:-2] == road_image_whole[:-2], "Images are not of the same shape"

    if verbose:
        print("=" * 20 + "IMAGES SHAPE" + "=" * 20)
        print("Satellite ".ljust(20) + "|" + str(satellite_image_whole.shape))
        print("Road".ljust(20) + "|" + str(road_image_whole.shape))
        print("=" * 52)

    satellite_image_whole[satellite_image_whole < 0] = 0
    road_image_whole[road_image_whole != 1] = 0

    band_means = np.mean(satellite_image_whole,axis=(0,1))
    band_stds = np.std(satellite_image_whole,axis=(0,1))

    print("Before: ",band_means, band_stds)

    satellite_image_whole = satellite_image_whole - band_means
    satellite_image_whole = satellite_image_whole/band_stds
    
    if verbose:
        print("="*5 + "="*10 + "MEAN" + "="*10 + "="*10 + "STD" + "="*10)
        for i in range(band_stds.shape[0]):
            print(str(i).ljust(3) + "| " + str(band_means[i]).ljust(20) + "| " + str(band_stds[i]).ljust(20))
        print("="*52)

    np.save(join(path, "band_means.npy"), band_means)
    np.save(join(path, "band_stds.npy"), band_stds)

    print("After: ", np.mean(satellite_image_whole,axis=(0,1)),np.std(satellite_image_whole,axis=(0,1)))

    if save:
        np.save(join(path, "satellite_image_whole.npy"), satellite_image_whole)
        np.save(join(path, "road_image_whole.npy"), road_image_whole)
        if verbose:
            print("=" * 50)
            print("Images array saved at " + str(path))
            print("=" * 50)

    return satellite_image_whole, road_image_whole


def gen_patches(image: np.ndarray, label: np.ndarray, patch_size: int = 512, stride: int = 512,
                aug_times: int = 0, save: bool = False, save_at: str = "", verbose: bool = False):
    """ Generate square patches of images

    :param image:  The ndarray image of which patches hs to be found
    :param label: The ndarray label of the image to create patches
    :param patch_size: The length of the single side of a patch
    :param stride: Distance between each batch
    :param aug_times: The number of rotated patches to augment
    :param save: Boolean value that specifies whether to save or not the images
    :param save_at: Path to save the images, works only if "param:save" is True
    :param verbose: Print statements for debug

    :return: A List of patches of the image, and
            A List of patches if any of the label
    """
    h_scaled, w_scaled = image.shape[:2]
    image_patches = []
    label_patches = []
    count = 0

    if verbose:
        print("=" * 17 + "GENERATING PATCHES" + "=" * 17)

    if save:
        makedirs(join(save_at, "Satellite"), exist_ok=True)
        makedirs(join(save_at, "Road"), exist_ok=True)

        # Extracting Patches
        for i in range(0, h_scaled - patch_size + 1, stride):
            for j in range(0, w_scaled - patch_size + 1, stride):
                x = image[i:i + patch_size, j:j + patch_size, :]
                y = label[i:i + patch_size, j:j + patch_size, :]

                if True:
                    image_patches.append(x)
                    label_patches.append(y)

                    if save:
                        tif.imsave(join(save_at, "Satellite", str(count) + ".tif"), x)
                        tif.imsave(join(save_at, "Road", str(count) + ".tif"), y)
                    count = count + 1

                    # Augmenting data
                    for k in range(0, aug_times):
                        rand_num = np.random.randint(0, 8)
                        augx = data_aug(x, mode=rand_num, verbose=verbose)
                        augy = data_aug(y, mode=rand_num, verbose=verbose)

                        image_patches.append(augx)
                        label_patches.append(augy)
                        if save:
                            tif.imsave(join(save_at, "Satellite", str(count)+".tif"), augx)
                            tif.imsave(join(save_at, "Road", str(count)+".tif"), augy)
                        count = count + 1

        image_patches = np.array(image_patches)
        label_patches = np.array(label_patches)

        if verbose:
            print("=" * 17 + "PATCHES GENERATED" + "=" * 17)
        # assert image_patches.shape[:-2] == label_patches.shape[:-2], "Generated patches are not of the same shape"

        if save:
            np.save(join(save_at, "satellite_images.npy"), image_patches)
            np.save(join(save_at, "road_images.npy"), label_patches)
            if verbose:
                print("=" * 51)
                print("Patched images save at " + str(save_at))
                print("=" * 51)

        if verbose:
            print("=" * 19 + "PATCHED IMAGES" + "=" * 19)
            print("Satellite".ljust(20) + "|" + str(image_patches.shape))
            print("Road".ljust(20) + "|" + str(label_patches.shape))
            print("=" * 52)

    return image_patches, label_patches


def data_aug(image: np.ndarray, mode: int = 0, verbose: bool = False) -> np.ndarray:
    """ Return a translated image

    :param image: The ndarray of the image to translate
    :param mode: The type of translation
    :param verbose: Print statements for debug

    :return: The ndarray of translated image
    """
    if mode == 0:
        return image
    elif mode == 1:
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))


def shuffle_dataset(items: Tuple, verbose: bool = False) -> Tuple:
    """ Load a data set from the given path into a ndarray

    :param items: The tuple of items to shuffle
    :param verbose:
    :return: A tuple of shuffled items
    """
    shuffled = []
    items = list(items)
    permutation = np.random.permutation(items[0].shape[0])
    for i in range(len(items)):
        temp = items[i]
        shuffled.append(temp[permutation])
    if verbose:
        print("=" * 17 + "SHUFFLED DATASET" + "=" * 17)

    return tuple(shuffled)