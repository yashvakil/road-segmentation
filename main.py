import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tifffile as tif

from keras import backend as K
from keras.utils import print_summary
from os.path import join, exists
from os import makedirs
from config import args
from datahandler import load_satellite_images, gen_patches, shuffle_dataset
from capsNet import build_arch2
from osgeo import gdal, osr

if __name__ == "__main__":

    args.root_dir = 'SegCaps_FinalModel'

    args.dataset = join(args.root_dir, 'org-dataset')
    args.logs_dir = join(args.root_dir, 'logs')  # Path to store various logs during training stage
    args.models_dir = join(args.root_dir, 'models')  # Path to store the saved models
    args.weights_dir = join(args.root_dir, 'weights')  # Path to store weights calculated during training
    args.layers_dir = join(args.root_dir, 'layers')  # Path to store the output of each layers
    args.tests_dir = join(args.root_dir, 'tests')  # Path to store predicted and reconstructed values
    args.patch_dataset_dir = join(args.root_dir, 'patch-dataset')  # Path to store the patched dataset
    args.weights = 'model_19_01_25-18_10_09.h5'

    directories = ["models_dir", "logs_dir", "weights_dir", "layers_dir", "patch_dataset_dir", "tests_dir"]
    for d in directories:
        print(getattr(args, d))
        makedirs(str(getattr(args, d)), exist_ok=True)

    makedirs(join(args.tests_dir, 'Predicted'), exist_ok=True)

    if not (exists(join(args.patch_dataset_dir, "satellite_images.npy"))
            and exists(join(args.patch_dataset_dir, "road_images.npy"))):
        if not (exists(join(args.dataset, "satellite_image_whole.npy"))
                and exists(join(args.dataset, "road_image_whole.npy"))):
            satellite_image_whole, road_image_whole = load_satellite_images(args.dataset, save=True,
                                                                            verbose=args.debug)
        else:
            satellite_image_whole = np.load(join(args.dataset, "satellite_image_whole.npy"))
            road_image_whole = np.load(join(args.dataset, "road_image_whole.npy"))

        satellite_images, road_images = gen_patches(satellite_image_whole, road_image_whole, patch_size=256,
                                                    stride=128, aug_times=0, save=True,
                                                    save_at=args.patch_dataset_dir, verbose=args.debug)
    else:
        satellite_images = np.load(join(args.patch_dataset_dir, "satellite_images.npy"))
        road_images = np.load(join(args.patch_dataset_dir, "road_images.npy"))

        if args.debug:
            print("=" * 19 + "PATCHED IMAGES" + "=" * 19)
            print("Satellite".ljust(20) + "|" + str(satellite_images.shape))
            print("Road".ljust(20) + "|" + str(road_images.shape))
            print("=" * 52)

    road_image_whole = tif.imread(join(args.dataset, "road.tif"))
    predicted_image = np.zeros(shape=road_image_whole.shape)
    
    predicted_image = np.expand_dims(predicted_image, axis=-1)
    print(predicted_image.shape)

    tif.imsave(join(args.tests_dir, "predicted.tif"), predicted_image)

    x_test = satellite_images
    y_test = road_images

    _, eval_model = build_arch2(x_test.shape[1:])
    print_summary(model=eval_model, positions=[.38, .65, .75, 1.])

    eval_model.load_weights(join(args.models_dir, args.weights))
    print("Model loaded from", join(args.models_dir, args.weights))

    predicted,_ = eval_model.predict(x_test, batch_size=10, verbose=1)
    np.save(join(args.tests_dir, "predicted_"+str(args.time)+".npy"), predicted)
    print("Predicted")
    print(predicted.shape)

    #for i in range(x_test.shape[0]):
        #tif.imsave(join(args.tests_dir, "Predicted", "pred_" + str(i) + ".tif"), predicted[i])
    

    print("Combining Image")
    predicted = predicted[:, 64:-64, 64:-64, :]
    print(predicted.shape)

    patch_size = predicted.shape[1]
    k=0
    for i in range(64, predicted_image.shape[0]-patch_size+1-64, patch_size):
        for j in range(64, predicted_image.shape[1]-patch_size+1-64, patch_size):
            predicted_image[i:i + patch_size, j:j + patch_size,:] = predicted[k]
            if k%1000 == 0:
                print(k)
            k = k+1

    tif.imsave(join(args.tests_dir, "predicted.tif"), predicted_image)
    np.save(join(args.tests_dir, "predicted_"+str(args.time)+".npy"), predicted)

    refrenced_image = gdal.Open(join(args.dataset, 'road.tif'))
    
    if args.debug:
        print("="*51)
        print("GEO REFRENCING IMAGE")
        print("="*51)

    driver = gdal.GetDriverByName('GTiff')
    to_refrence_image = driver.CreateCopy(join(args.tests_dir, 'predicted_refrenced.tif'), to_refrence_image, 1)
    refrenced_image = driver.CreateCopy(join(args.test_dir, 'original_refrenced.tif'), refrenced_image, 1)

    proj = osr.SpatialReference(wkt=refrenced_image.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY', 1)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    dest_wkt = srs.ExportToWkt()
    gt = np.asarray(refrenced_image.GetGeoTransform()).astype(np.float32)

    to_refrence_image.SetGeoTransform(gt)
    to_refrence_image.SetProjection(dest_wkt)
    to_refrence_image.FlushCache()
    refrenced_image.FlushCache()
    to_refrence_image = None
    refrenced_image = None

