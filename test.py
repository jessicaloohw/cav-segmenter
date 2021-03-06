import os
import warnings
import skimage.io
import numpy as np
import tensorflow as tf

import network

def main():

    ################################### USER INPUT #####################################################

    MODEL_DIR = './models'
    IMAGE_FILENAME = './path/to/image/filename.tif'
    CAV_SEGMENTER_NUMBER = 1    # 1, 2
    K_FOLD_NUMBER = 1           # 1, 2, 3, 4, 5, 6, 7, 8, 9

    #################################### SETTINGS ######################################################

    # Check user-input:
    if not os.path.exists(MODEL_DIR):
        print('Invalid MODEL_DIR.')
        return
    if not os.path.exists(IMAGE_FILENAME):
        print('Invalid IMAGE_FILENAME.')
        return
    if not (CAV_SEGMENTER_NUMBER == 1 or CAV_SEGMENTER_NUMBER == 2):
        print('Invalid CAV_SEGMENTER_NUMBER.')
        return
    if K_FOLD_NUMBER < 1 or K_FOLD_NUMBER > 9:
        print('Invalid K_FOLD_NUMBER.')
        return

    # Image:
    IMAGE_HEIGHT = 496
    IMAGE_WIDTH = 32
    OVERLAP_SIZE = 31
    THRESHOLD = 0.5
    MAX_SLICES = 350    # Change according to the size of GPU

    ###################################################################################################

    # Ignore these specific warnings:
    warnings.filterwarnings("ignore", message="Matplotlib is currently using agg")
    warnings.filterwarnings("ignore", message="Anti-aliasing will be enabled by default in skimage 0.15")

    # Placeholders:
    image_batch = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    # Network:
    logits = network.segment(image_batch)

    # Session configuration:
    sessConfig = tf.ConfigProto()
    sessConfig.allow_soft_placement = True
    sessConfig.gpu_options.allow_growth = True

    # Saver:
    saver = tf.train.Saver(max_to_keep=0)

    # Session:
    with tf.Session(config=sessConfig) as sess:

        # Load weights:
        SEGMENTATION_MODEL_PATH = os.path.join(MODEL_DIR, 'cav-segmenter{}'.format(CAV_SEGMENTER_NUMBER), 'model-{}'.format(K_FOLD_NUMBER))
        saver.restore(sess, SEGMENTATION_MODEL_PATH)
        print('Network restored: cav-segmenter{}/model-{}.'.format(CAV_SEGMENTER_NUMBER, K_FOLD_NUMBER))

        # Load image:
        image = skimage.io.imread(IMAGE_FILENAME)
        if len(image.shape) > 2:
            print('Error: Only grayscale images allowed.')
            return
        image = image.astype(np.float32)
        image = image[..., None]
        
        # Get mean and std:
        meanval, stdval = network.get_mean_and_std(K_FOLD_NUMBER)

        # Normalise and slice:
        image_slices, slice_idxs = network.slice_image(((image - meanval) / stdval), IMAGE_WIDTH, OVERLAP_SIZE)

        # Split into batches:
        num_slices = image_slices.shape[0]
        num_batches = int(np.ceil(num_slices/MAX_SLICES))

        # Initialise:
        logits_val = np.zeros([0, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        # Run in batches:
        for b in range(num_batches):

            # Get batch:
            start_b = b*MAX_SLICES
            end_b = (b+1)*MAX_SLICES
            if end_b > num_slices:
                end_b = num_slices
            image_slices_b = image_slices[start_b:end_b, :, :, :]

            # Get prediction:
            logits_val_b = sess.run(logits, feed_dict={image_batch: image_slices_b})

            # Concatenate:
            logits_val = np.concatenate((logits_val, logits_val_b), axis=0)

        # Probability:
        prob_val = logits_val[:, :, :, 0]

        # Stitch:
        stitched_prob = network.stitch_images(prob_val, slice_idxs, image.shape[1])

        # Prediction:
        stitched_pred = (stitched_prob > THRESHOLD)

        # Save prediction:
        SEG_SAVE_FILENAME = '{}_segmentation.{}'.format(os.path.splitext(IMAGE_FILENAME)[0], os.path.splitext(IMAGE_FILENAME)[1])
        skimage.io.imsave(SEG_SAVE_FILENAME, (255*stitched_pred).astype(np.uint8))

        print('Segmentation completed.')


if __name__ == '__main__':
    main()
