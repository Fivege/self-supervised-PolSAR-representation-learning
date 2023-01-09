import tensorflow as tf


def random_crop_flip_resize(image):
    # Random cropping
    #h_crop = tf.cast(tf.random.uniform(shape=[], minval=10, maxval=17, dtype=tf.int32), tf.float32)
    h_crop = tf.cast(tf.random.uniform(shape=[], minval=8, maxval=17, dtype=tf.int32), tf.float32)
    w_crop = h_crop * tf.random.uniform(shape=[], minval=0.67, maxval=1.0)
    h_crop, w_crop = tf.cast(h_crop, tf.int32), tf.cast(w_crop, tf.int32)
    opposite_aspectratio = tf.random.uniform(shape=[])
    if opposite_aspectratio < 0.5:
        h_crop, w_crop = w_crop, h_crop
    image = tf.image.random_crop(image, size=[h_crop, w_crop, 6])

    # Horizontal flipping
    horizontal_flip = tf.random.uniform(shape=[])
    if horizontal_flip < 0.5:
        image = tf.image.random_flip_left_right(image)

    # Resizing to original size
    image = tf.image.resize(image, size=[16, 16])
    return image


def my_augment(image):
    h_crop = tf.cast(tf.random.uniform(shape=[], minval=6, maxval=17, dtype=tf.int32), tf.float32)
    w_crop = h_crop * tf.random.uniform(shape=[], minval=0.67, maxval=1.0)
    h_crop, w_crop = tf.cast(h_crop, tf.int32), tf.cast(w_crop, tf.int32)
    opposite_aspectratio = tf.random.uniform(shape=[])
    if opposite_aspectratio < 0.5:
        h_crop, w_crop = w_crop, h_crop
    image = tf.image.random_crop(image, size=[h_crop, w_crop, 6])
    
    # # Horizontal flipping
    # horizontal_flip = tf.random.uniform(shape=[])
    # if horizontal_flip < 0.5:
    #     image = tf.image.random_flip_left_right(image)
        
    # # Vertical flip
    # vertical_flip = tf.random.uniform(shape=[])
    # if vertical_flip < 0.5:
    #     image = tf.image.random_flip_up_down(image)
    
    # # rotate 90 degrees
    # rotate =  tf.random.uniform(shape=[])
    # if rotate < 0.5:
    #     image = tf.image.rot90(image)

    # Resizing to original size
    image = tf.image.resize(image, size=[16, 16])
    return image




def random_color_distortion(image):
    # Random color jittering (strength 0.5)
    color_jitter = tf.random.uniform(shape=[])
    if color_jitter < 0.8:
        image = tf.image.random_brightness(image, max_delta=0.4)
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        # channel_inverse = tf.random.uniform(shape=[])
        # if channel_inverse < 0.5:
        image = tf.concat([image[:,:,3:6], image[:,:,0:3]], axis=2)
        image = tf.concat([tf.image.random_saturation(image[:,:,0:3], lower=0.6, upper=1.4), 
                          tf.image.random_saturation(image[:,:,3:6], lower=0.6, upper=1.4)], axis=2)
        image = tf.concat([tf.image.random_hue(image[:,:,0:3], max_delta=0.1), 
                          tf.image.random_hue(image[:,:,3:6], max_delta=0.1)], axis=2)
        #image = tf.clip_by_value(image, 0, 1)

    # Color dropping
    color_drop = tf.random.uniform(shape=[])
    if color_drop < 0.2:
        image = (tf.image.rgb_to_grayscale(image[:,:,0:3]) + tf.image.rgb_to_grayscale(image[:,:,3:6])) / 2
        image = tf.tile(image, [1, 1, 6])

    return image


@tf.function
def augment_image_pretraining(image):
    image = random_crop_flip_resize(image)
    image = random_color_distortion(image)
    return image

def augment_image_pretraining_mixup(image):
    image = random_crop_flip_resize(image)
    return image

@tf.function
def augment_image_finetuning(image):
    image = random_crop_flip_resize(image)
    return image