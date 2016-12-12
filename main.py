import argparse
import cv2
import numpy as np
import os
import skimage.io
import tensorflow as tf

import vgg19

CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS= ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

ALPHA = 1.0
BETA = 50.0
LR = 1.0

def load_image(path):
    img = skimage.io.imread(path)
    yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
    img = img - vgg19.VGG_MEAN
    img = img[:,:,(2,1,0)]  # rgb to bgr
    return img[np.newaxis, :, :, :], yuv

def save_image(img, path, content_yuv=None):
    img = np.squeeze(img)
    img = img[:,:,(2,1,0)]  # bgr to rgb
    img = img + vgg19.VGG_MEAN
    if content_yuv is not None:
        yuv = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2YUV)
        yuv[:,:,1:3] = content_yuv[:,:,1:3]
        img = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    img = np.clip(img, 0, 255).astype(np.uint8)
    skimage.io.imsave(path, img)

def feature_to_gram(f):
    shape = f.get_shape()
    n_channels = shape[3].value
    size = np.prod(shape).value
    f = tf.reshape(f, [-1, n_channels])
    return tf.matmul(tf.transpose(f), f) / size

def get_style_rep(vgg):
    return map(feature_to_gram, map(lambda l: getattr(vgg, l), STYLE_LAYERS))

def compute_style_loss(style_rep, image_vgg):
    style_losses = map(tf.nn.l2_loss, [a - b for (a, b) in zip(style_rep, get_style_rep(image_vgg))])
    style_losses = [style_losses[i] / (style_rep[i].size) for i in range(len(style_losses))]
    return reduce(tf.add, style_losses)

def main(content_path, style_path, output_dir, iterations, vgg_path, preserve_color):
    # mean subtract input images
    content_img, content_yuv = load_image(content_path)
    style_img, _ = load_image(style_path)

    # obtain content and style reps
    with tf.Session() as sess:
        content_vgg = vgg19.Vgg19(vgg_path)
        content = tf.placeholder("float", content_img.shape)
        content_vgg.build(content)
        style_vgg = vgg19.Vgg19(vgg_path)
        style = tf.placeholder("float", style_img.shape)
        style_vgg.build(style)

        sess.run(tf.global_variables_initializer())
        content_rep = sess.run(getattr(content_vgg, CONTENT_LAYER), feed_dict={content: content_img})
        style_rep = sess.run(get_style_rep(style_vgg), feed_dict={style: style_img})

    # start with white noise
    noise = tf.truncated_normal(content_img.shape, stddev=0.1*np.std(content_img))
    image = tf.Variable(noise)
    image_vgg = vgg19.Vgg19(vgg_path)
    image_vgg.build(image)

    # define losses and optimizer
    content_loss = tf.nn.l2_loss(getattr(image_vgg, CONTENT_LAYER) - content_rep) / content_rep.size
    style_loss = compute_style_loss(style_rep, image_vgg)
    loss = ALPHA*content_loss + BETA*style_loss
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

    # style transfer
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1, iterations+1):
            sess.run(optimizer)
            fmt_str = 'Iteration {:4}/{:4}    content loss {:14}  style loss {:14}'
            print(fmt_str.format(i, iterations, ALPHA*content_loss.eval(), BETA*style_loss.eval()))

            # undo mean subtract and save output image
            output_path = os.path.join(output_dir, 'output_{:04}.jpg'.format(i))
            save_image(image.eval(), output_path, content_yuv if preserve_color else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', dest='content', default='examples/content.jpg', help='path to content image')
    parser.add_argument('--style', dest='style', default='examples/style.jpg', help='path to style image')
    parser.add_argument('--output', dest='output', default='output/', help='output directory')
    parser.add_argument('--iterations', type=int, dest='iterations', default=1000, help='iterations')
    parser.add_argument('--vgg', dest='vgg', default='vgg19.npy', help='path to pretrained vgg-19 npy model')
    parser.add_argument('--preserve_color', dest='preserve_color', action='store_true', help='preserve color')
    args = parser.parse_args()
    print('Running style transfer with arguments: {}'.format(vars(args)))

    assert os.path.isfile(args.vgg), \
        'Pretrained vgg-19 model not found at {}. Please refer to ' \
        'https://github.com/machrisaa/tensorflow-vgg for download instructions.'.format(args.vgg)
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print('Saving output images to {}'.format(args.output))

    main(args.content, args.style, args.output, args.iterations, args.vgg, args.preserve_color)
