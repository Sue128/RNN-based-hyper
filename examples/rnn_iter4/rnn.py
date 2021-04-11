# -*- coding: utf-8 -*-
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Nonlinear transform coder with hyperprior for RGB images.

This is the image compression model published in:
J. Ballé, D. Minnen, S. Singh, S.J. Hwang, N. Johnston:
"Variational Image Compression with a Scale Hyperprior"
Int. Conf. on Learning Representations (ICLR), 2018
https://arxiv.org/abs/1802.01436

This is meant as 'educational' code - you can use this to get started with your
own experiments. To reproduce the exact results from the paper, tuning of hyper-
parameters may be necessary. To compress images with published models, see
`tfci.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob
import sys

from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow.compat.v1 as tf

import tensorflow_compression as tfc

from scipy.misc import imread, imsave
import time
import math

from msssim import msssim
from datetime import datetime


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def read_png(filename):
  """Loads a PNG image file."""
  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 255
  return image


def quantize_image(image):
  image = tf.round(image * 255)
  image = tf.saturate_cast(image, tf.uint8)
  return image


def write_png(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)


def rnn_conv(name, inputs, hiddens, filters, kernel_size, strides):
    '''Convolution RNN cell
    Args:
        name: name of current Conv RNN layer
        inputs: inputs tensor with shape (batch_size, height, width, channel)
        hiddens: hidden states from the previous iteration
        kernel_size: tuple of kernel size
        strides: strides size

    Output:
        hidden state and cell state of this layer
    '''
    gates_filters = 4 * filters
    hidden, cell = hiddens
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        conv_inputs = tf.layers.conv2d(inputs=inputs, filters=gates_filters,
                                       kernel_size=kernel_size, strides=strides, padding='same', name='conv_inputs')
        conv_hidden = tf.layers.conv2d(inputs=hidden, filters=gates_filters,
                                       kernel_size=kernel_size, padding='same', name='conv_hidden')
    in_gate, f_gate, out_gate, c_gate = tf.split(
        conv_inputs + conv_hidden, 4, axis=-1)
    in_gate = tf.nn.sigmoid(in_gate)
    f_gate = tf.nn.sigmoid(f_gate)
    out_gate = tf.nn.sigmoid(out_gate)
    c_gate = tf.nn.tanh(c_gate)
    new_cell = tf.multiply(f_gate, cell) + tf.multiply(in_gate, c_gate)
    new_hidden = tf.multiply(out_gate, tf.nn.tanh(new_cell))
    return new_hidden, new_cell


def initial_hidden(input_size, filters, kernel_size, name):
    """Initialize hidden and cell states, all zeros"""
    h_name = name + '_h'
    c_name = name + '_c'
    shape = [input_size] + kernel_size + [filters]
    hidden = tf.zeros(shape)
    cell = tf.zeros(shape)
    return hidden, cell


def padding(x, stride):
    if x % stride == 0:
        return x // stride
    else:
        return x // stride + 1

class encoder(object):
    """Encoder
    Args:
        batch_size: mini-batch size
        is_training: boolean variable controls quantizer behaviour
        height: height of input image data
        width: width of input image data
    """

    def __init__(self, batchsize, is_training= False, height = 256, width = 256):
        self.is_training = is_training
        self.batchsize = batchsize
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):
        """Initialize hidden and cell states"""
        height = padding(padding(self.height, 2), 2)
        width = padding(padding(self.width, 2), 2)
        self.hiddens1 = initial_hidden(
            self.batchsize, 256, [height, width], 'encoder1')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens2 = initial_hidden(
            self.batchsize, 512, [height, width], 'encoder2')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens3 = initial_hidden(
            self.batchsize, 512, [height, width], 'encoder3')

    def encode(self, inputs):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            encoder_rnn_input = tf.layers.conv2d(
                inputs = inputs, filters=64, kernel_size=[3, 3], strides=(2, 2), padding='same', name='encoder_rnn_input')
            self.hiddens1 = rnn_conv('encoder_rnn_conv_1',
                                     encoder_rnn_input, self.hiddens1, 256, [3, 3], (2, 2))
            self.hiddens2 = rnn_conv('encoder_rnn_conv_2',
                                     self.hiddens1[0], self.hiddens2, 512, [3, 3], (2, 2))
            self.hiddens3 = rnn_conv('encoder_rnn_conv_3',
                                     self.hiddens2[0], self.hiddens3, 512, [3, 3], (2, 2))
            encoder_output = tf.layers.conv2d(inputs=self.hiddens3[0], filters=128,
                                               kernel_size=[1, 1], padding='same', name='encoder_outputs', activation = None)

        return encoder_output

class decoder(object):
    """Decoder
    Args:
        batch_size: mini-batch size
        height: height of input image data
        width: width of input image data
    """

    def __init__(self, batchsize, height=256, width=256):
        self.batchsize = batchsize
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):
        height = padding(self.height, 2)
        width = padding(self.width, 2)
        self.hiddens4 = initial_hidden(
            self.batchsize, 128, [height, width], 'decoder4')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens3 = initial_hidden(
            self.batchsize, 256, [height, width], 'decoder3')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens2 = initial_hidden(
            self.batchsize, 512, [height, width], 'decoder2')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens1 = initial_hidden(
            self.batchsize, 512, [height, width], 'decoder1')

    def decode(self, inputs):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            decoder_rnn_input = tf.layers.conv2d(inputs=inputs, filters=128, kernel_size=[
                3, 3], strides=(1, 1), padding='same', name='decoder_rnn_input')
            self.hiddens1 = rnn_conv('decoder_rnn_conv_1',
                                     decoder_rnn_input, self.hiddens1, 512, [2, 2], (1, 1))
            d_rnn_h1 = tf.depth_to_space(self.hiddens1[0], 2)
            self.hiddens2 = rnn_conv('decoder_rnn_conv_2',
                                     d_rnn_h1, self.hiddens2, 512, [3, 3], (1, 1))
            d_rnn_h2 = tf.depth_to_space(self.hiddens2[0], 2)
            self.hiddens3 = rnn_conv('decoder_rnn_conv_3',
                                     d_rnn_h2, self.hiddens3, 256, [3, 3], (1, 1))
            d_rnn_h3 = tf.depth_to_space(self.hiddens3[0], 2)
            self.hiddens4 = rnn_conv('decoder_rnn_conv_4',
                                     d_rnn_h3, self.hiddens4, 128, [3, 3], (1, 1))
            d_rnn_h4 = tf.depth_to_space(self.hiddens4[0], 2)

            output = tf.layers.conv2d(inputs=d_rnn_h4, filters=3, kernel_size=[
                3, 3], strides=(1, 1), padding='same', name='output', activation=tf.nn.tanh)
        return output / 2

class HyperEncoder(object):
    def __init__(self, batchsize, is_training=False, height=16, width=16):
        self.is_training = is_training
        self.batchsize = batchsize
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):
        """Initialize hidden and cell states"""
        height = padding(self.height, 2)
        width = padding(self.width, 2)
        self.hiddens1 = initial_hidden(
            self.batchsize, 128, [height, width], 'hyper_encoder1')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens2 = initial_hidden(
            self.batchsize, 128, [height, width], 'hyper_encoder2')

    def hyper_encode(self, inputs):
        with tf.variable_scope('hyper_encoder', reuse=tf.AUTO_REUSE):
            hyper_encoder_rnn_input = tf.layers.conv2d(
                inputs=inputs, filters=128, kernel_size=[3, 3], strides=(1, 1),padding='same', name='hyper_encoder_rnn_input')
            self.hiddens1 = rnn_conv('hyper_encoder_rnn_conv_1',
                                     hyper_encoder_rnn_input, self.hiddens1, 128, [3, 3], (2, 2))
            self.hiddens2 = rnn_conv('encoder_rnn_conv_2',
                                     self.hiddens1[0], self.hiddens2, 128, [3, 3], (2, 2))
            hyper_encoder_output = tf.layers.conv2d(inputs=self.hiddens2[0], filters=128,
                                              kernel_size=[1, 1], padding='same', name='hyper_encoder_outputs',
                                              activation=None)

        return hyper_encoder_output


class HyperDecoder(object):
    def __init__(self, batchsize, height=16, width=16):
        self.batchsize = batchsize
        self.height = height
        self.width = width
        self.init_hidden()

    def init_hidden(self):
        height = padding(self.height, 2)
        width = padding(self.width, 2)
        self.hiddens2 = initial_hidden(
            self.batchsize, 128, [height, width], 'hyper_decoder2')
        height = padding(height, 2)
        width = padding(width, 2)
        self.hiddens1 = initial_hidden(
            self.batchsize, 128, [height, width], 'hyper_decoder1')

    def hyper_decode(self, inputs):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            hyper_decoder_rnn_input = tf.layers.conv2d(inputs=inputs, filters=128, kernel_size=[
                3, 3], strides=(1, 1), padding='same', name='hyper_decoder_rnn_input')
            self.hiddens1 = rnn_conv('hyper_decoder_rnn_conv_1',
                                     hyper_decoder_rnn_input, self.hiddens1, 128, [3, 3], (1, 1))
            d_rnn_h1 = tf.depth_to_space(self.hiddens1[0], 2)

            self.hiddens2 = rnn_conv('hyper_decoder_rnn_conv_2',
                                     d_rnn_h1, self.hiddens2, 128, [3, 3], (1, 1))
            d_rnn_h2 = tf.depth_to_space(self.hiddens2[0], 2)

            hyper_output = tf.layers.conv2d(inputs=d_rnn_h2, filters=128, kernel_size=[
                3, 3], strides=(1, 1), padding='same', name='hyper_outputs', activation=None)


        return hyper_output

def train(args):
  """Trains the model."""

  if args.verbose:
    tf.logging.set_verbosity(tf.logging.INFO)

  # Create input data pipeline.
  with tf.device("/cpu:0"):
    train_files = glob.glob(args.train_glob)
    if not train_files:
      raise RuntimeError(
          "No training images found with glob '{}'.".format(args.train_glob))
    train_dataset = tf.data.Dataset.from_tensor_slices(train_files)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).repeat()
    train_dataset = train_dataset.map(
        read_png, num_parallel_calls=args.preprocess_threads)
    train_dataset = train_dataset.map(
        lambda x: tf.random_crop(x, (args.patchsize, args.patchsize, 3)))
    train_dataset = train_dataset.batch(args.batchsize)
    train_dataset = train_dataset.prefetch(32)


  num_pixels = args.batchsize * args.patchsize ** 2

  lmbda = [0.01, 0.02, 0.04, 0.08]
  iter = [0,1,2,3]

  # Get training patch from dataset.
  inputs = train_dataset.make_one_shot_iterator().get_next()
  x = inputs - 0.5

  e = encoder(args.batchsize, is_training=True)
  d = decoder(args.batchsize)
  he = HyperEncoder(args.batchsize, is_training=True)
  hd = HyperDecoder(args.batchsize)


  #iterations
  # Build autoencoder and hyperprior.
  entropy_bottlenecks = []
  train_loss = 0
  Train_BPP = 0
  Train_BPP1 = []
  Train_MSE = 0
  Train_MSE1 = []
  output = tf.zeros_like(x) + 0.5
  train_ops = []
  psnr_mul = []
  for i, lmb in zip(iter, lmbda):
      y = e.encode(x)
      z = he.hyper_encode(abs(y))
      entropy_bottleneck = tfc.EntropyBottleneck(name='entropy_iter'+ str(i))
      entropy_bottlenecks.append(entropy_bottleneck)
      z_tilde, z_likelihoods = entropy_bottleneck(z, training=True)
      sigma = hd.hyper_decode(z_tilde)
      scale_table = np.exp(np.linspace(
          np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
      conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table, name='conditional'+ str(i))
      y_tilde, y_likelihoods = conditional_bottleneck(y, training=True)
      x_tilde = d.decode(y_tilde)

      # Total number of bits divided by number of pixels.
      train_bpp = (tf.reduce_sum(tf.log(y_likelihoods)) +
                   tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels)

      # Mean squared error across pixels.
      train_mse = tf.reduce_mean(tf.squared_difference(x, x_tilde))
      # Multiply by 255^2 to correct for rescaling.
      train_mse *= 255 ** 2

      # The rate-distortion cost.
      Train_BPP += train_bpp
      Train_MSE += train_mse
      Train_BPP1.append(train_bpp)
      Train_MSE1.append(train_mse)

      train_loss += (lmb * train_mse + Train_BPP)

      #residual
      x = x - x_tilde
      #output
      output += x_tilde


  # Minimize loss and auxiliary loss, and execute update op.
  step = tf.train.create_global_step()
  main_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  main_step = main_optimizer.minimize(train_loss, global_step=step)

  aux_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  for i in range(args.iter):
    aux_step = aux_optimizer.minimize(entropy_bottlenecks[i].losses[0])
    train_op = tf.group(main_step, aux_step, entropy_bottlenecks[i].updates[0])
    train_ops.append(train_op)

  tf.summary.scalar("loss", train_loss)
  tf.summary.scalar("bpp", Train_BPP)
  tf.summary.scalar("mse", Train_MSE)

  tf.summary.image("original", quantize_image(inputs))
  tf.summary.image("reconstruction", quantize_image(output))

  class LoggerHook(tf.train.SessionRunHook):
      """
      print training information
      """
      def begin(self):
          self.step = -1
          self.start_time = time.time()

      def before_run(self, run_context):
          self.step += 1
          # this function called automatically during training
          # return all training information
          #print(tf.train.SessionRunArgs(inputs))
          return tf.train.SessionRunArgs([train_loss, Train_BPP, Train_BPP1, Train_MSE, Train_MSE1, inputs, output])


      def after_run(self, run_context, run_values):

          # step interval
          display_step = 50
          if self.step % display_step == 0:
              current_time = time.time()
              duration = current_time - self.start_time
              self.start_time = current_time
              # return the results of before_run(), which is loss
              loss, bpp, bpp1, mse, mse1, original, compressed_img = run_values.results
              print(bpp1, mse1)
              original *=255.0
              compressed_img = np.array(np.clip((compressed_img+0.5)*255.0, 0.0, 255.0), dtype=np.uint8)
              ms_ssim = msssim(compressed_img,original)
              psnr = 20 * math.log10( 255.0 / math.sqrt(mse))
              for i in range(args.iter):
                  psnrs = 20 * math.log10( 255.0 / math.sqrt(mse1[i]))
                  psnr_mul.append(psnrs)
              print(psnr_mul)
              psnr_mul.clear()

              # samples per second
              examples_per_sec = display_step * args.batchsize / duration
              # 每batch使用的时间
              sec_per_batch = float(duration / display_step)
              format_str = ('%s: step %d, loss = %.2f, Bpp = %.2f, MSE = %.2f, MS-SSIM = %.2f, PSNR = %.2f (%.1f examples/sec; %.3f '
                            'sec/batch)')
              print(format_str % (datetime.now(), self.step, loss, bpp, mse, ms_ssim, psnr,
                                  examples_per_sec, sec_per_batch))
          if self.step % (display_step * 20) == 0:
              loss, bpp, bpp1, mse, mse1, original, compressed_img = run_values.results
              original *=255.0
              compressed_img = np.array(np.clip((compressed_img + 0.5)*255.0, 0.0, 255.0), dtype=np.uint8)
              ms_ssim = msssim(compressed_img,original)
              psnr = 20 * math.log10( 255.0 / math.sqrt(mse))

              format_str = ('%s: step %d, loss = %.2f, Bpp = %.2f, MSE = %.2f, MS-SSIM = %.2f, PSNR = %.2f')
              fin = open("rnn_256-512_0.01-0.08_loss.txt", 'a+')
              fin.write(format_str % (datetime.now(), self.step, loss, bpp, mse, ms_ssim, psnr))
              fin.write("\n")



  with tf.train.MonitoredTrainingSession(
      hooks=[
          tf.train.StopAtStepHook(last_step=args.last_step),
          tf.train.NanTensorHook(train_loss),
          LoggerHook()], checkpoint_dir=args.checkpoint_dir,
      save_checkpoint_secs=300, save_summaries_secs=60) as sess:
    while not sess.should_stop():
        for i in range(args.iter):
            sess.run(train_ops[i])



def compress(args):
  """Compresses an image."""

  # Load input image and add batch dimension.
  image = imread(args.input_file).astype(np.float32)

  img = read_png(args.input_file)
  img = tf.expand_dims(img, 0)
  img.set_shape([1, img.shape[1], img.shape[2], 3])
  x_shape = tf.shape(img)
  x = img - 0.5

  # Transform and compress the image, then remove batch dimension.
  e = encoder(args.batchsize, height=image.shape[0], width=image.shape[1])
  d = decoder(args.batchsize, height=image.shape[0], width=image.shape[1])
  he = HyperEncoder(args.batchsize, height=image.shape[0] // 16, width=image.shape[1] // 16)
  hd = HyperDecoder(args.batchsize, height=image.shape[0] // 16, width=image.shape[1] // 16)


  #iteration
  # Transform and compress the image.
  encodes = []
  hyper_encodes = []
  strings = []
  side_strings = []
  MSE = []
  PSNR = []
  MSSSIM = []
  eval_bpp = 0
  x_hats = tf.zeros_like(x) + 0.5
  num_pixels = tf.cast(tf.reduce_prod(tf.shape(img)[:-1]), dtype=tf.float32)
  comps = []
  for i in range(args.iter):
      y = e.encode(x)
      encodes.append(y)
      y_shape = tf.shape(y)

      z = he.hyper_encode(abs(y))
      hyper_encodes.append(z)

      entropy_bottleneck = tfc.EntropyBottleneck(name='entropy_iter'+ str(i))
      z_hat, z_likelihoods = entropy_bottleneck(z, training=False)

      sigma = hd.hyper_decode(z_hat)
      sigma = sigma[:, :y_shape[1], :y_shape[2], :]
      scale_table = np.exp(np.linspace(
          np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
      conditional_bottleneck = tfc.GaussianConditional(sigma, scale_table,name='conditional'+ str(i))

      side_string = entropy_bottleneck.compress(z)
      side_strings.append(side_string)
      string = conditional_bottleneck.compress(y)
      strings.append(string)

      # Transform the quantized image back (if requested).
      y_hat, y_likelihoods = conditional_bottleneck(y, training=False)
      x_hat = d.decode(y_hat)
      x_hat = x_hat[:, :x_shape[1], :x_shape[2], :]



      # Total number of bits divided by number of pixels.
      eval_bpp += ((tf.reduce_sum(tf.log(y_likelihoods)) +
                  tf.reduce_sum(tf.log(z_likelihoods))) / (-np.log(2) * num_pixels))

      x = x - x_hat

      x_hats += x_hat

      # Bring both images back to 0..255 range.
      original = img * 255
      compressdes = tf.clip_by_value(x_hats, 0, 1)
      compressdes = tf.round(compressdes * 255)
      comps.append(compressdes)

      mse = tf.reduce_mean(tf.squared_difference(original, compressdes))
      psnr = tf.squeeze(tf.image.psnr(compressdes, original, 255))
      msssim = tf.squeeze(tf.image.ssim_multiscale(compressdes, original, 255))

      MSE.append(mse)
      PSNR.append(psnr)
      MSSSIM.append(msssim)

  with tf.Session() as sess:
    # Load the latest model checkpoint, get the compressed string and the tensor
    # shapes.
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    bpp = 0
    for i in range(args.iter):
        tensors = [strings[i], side_strings[i],
                   tf.shape(img)[1:-1], tf.shape(encodes[i])[1:-1], tf.shape(hyper_encodes[i])[1:-1]]
        arrays = sess.run(tensors)

        # Write a binary file with the shape information and the compressed string.
        packed = tfc.PackedTensors()
        packed.pack(tensors, arrays)
        with open(args.output_file, "wb") as f:
          f.write(packed.string)

        # If requested, transform the quantized image back and measure performance.

        eval_bpps, mses, psnrs, msssims, num_pixelses = sess.run(
            [eval_bpp, MSE[i], PSNR[i], MSSSIM[i], num_pixels])
        comp = comps[i].eval()
        # The actual bits per pixel including overhead.
        bpp += (len(packed.string) * 8 / num_pixelses)

        print("Mean squared error: {:0.4f}".format(mses))
        print("PSNR (dB): {:0.2f}".format(psnrs))
        print("Multiscale SSIM: {:0.4f}".format(msssims))
        print("Multiscale SSIM (dB): {:0.2f}".format(-10 * np.log10(1 - msssims)))
        print("Information content in bpp: {:0.4f}".format(eval_bpps))
        print("Actual bits per pixel: {:0.4f}".format(bpp))
        fin = open("rnn_256-512_0.01-0.08_results.txt", 'a+')
        fin.write("Iter %d, %.8f,  %.8f,  %.8f, %.8f" % (i, mses, psnrs, msssims, bpp))
        fin.write("\n")
  
        comp = np.squeeze(comp)
        imsave('compressed/recon_'+str(i) + '.png', comp)


def decompress(args):
  """Decompresses an image."""

  # Read the shape information and compressed string from the binary file.
  string = tf.placeholder(tf.string, [1])
  side_string = tf.placeholder(tf.string, [1])
  x_shape = tf.placeholder(tf.int32, [2])
  y_shape = tf.placeholder(tf.int32, [2])
  z_shape = tf.placeholder(tf.int32, [2])
  with open('/media/xproject/file/Surige/compression-master/examples/rnn_baseline/recon/recon.bin', "rb") as f:
      packed = tfc.PackedTensors(f.read())

  tensors = [string, side_string, x_shape, y_shape, z_shape]
  arrays = packed.unpack(tensors)

  # Add a batch dimension, then decompress and transform the image back.
  d = decoder(args.batchsize, height=x_shape[0], width=x_shape[1])
  hd = HyperDecoder(args.batchsize, height=x_shape[0] // 16, width=x_shape[1] // 16)
  entropy_bottleneck = tfc.EntropyBottleneck(name='entropy_iter', dtype=tf.float32)

  # Decompress and transform the image back.
  z_shape = tf.concat([z_shape, [args.num_filters]], axis=0)
  z_hat = entropy_bottleneck.decompress(
      side_string, z_shape, channels=args.num_filters)
  sigma = hd.hyper_decode(z_hat)
  sigma = sigma[:, :y_shape[0], :y_shape[1], :]
  scale_table = np.exp(np.linspace(
      np.log(SCALES_MIN), np.log(SCALES_MAX), SCALES_LEVELS))
  conditional_bottleneck = tfc.GaussianConditional(
      sigma, scale_table, dtype=tf.float32)
  y_hat = conditional_bottleneck.decompress(string)
  x_hat = d.decode(y_hat)

  # Remove batch dimension, and crop away any extraneous padding on the bottom
  # or right boundaries.
  x_hat = x_hat[0, :x_shape[0], :x_shape[1], :]

  # Write reconstructed image out as a PNG file.
  op = write_png(args.output_file, x_hat)

  # Load the latest model checkpoint, and perform the above actions.
  with tf.Session() as sess:
    latest = tf.train.latest_checkpoint(checkpoint_dir=args.checkpoint_dir)
    tf.train.Saver().restore(sess, save_path=latest)
    sess.run(op, feed_dict=dict(zip(tensors, arrays)))


def parse_args(argv):
  """Parses command line arguments."""
  parser = argparse_flags.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # High-level options.
  parser.add_argument(
      "--verbose", "-V", action="store_true",
      help="Report bitrate and distortion when training or compressing.")
  parser.add_argument(
      "--num_filters", type=int, default=128,
      help="Number of filters per layer.")
  parser.add_argument(
      "--checkpoint_dir", default="train_rnn256-512_0.01-0.08",
      help="Directory where to save/load model checkpoints.")
  parser.add_argument(
      "--iter", type=int, default=4,
      help="number of iteration.")
  subparsers = parser.add_subparsers(
      title="commands", dest="command",
      help="What to do: 'train' loads training data and trains (or continues "
           "to train) a new model. 'compress' reads an image file (lossless "
           "PNG format) and writes a compressed binary file. 'decompress' "
           "reads a binary file and reconstructs the image (in PNG format). "
           "input and output filenames need to be provided for the latter "
           "two options. Invoke '<command> -h' for more information.")

  # 'train' subcommand.
  train_cmd = subparsers.add_parser(
      "train",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Trains (or continues to train) a new model.")
  train_cmd.add_argument(
      "--train_glob", default="/media/lzou/file/SURIGE/tensorflow_compression/examples/image_256/*.jpg",
      help="Glob pattern identifying training data. This pattern must expand "
           "to a list of RGB images in PNG format.")
  train_cmd.add_argument(
      "--batchsize", type=int, default=4,
      help="Batch size for training.")
  train_cmd.add_argument(
      "--patchsize", type=int, default=256,
      help="Size of image patches for training.")
  #train_cmd.add_argument(
      #"--lambda", type=float, default=0.01, dest="lmbda",
      #help="Lambda for rate-distortion tradeoff.")
  train_cmd.add_argument(
      "--last_step", type=int, default=400000,
      help="Train up to this number of steps.")
  train_cmd.add_argument(
      "--preprocess_threads", type=int, default=16,
      help="Number of CPU threads to use for parallel decoding of training "
           "images.")

  # 'compress' subcommand.
  compress_cmd = subparsers.add_parser(
      "compress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a PNG file, compresses it, and writes a TFCI file.")
  compress_cmd.add_argument(
      "--batchsize", type=int, default=1,
      help="Batch size for training.")

  # 'decompress' subcommand.
  decompress_cmd = subparsers.add_parser(
      "decompress",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
      description="Reads a TFCI file, reconstructs the image, and writes back "
                  "a PNG file.")

  # Arguments for both 'compress' and 'decompress'.
  for cmd, ext in ((compress_cmd, ".tfci"), (decompress_cmd, ".png")):
    cmd.add_argument(
        "input_file",
        help="Input filename.")
    cmd.add_argument(
        "output_file", nargs="?",
        help="Output filename (optional). If not provided, appends '{}' to "
             "the input filename.".format(ext))

  # Parse arguments.
  args = parser.parse_args(argv[1:])
  if args.command is None:
    parser.print_usage()
    sys.exit(2)
  return args


def main(args):
  # Invoke subcommand.
  if args.command == "train":
    train(args)
  elif args.command == "compress":
    if not args.output_file:
      args.output_file = args.input_file + ".tfci"
    compress(args)
  elif args.command == "decompress":
    if not args.output_file:
      args.output_file = args.input_file + ".png"
    decompress(args)


if __name__ == "__main__":
  app.run(main, flags_parser=parse_args)
