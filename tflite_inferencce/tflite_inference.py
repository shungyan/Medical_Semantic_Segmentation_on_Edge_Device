# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""An example of semantic segmentation.

The following command runs this script and saves a new image showing the
segmented pixels at the location specified by `output`:

```
bash examples/install_requirements.sh semantic_segmentation.py

python3 examples/semantic_segmentation.py \
  --model test_data/deeplabv3_mnv2_pascal_quant_edgetpu.tflite \
  --input test_data/bird.bmp \
  --keep_aspect_ratio \
  --output ${HOME}/segmentation_result.jpg
```
"""

import argparse

import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

# from pycoral.adapters import common
# from pycoral.adapters import segment
# from pycoral.utils.edgetpu import make_interpreter


# def create_pascal_label_colormap():
#   """Creates a label colormap used in PASCAL VOC segmentation benchmark.

#   Returns:
#     A Colormap for visualizing segmentation results.
#   """
#   colormap = np.zeros((256, 3), dtype=int)
#   indices = np.arange(256, dtype=int)

#   for shift in reversed(range(8)):
#     for channel in range(3):
#       colormap[:, channel] |= ((indices >> channel) & 1) << shift
#     indices >>= 3

#   return colormap


# def label_to_color_image(label):
#   """Adds color defined by the dataset colormap to the label.

#   Args:
#     label: A 2D array with integer type, storing the segmentation label.

#   Returns:
#     result: A 2D array with floating type. The element of the array
#       is the color indexed by the corresponding element in the input label
#       to the PASCAL color map.

#   Raises:
#     ValueError: If label is not of rank 2 or its value is larger than color
#       map maximum entry.
#   """
#   if label.ndim != 2:
#     raise ValueError('Expect 2-D input label')

#   colormap = create_pascal_label_colormap()

#   if np.max(label) >= len(colormap):
#     raise ValueError('label value too large.')

#   return colormap[label]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', required=True,
                      help='Path of the segmentation model.')
  parser.add_argument('--input', required=True,
                      help='File path of the input image.')
  parser.add_argument('--output', default='semantic_segmentation_result.png',
                      help='File path of the output image.')
  # parser.add_argument(
  #     '--keep_aspect_ratio',
  #     action='store_true',
  #     default=False,
  #     help=(
  #         'keep the image aspect ratio when down-sampling the image by adding '
  #         'black pixel padding (zeros) on bottom or right. '
  #         'By default the image is resized and reshaped without cropping. This '
  #         'option should be the same as what is applied on input images during '
  #         'model training. Otherwise the accuracy may be affected and the '
  #         'bounding box of detection result may be stretched.'))
  args = parser.parse_args()

  # interpreter = make_interpreter(args.model, device=':0')
  # interpreter.allocate_tensors()
  # width, height = common.input_size(interpreter)

  #img = Image.open(args.input)
  # if args.keep_aspect_ratio:
  #   resized_img, _ = common.set_resized_input(
  #       interpreter, img.size, lambda size: img.resize(size, Image.ANTIALIAS))
  # else:
  #   resized_img = img.resize((width, height), Image.ANTIALIAS)
  #   common.set_input(interpreter, resized_img)


    # Load the image
  # im = Image.open(args.input)
  # im = im.convert("RGB")
  # # Convert the image to a NumPy array
  # im_np = np.array(im)
  # print(im_np.shape)
  
  # # Reshape the NumPy array to include the color channels
  # img = im_np.reshape((im_np.shape[0], im_np.shape[1], 3))
  # print(img.shape)
  # img=np.resize(im_np,(128,128))[:,:,0]
  # print(img.shape)

  ''''''
  img=cv2.imread(args.input)
  print(img.shape)
  img = cv2.resize(img, (128,128))[:,:,0]
  print(img.shape)
  ''''''
  
  image = np.expand_dims(img, axis=0)
  image = np.expand_dims(image, axis=3)
  print(image.shape)
  image=np.float32(image)


  interpreter = tf.lite.Interpreter(model_path=args.model)
  interpreter.allocate_tensors()
  

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Load image
  input_data = image

  interpreter.set_tensor(input_details[0]['index'], input_data)
  interpreter.invoke()
  output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
  output_img=output_data_tflite[0,:,:,0]
  print(output_img.shape)

  # result = segment.get_output(interpreter)
  # if len(result.shape) == 3:
  #   result = np.argmax(result, axis=-1)

  # # If keep_aspect_ratio, we need to remove the padding area.
  # new_width, new_height = resized_img.size
  # result = result[:new_height, :new_width]
  # mask_img = Image.fromarray(label_to_color_image(result).astype(np.uint8))

  # # Concat resized input image and processed segmentation results.
  # output_img = Image.new('RGB', (2 * new_width, new_height))
  # output_img.paste(resized_img, (0, 0))
  # output_img.paste(mask_img, (width, 0))

  # from matplotlib import pyplot as plt
  # plt.imshow(output_img)
  # plt.show()
  # plt.savefig(args.output)

  # print(output_img.dtype)
  # output_img=output_img.astype(np.uint8)
  # print(output_img.dtype)

  output_img = (output_img / np.max(output_img) * 255).astype(np.uint8)

  output_img=Image.fromarray(output_img,"L")
  output_img.save(args.output)
  output_img.show()
  print('Done. Results saved at', args.output)

if __name__ == '__main__':
  main()