import argparse
import numpy as np
from PIL import Image

from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.utils.edgetpu import make_interpreter

def main():
    # Load the TFLite model
    interpreter = make_interpreter("model.tflite", device=':0')
    interpreter.allocate_tensors()

    resized_img=np.load('input.npy')
    common.set_input(interpreter, resized_img)
    interpreter.invoke()

    result = segment.get_output(interpreter)
    result = np.squeeze(result)
    np.save("result.npy",result)

if __name__ == '__main__':
  main()