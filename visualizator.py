import numpy as np


def deproc_img(data):
  """
  The function converts numpy array with float values to numpy array [0, 255]
  Input:
  - numpy array
  """
  data = np.clip(data, 0, 1)
  data *= 255
  data = np.clip(data, 0, 255).astype("uint8")
  return data


def visualize(data, img_w, img_h, ncols=10, margin=5, depth=3):
  nrows = len(data) // ncols + (len(data) % ncols != 0)
  h = img_h*nrows + margin*(nrows + 1)
  w = img_w*ncols + margin*(ncols + 1)
  z_array = np.zeros((depth, h, w), dtype='uint8')

  imgs_iterator = iter(data)
  for row in range(nrows):
    y = margin + row*(img_h + margin)
    for col in range(ncols):
      curr_array = next(imgs_iterator, None)
      if curr_array is not None:
        curr_array = deproc_img(curr_array)
        x = margin + col*(img_w + margin)
        z_array[:, y:y + img_h, x:x + img_w] = curr_array[:, :, :]
      else:
        break
  return z_array