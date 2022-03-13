import cv2
from openslide import OpenSlide
from PIL import Image
import numpy as np
import staintools


def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    dxabs = cv2.convertScaleAbs(dx)
    dyabs = cv2.convertScaleAbs(dy)
    mag = cv2.addWeighted(dxabs, 0.5, dyabs, 0.5, 0)
    return mag


def wsi_to_tiles(idx, wsi, refer_img, s):
    normalizer = staintools.StainNormalizer(method='vahadane')
    refer_img = staintools.read_image(refer_img)
    normalizer.fit(refer_img)
    count = 0
    sys.stdout.write('Start task %d: %s \n' % (idx, wsi))
    slide_id = wsi.rsplit('/', 1)[1].split('.')[0]
    tile_path = os.path.join('./tiles', slide_id)
    img = OpenSlide(os.path.join(wsi))
    if str(img.properties.values.__self__.get('tiff.ImageDescription')).split("|")[1] == "AppMag = 40":
        sz = 2048
        seq = 1536
    else:
        sz = 1024
        seq = 768
    [w, h] = img.dimensions
    for x in range(1, w, seq):
        for y in range(1, h, seq):
            img_tmp = img.read_region(location=(x, y), level=0, size=(sz, sz)) \
                            .convert("RGB").resize((299, 299), Image.ANTIALIAS)
            grad = getGradientMagnitude(np.array(img_tmp))
            unique, counts = np.unique(grad, return_counts=True)
            if counts[np.argwhere(unique <= 15)].sum() < 299 * 299 * s:
                img_tmp = normalizer.transform(np.array(img_tmp))
                img_tmp = Image.fromarray(img_tmp)
                img_tmp.save(tile_path + "/" + str(x) + "_" + str(y) + '.jpg', 'JPEG', optimize=True, quality=94)
                count += 1
    sys.stdout.write('End task %d with %d tiles\n' % (idx, count))

