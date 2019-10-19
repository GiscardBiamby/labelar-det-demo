import tensorflow as tf
from pathlib import Path

img_path = Path("../training/data/uist-mugs-v2-train/images/uist-mugs-v2_train")

tf.enable_eager_execution()
images = list(img_path.glob("*.png"))

for i, image in enumerate(images):

    with tf.gfile.GFile(str(image), "rb") as fid:
        image_data = fid.read()

    image_tensor = tf.image.decode_png(image_data, channels=3, name=None)
