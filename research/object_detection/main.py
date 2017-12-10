import os
import sys
sys.path.append("/home/sitework/git/models/research/object_detection")
sys.path.append("/home/sitework/git/models/research/slim")
sys.path.append("/home/sitework/git/models/research/")
import numpy as np
import tensorflow as tf
import tornado.web
from PIL import Image
import matplotlib as mll
mll.use("Agg")
from matplotlib import pyplot as plt

sys.path.append(".")

from utils import label_map_util

from utils import visualization_utils as vis_util

config = tf.ConfigProto(device_count={'GPU': 0})
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT ='/home/sitework/git/models/research/object_detection/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
IMAGE_SIZE = (1920, 1080)
with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')


def inference(image_path):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Visualization of the results of a detection.
    image_np1 = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=10)
    plt.figure(figsize=IMAGE_SIZE, dpi=500)
    plt.imshow(image_np1)
    if os.path.exists(image_path.split('.')[0] + '_.jpg'):
        os.remove(image_path.split('.')[0] + '_.jpg')
    plt.savefig(image_path.split('.')[0] + '_.jpg')  # Add
    return image_path.split('.')[0] + '_labeled.jpg'


# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        # self.write('''
        #           <html>
        #             <head><br>                      <title>Upload File</title><br>                    </head>
        #             <body>
        #               <form action='file' enctype="multipart/form-data" method='post'>
        #                 <input type='file' name='file'/><br/>
        #                 <input type='submit' value='submit'/>
        #               </form>
        #             </body>
        #           </html>
        #           ''')
        self.render("main.html")

    def post(self):
        upload_path = os.path.join(os.path.dirname(__file__), 'static')  # 文件的暂存路径
        file_metas = self.request.files['file']  # 提取表单中‘name’为‘file’的文件元数据
        for meta in file_metas:
            filename = meta['filename']
            filepath = os.path.join(upload_path, filename)
            with open(filepath, 'wb') as up:  # 有些文件需要已二进制的形式存储，实际中可以更改
                up.write(meta['body'])
            inference(filepath)
            self.write('''<!doctype html>
            <html lang="en">
            <head>
                <title>图像识别</title>
                <style type="text/css">
                body{text-align: center;
                    background-image: url(static/bj.jpg);
                    background-repeat: no-repeat;
                    background-size: 100%;
                }
                #tou{color: #00ffff;
                    font-size: 40px;}
                </style>
            </head>
            <body>
                <h1 id="tou">图像识别</h1>
                <form enctype="multipart/form-data" name="form1" >
            
                <p style="font-size: 20px;color: #00ffff"><b>结果显示:</b></p>
                <p>
                    <img id="preview" alt="" name="pic" style="width: 50%" src=''' + "static/{}_.jpg".format(filename.
                                                                                                             split(".")[
                                                                                                                                        0]) +
                       '''/>
                           </p>
                           </form>
                       </body>
                       </html>'''.format(a="static/{}_.jpg".format(filename.split(".")[0])))
            l = "static/{}_.jpg".format(filename.split(".")[0])
            pass


settings = {
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
}
app = tornado.web.Application([
    (r'/file', MainHandler),
    (r'/', MainHandler)
], **settings)

if __name__ == "__main__":
    app.listen(1000)
    tornado.ioloop.IOLoop.instance().start()
