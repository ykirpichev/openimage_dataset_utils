import tensorflow as tf
from PIL import Image  # uses pillow
import pandas as pd
import glob
import yaml
import os

from sets import Set

flags = tf.app.flags
flags.DEFINE_string('output_tfrecord', 'output.tfrecord', 'Path to output TFRecord')
flags.DEFINE_string('output_label_map', 'output.pbtxt', 'Path to output label map')
flags.DEFINE_string('input_images_dir', 'input', 'Path to input images')

FLAGS = flags.FLAGS

FILTER = {
    "Window" : 1,
    "Door" : 2,
    "Billboard" : 3,
}


def create_tf_example(example, normalize_size=True):
    filename = example['path'].encode() # Filename of the image. Empty if image is not from file

    im = Image.open(example['path'])
    print im.size

    if normalize_size:
        im = im.resize((800, 600), Image.ANTIALIAS)
        im_path = os.path.join('800x600', example['path'])
        im.save(im_path, "JPEG")
        example['path'] = im_path

    width = im.size[0]
    height = im.size[1]

    with tf.gfile.GFile(example['path'], 'rb') as fid:
        encoded_image_data = fid.read()


    xmin='xmin'
    ymin='ymin'
    x_width='x_width'
    y_height='y_height'
    #todo: use file extension
    image_format = 'jpg'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in example['annotations']:
	if not FILTER.get(box['class'], None):
            continue

        #if box['occluded'] is False:

        xmins.append(float(box[xmin]))
        xmaxs.append(float(box[xmin] + box[x_width]))
        ymins.append(float(box[ymin]))
        ymaxs.append(float(box[ymin] + box[y_height]))
        classes_text.append(box['class'].encode())
        classes.append(int(FILTER[box['class']]))

    tf_example = None

    if classes:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        }))

    return tf_example


def main(_):

    writer = tf.python_io.TFRecordWriter(FLAGS.output_tfrecord)

    input_dir = FLAGS.input_images_dir

    print("output file: {}, input dir: {}".format(FLAGS.output_tfrecord, input_dir))

    class_descriptions = pd.read_csv('class-descriptions-boxable.csv', names=['key', 'name'])
    print(class_descriptions.head())

    class_descriptions = class_descriptions[class_descriptions['name'].isin(list(FILTER.iterkeys()))]
    print(class_descriptions)
    FILTER_KEYS = {}
    for i, row in class_descriptions.iterrows():
        FILTER_KEYS[row['key']] = row['name']

# ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
# 000002b66c9c498e,xclick,/m/01g317,1,0.012500,0.195312,0.148438,0.587500,0,1,0,0,0
    bboxes_pd = pd.read_csv('train-annotations-bbox.csv')

    filter_ids = Set()
    for img in glob.glob(os.path.join(input_dir, '*.jpg')):
        filename = os.path.splitext(os.path.basename(img))[0]
        filter_ids.add(filename)

    bboxes_pd = bboxes_pd[bboxes_pd['ImageID'].isin(filter_ids)]

    for img in glob.glob(os.path.join(input_dir, '*.jpg')):
        try:
            filename = os.path.splitext(os.path.basename(img))[0]
            print("write {} to tf records, id is {}".format(img,filename))
            ex = dict()
            ex['path'] = img

            annotations = bboxes_pd[bboxes_pd['ImageID'] == filename]
            ex['annotations'] = []

            for index, row in annotations.iterrows():
                a = {}

                label_name = FILTER_KEYS.get(row['LabelName'], None)
                if not label_name:
                    continue
                a['class'] = label_name
                a['xmin'] = row['XMin']
                a['ymin'] = row['YMin']
                a['x_width'] = row['XMax'] - row['XMin']
                a['y_height'] = row['YMax'] - row['YMin']
                ex['annotations'].append(a)
            
            print("add tf_example {}".format(ex))

            tf_example = create_tf_example(ex)
            writer.write(tf_example.SerializeToString())
        except Exception as ex:
            print('exception {} \n while downlading {}'.format(img, ex))

    writer.close()

    print("Writing label map pbtxt...")

    with open(FLAGS.output_label_map, "w") as f:
       for key, value in FILTER.iteritems():
           f.write('''
item {{
  id: {}
  name: "{}"
}}'''.format(value, key))


if __name__ == '__main__':
    tf.app.run()
