#!/usr/bin/python

# pip install lxml

# reference: https://github.com/yukkyo/voc2coco

import os
import json
import xml.etree.ElementTree as ET
import glob
from PIL import Image


START_BOUNDING_BOX_ID = 0

PRE_DEFINE_CATEGORIES = {"table": 0 }

# Assign images id according to their filenames
def get_id_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        idx = filename[7:12]
        return int(idx)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename[7:12]))


def convert(xml_files, json_file, data_path):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}

    categories = PRE_DEFINE_CATEGORIES

    bnd_id = START_BOUNDING_BOX_ID

    for xml_file in xml_files:

        # Read the structure of XML files
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.attrib['filename']

        # The image id must be a number
        image_id = get_id_as_int(filename)

        # Get the width and height of images
        im = Image.open(data_path+"/"+filename)
        width, height = im.size

        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)

        # Get boundung box points and covert them to Coco format 
        # (starting point[x], starting point[y], width, height)
        for table in root.findall('table'):
            coords = table.find('Coords').get('points')
            points = coords.replace(',',' ').split()
            points = list(map(int, points))
            o_width = abs(points[4] - points[0])
            o_height = abs(points[5] - points[1])
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [points[0], points[1], o_width, o_height],
                "category_id": 0,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1
            

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    # Save Coco format annotations in a json file
    os.makedirs(os.path.dirname("annotations/"+json_file), exist_ok=True)
    json_fp = open("annotations/"+json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == "__main__":

    # Dataset path 
    data_path = "./TRACKA_training_modern"
    
    xml_files = []
    for i in range(0, 420):
        xml_files = xml_files + glob.glob(os.path.join(data_path, "cTDaR_t"+str(10000+i)+".xml"))

    print("Number of xml files (train): {}".format(len(xml_files)))
    # Generate Coco format annotations
    convert(xml_files, "./train.json", data_path)

    xml_files = []
    for i in range(420, 540):
        xml_files = xml_files + glob.glob(os.path.join(data_path, "cTDaR_t"+str(10000+i)+".xml"))

    print("Number of xml files (val): {}".format(len(xml_files)))
    # Generate Coco format annotations
    convert(xml_files, "./val.json", data_path)

    xml_files = []
    for i in range(540, 600):
        xml_files = xml_files + glob.glob(os.path.join(data_path, "cTDaR_t"+str(10000+i)+".xml"))

    print("Number of xml files (test): {}".format(len(xml_files)))
    # Generate Coco format annotations
    convert(xml_files, "./test.json", data_path)



