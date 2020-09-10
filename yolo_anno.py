import glob
import json
import os
import xml.etree.ElementTree as ET
import xmljson
import random

def yolo_ann(xmin, ymin, xmax, ymax, W, H):
    dw = 1. / (W)
    dh = 1. / (H)
    x = (int(xmin) + int(xmax)) / 2.0 - 1
    y = (int(ymin) + int(ymax)) / 2.0 - 1
    w = int(xmax)-int(xmin)
    h = int(ymax)-int(ymin)
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (str(x), str(y), str(w), str(h))


def generate_config_files():
    folder_path = os.path.realpath("./images")
    all_annot = glob.glob("./annotations/*.xml")
    class_names = ["with_mask", "without_mask", "mask_weared_incorrect"]
    img_paths = []
    for annot in all_annot:
        tree = ET.parse(annot)
        root = tree.getroot()
        anno_json = json.loads(json.dumps(xmljson.parker.data(root)))
        path_img = os.path.join(folder_path, anno_json["filename"])
        img_paths.append(path_img)
        img_w = anno_json["size"]["width"]
        img_h = anno_json["size"]["height"]
        ann_str = ""
        if isinstance(anno_json["object"], list):
            for obj in anno_json["object"]:
                yolo_box = yolo_ann(obj["bndbox"]["xmin"], obj["bndbox"]["ymin"], obj["bndbox"]["xmax"], obj["bndbox"]["ymax"], img_w, img_h)
                if obj["name"] not in class_names:
                    class_names.append(obj["name"])
                ann_str += str(class_names.index(obj["name"])) + " " + " ".join(yolo_box) + "\n"
        else:
            yolo_box = yolo_ann(anno_json["object"]["bndbox"]["xmin"], anno_json["object"]["bndbox"]["ymin"], anno_json["object"]["bndbox"]["xmax"], anno_json["object"]["bndbox"]["ymax"],
                                img_w, img_h)
            if anno_json["object"]["name"] not in class_names:
                class_names.append(anno_json["object"]["name"])
            ann_str += str(class_names.index(anno_json["object"]["name"])) + " " + " ".join(yolo_box) + "\n"

        f = open("./images/"+os.path.basename(path_img).replace(".png", ".txt"), "w")
        f.write(ann_str)
        f.close()
    f = open("data/mask.names", "w")
    f.write("\n".join(class_names))
    f.close()

    # for train test split
    random.shuffle(img_paths)
    f = open("./data/train.txt", "w")
    f.write("\n".join(img_paths))
    f.close()
    f = open("./data/test.txt", "w")
    f.write("")
    f.close()

    # writing the data file
    f = open("./data/mask.data", "w")
    ini_cfg = "classes= 3\n" \
              "train  = {}\n" \
              "valid  = {}\n" \
              "names = {}\n" \
              "backup = {}\n".format(os.path.realpath("./data/train.txt"),
                                     os.path.realpath("./data/test.txt"),
                                     os.path.realpath("./data/mask.names"),
                                     os.path.realpath("./data/weights/"))
    f.write(ini_cfg)
    f.close()

    # writing the congig ini file
    f = open("./cfg/yolov4_mask.ini", "w")
    ini_cfg = "[yolo]\n" \
              "cfg_file={}\n" \
              "weight_file={}\n" \
              "data_file={}\n".format(os.path.realpath("./data/yolov4-mask.cfg"),
                                      os.path.realpath("./data/weights/yolov4-mask_final.weights"),
                                      os.path.realpath("./data/mask.data"))
    f.write(ini_cfg)
    f.close()
