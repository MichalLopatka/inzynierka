import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import math
import glob
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from IPython import display
import PIL
print(torch.__version__, torch.cuda.is_available())
import json
import gc
gc.collect()
torch.cuda.empty_cache()
setup_logger()
import cv2
from time import perf_counter

def movie_to_frames():
    vidcap = cv2.VideoCapture('own/clips/VID_20211008_173852.mp4')
    if not os.path.exists("own/clips/VID_20211008_173852"):
        os.makedirs("own/clips/VID_20211008_173852")
    success,image = vidcap.read()
    count = 0
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000/30)) 
        count=count+8
        if count < 10:
            cv2.imwrite("own/clips/VID_20211008_173852/frame000%d.png" % count, image)   
        elif count < 100:
            cv2.imwrite("own/clips/VID_20211008_173852/frame00%d.png" % count, image)
        elif count < 1000:
            cv2.imwrite("own/clips/VID_20211008_173852/frame0%d.png" % count, image)
        elif count < 10000:
            cv2.imwrite("own/clips/VID_20211008_173852/frame%d.png" % count, image)
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
            (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
            image.
    """
    a = a.clip(0, 255).astype('uint8')
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(PIL.Image.fromarray(a))

class DangerDetection:
    def __init__(self, threshhold, jupyter):
        self.DETECTRON2_DATASETS = "/datasets"
        self.init_config(threshhold)
        self.predictor = DefaultPredictor(self.cfg)
        self.dataset_dicts = DatasetCatalog.get("cityscapes_fine_instance_seg_test")
        self.jupyter = jupyter
    
    def init_config(self, threshhold):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  
        self.cfg.MODEL.DEVICE = "cpu"
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.cfg.MODEL.WEIGHTS = os.path.join(".\output", "model_part_5")  
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshhold
        self.cfg.DATASETS.TRAIN = ("cityscapes_fine_instance_seg_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 1
        self.cfg.SOLVER.IMS_PER_BATCH = 1
    
    def count_numbers(self, outputs, im):
        boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
        centers = outputs['instances'].pred_boxes.get_centers().cpu().numpy()
        areas = outputs['instances'].pred_boxes.area().cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        height, width, channels = im.shape
        return boxes, centers, areas, classes, height, width, channels
    
    def print_numbers(self, boxes, centers, areas, classes, height, width, channels):
        print(classes)
        print(boxes)
        print(height, width, channels)
        print(areas)
        
    def show_image(self, out):
        if self.jupyter:
            cv2_imshow(out)
        else:
            cv2.imshow("img", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    def detection(self):
        images = {}
        for d in sorted(glob.glob("own/clips/VID_20211008_170733/frame*.png")):
            strt = perf_counter()
            im = cv2.imread(d)        
            outputs = self.predictor(im)
            stp = perf_counter()
            v = Visualizer(im[:, :, ::-1],
                           scale=1,
                           instance_mode=ColorMode.IMAGE_BW
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            self.show_image(out)
            print(stp-strt)
    
    def dangerStatic(self, no_of_images, show_numbers):
        images = {}
        for d in random.sample(glob.glob("own/clips/VID_20211008_173852/frame*.png"), no_of_images):
            im = cv2.imread(d)
            outputs = self.predictor(im)  
            v = Visualizer(im[:, :, ::-1],
                           scale=1,
                           instance_mode=ColorMode.IMAGE_BW
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out = out.get_image()[:, :, ::-1]
            boxes, centers, areas, classes, height, width, channels = self.count_numbers(outputs, im)
            if show_numbers:
                self.print_numbers(boxes, centers, areas, classes, height, width, channels)
            
            # images[d["file_name"]] = {"height": height, "width": width, "boxes": []}
            safe = True
            
            for c, b, ce, a in zip(classes, boxes, centers, areas):
                
                if self.danger(c, b, ce, a):
                    # images[d["file_name"]]["boxes"].append({"class": c.tolist(),
                    #                                         "box": b.tolist(), "center": ce.tolist(), "area": a.tolist(), "safe": False})
                    safe = False
                    b=np.array(b)
                    out = cv2.rectangle(np.array(out), (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255), 8)
                # else:
                    # images[d["file_name"]]["boxes"].append({"class": c.tolist(),
                    #                                         "box": b.tolist(), "center": ce.tolist(), "area": a.tolist(), "safe": True})
            self.show_image(out)
            if not safe:
                print("ATTENTION")
                
        print(images)
        # with open('dropped/data.json', 'w') as fp:
        #     json.dump(images, fp)
            
    def dangerSequence(self, show_numbers, folder, start, stop):
        previous_boxes = []
        current_boxes = []
        for d in sorted(glob.glob(f"{folder}/*"))[start:stop]:
            im = cv2.imread(d)
            outputs = self.predictor(im)  
            v = Visualizer(im[:, :, ::-1],
                           scale=1,
                           instance_mode=ColorMode.IMAGE_BW
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out = out.get_image()[:, :, ::-1]
            boxes, centers, areas, classes, height, width, channels = self.count_numbers(outputs, im)
            if show_numbers:
                self.print_numbers(boxes, centers, areas, classes, height, width, channels)
            safe = True
            current_boxes.clear()
            for c, b, ce, a in zip(classes, boxes, centers, areas):
                current_boxes.append({"class": c, "box": b, "center": ce, "area": a})
                if self.danger(c, b, ce, a):
                    safe = False
            action, out = self.action(out, previous_boxes, current_boxes)
            self.show_image(out)
            if action != -1:
                print(action)
            if not safe:
                print("ATTENTION")
            
            previous_boxes = current_boxes.copy()
    
    def danger(self, class_type, box, center, area):
        bad = False
        # pedestrian
        center_dist = abs(center[0]-2048/2)
        if class_type in [0, 1]:
            if area > 15000 and center_dist < 400:
                bad = True
                print(class_type, ":", area)
            elif area > 15000 + ((center_dist - 400) * 15) and (center_dist) > 400:
                bad = True
                print(class_type, ":", area)
               
        # car
        if class_type == 2:
            if area > 60000 and (center_dist) < 400:
                bad = True
                print(class_type, ":", area)
            elif area > 60000 + ((center_dist - 400) * 150) and center_dist < 800 and center_dist > 400:
                bad = True
                print(class_type, ":", area)
                
        # big car
        if class_type == 3:
            if area > 80000 and (center_dist) < 400:
                bad=True
                print(class_type, ":", area)
            elif area > 80000 + ((center_dist - 400) * 200) and (center_dist) > 400:
                bad=True
                print(class_type, ":", area)
        # rails and buses
        if class_type in [4, 5]:
            if area > 200000:
                bad = True
                print(class_type, ":", area)
        # bicycles and motos
        if class_type in [6, 7]:
            if area > 20000 and (center_dist) < 700:
                bad = True
                print(class_type, ":", area)
        return bad

    def action(self, out, previous, current):
        for el_prev in previous:
            found = -1
            dist = 5000
            for el_curr in current:
                euclidean_dist = math.sqrt((el_prev["center"][0] - el_curr["center"][0])**2 + (el_prev["center"][1] - el_curr["center"][1])**2)
                if el_prev["class"] == el_curr["class"] and euclidean_dist < 700 and euclidean_dist < dist and el_curr["area"]/el_prev["area"] < 2.5 and el_curr["area"] > 10000 and el_prev["area"]>10000:
                    found = el_curr
                    dist = euclidean_dist
            if found != -1:
                area_ratio = found["area"]/el_prev["area"]
                center_dist = abs(found["center"][0]-2048/2)
                center_prev = abs(el_prev["center"][0]-2048/2)
                if area_ratio > 1.15 and found["class"] == 2 and found["area"] > 45000 and (center_dist) < 400:
                    curr = found["area"]
                    prev = el_prev["area"]
                    b = found["box"]
                    out = cv2.rectangle(np.array(out), (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255), 8)
                    return f"ATTENTION ACTION VEHICLE {curr}, {prev}", out
                elif area_ratio > 1.15 and found["class"] == 2 and found["area"] > 45000 + ((center_dist - 400) * 300) and center_dist > 400:
                    curr = found["area"]
                    prev = el_prev["area"]
                    b = found["box"]
                    out = cv2.rectangle(np.array(out), (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255), 8)
                    return f"ATTENTION ACTION VEHICLE {curr}, {prev}", out
                elif area_ratio > 1.15 and found["class"] == 1 and found["area"] > 5000 and (center_dist) < 600:
                    curr = found["area"]
                    prev = el_prev["area"]
                    b = found["box"]
                    out = cv2.rectangle(np.array(out), (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255), 8)
                    return f"ATTENTION ACTION RIDER {curr}, {prev}", out
                elif area_ratio > 1.15 and found["class"] == 0 and found["area"] > 10000 and (center_dist) < 600:
                    curr = found["area"]
                    prev = el_prev["area"]
                    b = found["box"]
                    out = cv2.rectangle(np.array(out), (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255), 8)
                    return f"ATTENTION ACTION PEDESTRIAN {curr}, {prev}", out
                elif  found["class"] in [0,1]  and found["area"] > 10000 and (center_dist) < 800 and center_dist < center_prev:
                    curr = found["area"]
                    prev = el_prev["area"]
                    b = found["box"]
                    out = cv2.rectangle(np.array(out), (int(b[0]),int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255), 8)
                    return f"ATTENTION ACTION PEDESTRIAN INCOMING {curr}, {prev}", out
        return -1, out

    def evaluator(self):
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=True)
        evaluator = COCOEvaluator("cityscapes_fine_instance_seg_val", tasks=("bbox",), distributed=False,
                                  output_dir="./output/")
        val_loader = build_detection_test_loader(self.cfg, "cityscapes_fine_instance_seg_val")
        print(inference_on_dataset(trainer.model, val_loader, evaluator))


if __name__ == '__main__':
    detection = DangerDetection(0.75, False)




