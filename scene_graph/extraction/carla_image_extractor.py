#Copied from https://github.com/AICPS/roadscene2vec
#Copyright (c) 2021 UC Irvine Advanced Integrated Cyber-Physical Systems Lab (AICPS)

import os
import pdb
import cv2
from pathlib import Path
from os.path import isfile, join
import sys
sys.path.append(os.path.dirname(sys.path[0]))
from scene_graph.extraction.image_extractor import RealExtractor
from scene_graph.scene_graph import SceneGraph
from tqdm import tqdm

class CarlaRealExtractor(RealExtractor):
    def __init__(self, config):
        super(CarlaRealExtractor, self).__init__(config)
        
    def parse_img_info(self, path, seq):
        raw_images_loc = (path/'sensor/vehicle').resolve()
        self.dataset.scene_graphs[seq] = dict()
        for vehicle_idx in sorted(os.listdir(raw_images_loc))[1:]:
            direction_paths = dict()
            for direction in self.conf.sensor_directions:
                direction_paths[direction] = []
                direction_path = (raw_images_loc/vehicle_idx/'perception/camera/Camera RGB'/direction).resolve()
                image_paths = \
                sorted([(direction_path/f).resolve() for f in os.listdir(direction_path) if isfile(join(direction_path, f)) and ".DS_Store" not in f and "Thumbs" not in f], 
                       key = lambda x: int(x.stem.split(".")[0]))
                modulo = 0
                acc_number = 0
                if(self.framenum != None):
                    modulo = int(len(image_paths) / self.framenum)  #subsample to frame limit
                if(self.framenum == None or modulo == 0):
                    modulo = 1
                for i in range(0, len(image_paths)):
                    if (i % modulo == 0 and self.framenum == None) or (i % modulo == 0 and acc_number < self.framenum):
                        image_path = image_paths[i]
                        im = cv2.imread(str(image_path), cv2.IMREAD_COLOR) 
                        out_img_path = None
                        bounding_boxes = self.get_bounding_boxes(img_tensor=im, out_img_path=out_img_path)
                        scenegraph = SceneGraph(self.relation_extractor,    
                                                bounding_boxes = bounding_boxes, 
                                                bev = self.bev,      
                                                coco_class_names=self.coco_class_names, 
                                                platform=self.dataset_type)
                        direction_paths[direction].append(scenegraph)
                        acc_number += 1
                        
            self.dataset.scene_graphs[seq][vehicle_idx] = direction_paths

        
    def parse_label(self, path, seq):
        import random
        self.dataset.labels[seq] = random.choice([1.0, 0.0])
    
    def load(self): #seq_tensors[seq][frame/jpgname] = frame tensor
        try:
            all_sequence_dirs = []
            seq = 0
            for scenario_path in Path(self.input_path).iterdir():
                if scenario_path.is_dir():
                    for seq_path in scenario_path.iterdir():
                        all_sequence_dirs.append(seq_path)
                        seq += 1
                        self.dataset.action_types[seq] = scenario_path.stem.split('_')[0] 
                        scenario_path.stem.split('_')[0]
                        print(self.dataset.action_types[seq])
            self.dataset.folder_names = [path.stem for path in all_sequence_dirs]
            for seq, path in enumerate(tqdm(all_sequence_dirs)):
                self.parse_img_info(path, seq)
                self.parse_label(path, seq)

        except Exception as e:
            import traceback
            traceback.print_exc()
            pdb.set_trace()
            print('We have problem creating the real image scenegraphs')
            print(e)
                        