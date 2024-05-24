# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license
#Release v10.0.51
import argparse
from functools import partial
from pathlib import Path

import torch
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from PIL import Image
import os
import cv2
import csv

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer
from collections import deque

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', )) 

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results

# variables
id_buffers = {}
id_position_buffers = {}
disappearing_position_prediction_buffer ={}
disappearing_direction_prediction_buffer ={}
total_count = 0
novelty_frame = []

#set person
person = 0
save_inference = 0

source_folder = os.path.abspath(os.path.join(os.getcwd(), '../dataset/dataset3'))

for file_name in os.listdir(source_folder):
    if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
        first_image_path = os.path.join(source_folder, file_name)
        break 

image = Image.open(first_image_path)
frame_width, frame_height = image.size


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


def buffers_update(id_buffers,id_position_buffers,ids,boxes,clses):
    #update the id_buffers and id_position buffers
    
    for id in id_buffers:
        if id not in ids:
            id_buffers[id].append(0) # signal buffer
            id_position_buffers[id].append((0, 0)) # position buffer
            
    for id, box,cls in zip(ids,boxes,clses):
        id = int(id.item())
        #if cls != 2 and cls != 7: 
        if person == 1:
            if cls != 0: 
                continue
        elif person == 0:
            if cls != 2 and cls != 7: 
                continue
        x1,y1,x2,y2 = box[:4]
        width = x2 - x1
        height = y2 - y1
        if width*height < opt.roi_threshold:
            if id in id_buffers:
                id_buffers[id].append(0)
                id_position_buffers[id].append((0,0))
        else:
            if id in id_buffers:
                id_buffers[id].append(1) # signal buffer
                x1, y1, x2, y2 = box[:4] # position buffer
                center_x = ((x1 + x2) / 2).item()
                center_y = ((y1 + y2) / 2).item()
                id_position_buffers[id].append((center_x, center_y))
            else:
                id_buffers[id] = deque([1], maxlen=opt.buffer_size) # signal buffer
                x1, y1, x2, y2 = box[:4] # position buffer 
                center_x = ((x1 + x2) / 2).item()
                center_y = ((y1 + y2) / 2).item()
                id_position_buffers[id] = deque([(center_x, center_y)], maxlen=opt.buffer_size)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def disappearing_position_prediction(id_position_buffers,disappearing_ids,disappearing_position_prediction_buffer,disappearing_direction_prediction_buffer):
    # create a fixed number of predictions for disappearing_ids
    for id in disappearing_ids:
        if (len(id_position_buffers[id]) >= opt.predict_size + 1):
            disappearing_position_prediction_buffer[id]= posestimate(id_position_buffers,id,disappearing_direction_prediction_buffer)
        else:
            pass

def disappearing_position_prediction_buffer_update(disappearing_position_prediction_buffer):
    
    # After every loop pop one

    # the significance of key_to_delete is to delete the key after iteration to avoid exception
    keys_to_delete = []  

    # If the stack is not empty, pop the earliest position predcition
    for id in disappearing_position_prediction_buffer:
        if disappearing_position_prediction_buffer[id]:  # If the stack is not empty
            disappearing_position_prediction_buffer[id].pop()
            # if there is no element left in the stack(after poping), delete the corresponsing disappearing id
            if len(disappearing_position_prediction_buffer[id]) == 0:
                keys_to_delete.append(id)
        else: # if there is no element left in the stack, delete the corresponsing disappearing id
            keys_to_delete.append(id)
    
    # Delete the keys after the iteration
    for key in keys_to_delete:
        del disappearing_position_prediction_buffer[key]


def disappear_id_filter(id_position_buffers, disappear_id, frame_width, frame_length):
   #Objects cannot disappear within 10 units of the edge, so I will record the IDs of those that do disappear within 10 units of the edge. Only these IDs will be eligible to match with the appear_id
    border_distance = 10
    filtered_disappear_id = []

    for id in disappear_id:
        x, y = id_position_buffers[id][-2]  #Retrieve the coordinates of the last position.
        
        #Check if it is within 10 units of the edge.
        if x > border_distance and x < frame_width - border_distance:
            
            filtered_disappear_id.append(id)
    
    return filtered_disappear_id



def appear_disappear_ids_match(id_buffers,id_position_buffers, appearing_ids, disappearing_position_prediction_buffer,prev_list,disappearing_direction_buffer):
    
    #id to be deleted from disappearing_position_prediction_buffer due to appear_id = disappear_id
    #For each appearing object (appear_id), it calculates the Euclidean distance between its position (center_x1, center_y1) and the predicted position (center_x2, center_y2) of each disappearing object (disappear_id).
    for appear_id in appearing_ids:
        
        delete_id = []
        min_distance = 500 # dummmy intialize
        min_appear_id = None  # Initialize min_appear_id as None
        min_disappear_id = None  # Initialize min_disappear_id as None
        center_x1, center_y1 = id_position_buffers[appear_id][-1]
        #appear_direction = get_direction(id_position_buffers, appear_id)

        #If the current distance is less than the minimum distance (min_distance), it updates the minimum distance along with the corresponding appearing object ID (min_appear_id) and disappearing object ID (min_disappear_id).
        #If the appear_direction is None, do not consider the angle comparison

        for disappear_id in disappearing_position_prediction_buffer:
            center_x2, center_y2 = disappearing_position_prediction_buffer[disappear_id][-1]
            distance = euclidean_distance((center_x1, center_y1), (center_x2, center_y2))
            
            # get the closest two ids in terms of distance   
            if distance < opt.diff_threshold : 
                min_appear_id = appear_id
                min_disappear_id = disappear_id   
        
        # if no matching, returns 0
        if min_appear_id is None and min_disappear_id is None:
            return 0
        
        else: 
            if(min_appear_id == min_disappear_id) :  

                delete_id.append(min_disappear_id) # at the end of this function, delete min_appear_id = min_disappear_id from disppearing_position_prediction_buffer

                # nothing need to be done to id_buffers
                # the novelty assessment of this will be carried out in the noveltysignal function

            else: 
                # for example id_list
                # 1st frame [1,2,3] 
                # 2nd frame [1,2]
                # 3rd frame [1,2,5]
                # 3 and 5 match, this means 5 is actually 3
                # prev_list = 3rd frame, remove 5 and append 3
                # also update the id_buffers and id_position_buffers

                prev_list.remove(min_appear_id) # if appear_id has a matched disappear_id, disappear id is removed from the prev_list as if it has not occured
                prev_list.append(min_disappear_id) # if appear_id has a matched disappear_id, disappear id is added to the prev_list as if it has occured
        
                id_buffers[min_disappear_id][-1] = 1
                id_position_buffers[min_disappear_id].append(id_position_buffers[min_appear_id][-1])
                id_buffers[min_appear_id][-1] = 0
                id_position_buffers[min_appear_id][-1] = (0,0)

                delete_id.append(min_disappear_id)

            for id in delete_id:
                del disappearing_position_prediction_buffer[id]

    return 0
    


def posestimate(id_position_buffers, id, disappearing_direction_prediction_buffer):
    #example [(1,1),(2,2),(3,3)] -> [(8,8),(7,7),(6,6),(5,5),(4,4)]
    #pop returns (4,4)

    numpred = 30  # number of prediction

    #sample a sequence of past positions
    end = -1  
    start = - 1 - opt.predict_size 
    past_pos = list(id_position_buffers[id])[start:end]
    

    past_pos = interpolate_zeros(past_pos)

    # Prepare the data for linear regression
    X = [[i] for i in range(len(past_pos))]  # the indices of the positions
    y = past_pos  # the positions

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next numpred positions
    future_X = [[i] for i in range(len(past_pos), len(past_pos) + numpred)]
    future_pos = model.predict(future_X)
    future_pos = [tuple(pos) for pos in future_pos]  # convert each predicted position to a tuple

    return future_pos[::-1]


def noveltysignal(id_buffers,id_position_buffers,ignore_list):
    
    novelty_signal = False

    for id in id_buffers:
        if len(id_buffers[id]) == 1 : # if very first appears, always novelty
                novelty_signal = True

        else:
            cur_state = id_buffers[id][-1]
            prev_state = id_buffers[id][-2]  

            if (cur_state == 1) and (prev_state == 0):
                  
                cur_position = id_position_buffers[id][-1]

                prev_position = (0, 0) 

                # to deal with the oscillation scenerio
                for idx, position in reversed(list(enumerate(id_position_buffers[id]))):

                    if idx == len(id_position_buffers[id]) - 2 or position != (0, 0):
                        prev_position = position
                        break
                    
                distance = euclidean_distance(cur_position, prev_position)
                if distance >= opt.movement_threshold:  
                    novelty_signal = True                   

    return novelty_signal

def get_direction(id_position_buffers, appearing_id):

    # If there is only one frame, we cannot compute the direction
    if len(id_position_buffers[appearing_id]) == 1:
        return None

    # If there are less than 10 frames, we compute the direction based on all the previous frames
    elif len(id_position_buffers[appearing_id]) < 10:
        past_pos = list(id_position_buffers[appearing_id])
    # If there are 10 or more frames, we compute the direction based on the last 10 frames
    else:
        past_pos = list(id_position_buffers[appearing_id])[-10:]
    # Convert the positions to a numpy array
    past_pos = np.array(past_pos)

    # Compute the differences between consecutive positions
    differences = np.diff(past_pos, axis=0)

    # Compute the average difference
    average_difference = np.mean(differences, axis=0)

    # Compute the direction
    direction = np.arctan2(average_difference[1], average_difference[0])

    direction = np.degrees(direction)

    return direction

def interpolate_zeros(lst):
    # cancel the zeros due to oscillation
    # example [(x1,y1), (0,0), (x3,y3), (0,0), (0,0), (x6,y6)] -> [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]

    arr = np.array(lst, dtype=float,copy=False)
    idx = np.arange(len(arr))
    interp_x = np.interp(idx, idx[arr[:, 0] != 0], arr[arr[:, 0] != 0, 0])
    interp_y = np.interp(idx, idx[arr[:, 1] != 0], arr[arr[:, 1] != 0, 1])
    interp  =  [(interp_x[i],  interp_y[i])  for  i  in  range(len(interp_x))] 

    return interp

def angle_difference(angle1, angle2):
    return abs(angle1 - angle2)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_boxes(image, boxes, ids):
    """
    Draw bounding boxes with IDs on the image.
    """
    for box, id in zip(boxes, ids):
        x1, y1, x2, y2 = [int(b) for b in box[:4]]
        label = f'ID {id}'
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put text for ID
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=args.agnostic_nms,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )


    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))

    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args
    

    csv_file_path = os.path.join(os.getcwd(), 'examples\predictedlabel.csv')
    f = open(csv_file_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['label'])  # Write the header
    result_dir = Path.cwd() / 'resultshow'
    ensure_dir(result_dir)
    # local variable
    loop = True
    ignore_list = [] # to ignore the dedicated signals
    for frame_idx, r in enumerate(results):

        if r.boxes.data.shape[1] == 7:
            
            orig_img = r.orig_img

            boxes = r.boxes.xyxy.cpu().numpy().tolist()  
            ids = r.boxes.id.cpu().numpy().tolist()

            if (save_inference == 1):
                # save inference 
                save_path = result_dir / f"frame_{frame_idx}.jpg"

                orig_img_cv = np.array(orig_img)  # Convert PIL image to numpy array
                orig_img_cv = cv2.cvtColor(orig_img_cv, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

                # Draw bounding boxes on the image
                drawn_img = draw_boxes(orig_img_cv, boxes, ids)
                cv2.imwrite(str(save_path), drawn_img)
                
            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )
            if len(r.boxes) == 0:  # If no objects are tracked in the current frame.
                for id in id_buffers:
                    id_buffers[id].append(0)  # For all IDs, add a 0 to their corresponding buffer.
                    id_position_buffers[id].append((0, 0))  # Simultaneously update the position buffer.

            else:
                if loop == True:
                    loop = False
                    buffers_update(id_buffers,id_position_buffers,r.boxes.id,r.boxes.xyxy,r.boxes.cls)
                    prev_list = r.boxes.id.tolist()
                    cls_list = r.boxes.cls.tolist()
                    if (person == 1):
                        prev_list = [id for id, cls,box in zip(prev_list, cls_list, r.boxes.xyxy ) if (cls == 0) and (box[2]-box[0])*(box[3]-box[1]) >= opt.roi_threshold]    
                    
                    else:
                        prev_list = [id for id, cls,box in zip(prev_list, cls_list, r.boxes.xyxy ) if (cls == 2 or cls == 7) and (box[2]-box[0])*(box[3]-box[1]) >= opt.roi_threshold]    
                        
                else:

                    if frame_idx == 3:
                        total_count = 1;
                    id_list = r.boxes.id.tolist()
                    cls_list = r.boxes.cls.tolist()
                    if (person == 1):
                        id_list = [id for id, cls,box in zip(id_list, cls_list, r.boxes.xyxy) if (cls == 0) and (box[2]-box[0])*(box[3]-box[1]) >= opt.roi_threshold]
                    else:
                        id_list = [id for id, cls,box in zip(id_list, cls_list, r.boxes.xyxy) if (cls == 2 or cls == 7) and (box[2]-box[0])*(box[3]-box[1]) >= opt.roi_threshold]
                    
                    buffers_update(id_buffers,id_position_buffers,r.boxes.id,r.boxes.xyxy,r.boxes.cls)
                    
                    appearing_ids = list(set(id_list) - set(prev_list))
                    
                    disappearing_ids = list(set(prev_list) - set(id_list))

                    disappearing_ids= disappear_id_filter(id_position_buffers, disappearing_ids, frame_width, frame_height)
                    
                    prev_list = id_list

                    # predict the future position of disappearing ids
                    disappearing_position_prediction(id_position_buffers,disappearing_ids,disappearing_position_prediction_buffer,disappearing_direction_prediction_buffer)

                    appear_disappear_ids_match(id_buffers,id_position_buffers, appearing_ids,disappearing_position_prediction_buffer,prev_list,disappearing_direction_prediction_buffer)

                    # everyround pop one and delete the ids that have length of zero                    disappearing_position_prediction_buffer_update(disappearing_position_prediction_buffer)

            novelty_signal = noveltysignal(id_buffers,id_position_buffers,ignore_list)
        
        if novelty_signal:
            novelty_frame.append(frame_idx)
            writer.writerow([1]) 
            print('NOVELTY')
        else:
            writer.writerow([0])
    print(novelty_frame)

    
    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')

    f.close()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8x',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default= source_folder,
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', default = False, action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--agnostic-nms', default=False, action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--roi-threshold', type=int, default=10, 
                        help='Description of ROI threshold')
    parser.add_argument('--buffer-size', type=int, default=30, 
                        help='Description of buffer size')
    parser.add_argument('--predict-size', type=int, default=20, 
                        help='how many past data used to predict')
    parser.add_argument('--movement-threshold', type=int, default=30, 
                        help='Description of movement threshold')
    parser.add_argument('--diff-threshold', type=int, default=20, 
                        help='Description of difference threshold')

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)

