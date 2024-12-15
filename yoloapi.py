from yolo.darknet_images import *

def yolo_init(config_file, data_file,weight):
    random.seed(3)  # deterministic bbox colors
    network, class_names,class_colors = darknet.load_network(
        config_file,
        data_file,
        weight,
        batch_size=int(1)
    )
    return network, class_names, class_colors

def yolo_image(image_name,network, class_names, class_colors):
    #images = load_images('test_data\R_20231206_084548_082.png')
    #width, height = get_image_size(network)
    # after darknet detection, bbox = (center_x, center_y, w, h)
    image=image_name
    thresh=float(.25)
    image, detections = image_detection(
        image_name, network, class_names, class_colors,thresh 
    )
    boxs = []
    output = np.array([], dtype = np.float64).reshape(0, 3)
    ellipse_axis = np.array([], dtype = np.float64).reshape(0, 2)
    for label, confidence, bbox in detections:
        # extract feature, we need bbox = (left, top, w, h)
        center_x, center_y, w, h = bbox
        xmin = int(round(center_x - (w / 2)))
        ymin = int(round(center_y - (h / 2)))
        boxs.append([xmin, ymin, w, h])
        ori_center_x,ori_center_y=round(center_x*1.5384615384,10),round(center_y*1.15384615385,10)      #640/416=1.5384615384   480/416=1.15384615385
        ori_w,ori_h=round(w*1.5384615384,10),round(h*1.15384615385,10)
        area = np.pi*ori_w*ori_h / 4
        elps_ax = np.array( [ori_w,ori_h]).reshape(1, -1)  
        elps_ax.sort()  
        rect_=np.array([ori_center_x, ori_center_y, area]).reshape(1, 3)
        output = np.vstack((output, rect_))
        ellipse_axis = np.vstack((ellipse_axis, elps_ax))

    return output, ellipse_axis

