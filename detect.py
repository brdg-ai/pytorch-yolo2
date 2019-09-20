import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

import scipy
import skvideo.io
import skvideo.utils
import skimage.transform
import jsonlines

def detect_video(cfgfile, weightfile, videofile, outputdir):

    ## Prepare json output
    json_filename = f'{outputdir}/annotations.jsonl'
    json_output = []

    ## Read frames from video
    video_reader = skvideo.io.FFmpegReader(videofile)
    video_filename = os.path.basename(videofile)
    output_video_filename = f"{outputdir}/annotated-{video_filename}"
    video_writer = skvideo.io.FFmpegWriter(output_video_filename)
    frames = []
    for frame_num, frame in enumerate(video_reader.nextFrame(), start=1):
        pil_frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        frames.append(pil_frame)

    ## Prepare model
    m = Darknet(cfgfile)
    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))
    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    use_cuda = 1
    if use_cuda:
        m.cuda()
    class_names = load_class_names(namesfile)

    ## Run model
    for img_id, img in enumerate(frames):
        sized = img.resize((m.width, m.height))
        
        print(f"Analyzing frame {img_id}")
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

        ## Add frame to output json
        frame_json = {}
        frame_json["frame_no"] = img_id
        frame_json["boxes"] = []
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = (box[0] - box[2]/2.0) * img.width
            y1 = (box[1] - box[3]/2.0) * img.height
            x2 = (box[0] + box[2]/2.0) * img.width
            y2 = (box[1] + box[3]/2.0) * img.height
            cls_conf = box[5]
            cls_id = box[6]
            box_json = {"x1": int(x1.item()),
                        "x2": int(x2.item()),
                        "y1": int(y1.item()),
                        "y2": int(y2.item()),
                        "conf":round(cls_conf.item(), 3),
                        "class_id":cls_id.item(),
                        "class":class_names[cls_id.item()]}
            frame_json["boxes"].append(box_json)
        json_output.append(frame_json)

        ## Add frame to output video
        annotated_img = plot_boxes(img, boxes, None, class_names)
        annotated_img = np.array(annotated_img)
        video_writer.writeFrame(annotated_img)

    with jsonlines.open(json_filename, mode='w') as json_writer:
        json_writer.write_all(json_output)
        print(f"Writing annotations to {json_filename}")

    print(f"Writing annotated video to {output_video_filename}")
    video_writer.close()

def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = Image.open(imgfile).convert('RGB')
    sized = img.resize((m.width, m.height))
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes(img, boxes, 'predictions.jpg', class_names)

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)

def detect_skimage(cfgfile, weightfile, imgfile):
    from skimage import io
    from skimage.transform import resize
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if m.num_classes == 20:
        namesfile = 'data/voc.names'
    elif m.num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'
    
    use_cuda = 1
    if use_cuda:
        m.cuda()

    img = io.imread(imgfile)
    sized = resize(img, (m.width, m.height)) * 255
    
    for i in range(2):
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        if i == 1:
            print('%s: Predicted in %f seconds.' % (imgfile, (finish-start)))

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes, savename='predictions.jpg', class_names=class_names)




if __name__ == '__main__':
    if len(sys.argv) == 5:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        outputdir = sys.argv[4]
        detect_video(cfgfile, weightfile, imgfile, outputdir)
        #detect(cfgfile, weightfile, imgfile)
        #detect_cv2(cfgfile, weightfile, imgfile)
        #detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile outputdir')
        #detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
