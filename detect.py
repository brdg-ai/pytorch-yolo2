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

    ## Read frames from video
    video_reader = skvideo.io.FFmpegReader(videofile)
    video_filename = os.path.basename(videofile)
    video_filename_base = video_filename.split(".")[0]
    output_video_filename = f"{outputdir}/annotated-{video_filename}"
    video_writer = skvideo.io.FFmpegWriter(output_video_filename)
    frames = []
    for frame_num, frame in enumerate(video_reader.nextFrame(), start=1):
        pil_frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        frames.append(pil_frame)

    ## Prepare annotation output dir to put json and per image annotations
    annotation_output_dir = f'{outputdir}/annotations/{video_filename_base}'
    if not os.path.exists(annotation_output_dir):
        os.makedirs(annotation_output_dir)

    ## Prepare json output
    json_filename = f'{annotation_output_dir}/annotations-{video_filename_base}.jsonl'
    json_output = []

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
    for img_id, img in enumerate(frames, start=1):
        print(f"Analyzing frame {img_id}")

        sized = img.resize((m.width, m.height))
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

        ## Write detections to per-frame file to calculate mAP
        mAP_output_file = f"{annotation_output_dir}/image_{img_id}.txt"

        ## Add frame to output json
        frame_json = {}
        frame_json["frame_no"] = img_id
        frame_json["boxes"] = []
        with open(mAP_output_file, "w+") as mf:
            for i in range(len(boxes)):
                box = boxes[i]
                x1 = int(((box[0] - box[2]/2.0) * img.width).item())
                y1 = int(((box[1] - box[3]/2.0) * img.height).item())
                x2 = int(((box[0] + box[2]/2.0) * img.width).item())
                y2 = int(((box[1] + box[3]/2.0) * img.height).item())
                cls_conf = round(box[5].item(), 4)
                cls_id = box[6].item()
                class_name = class_names[cls_id].replace(" ", "-")
                # (0,0) from upper left
                box_json = {"x1": x1,
                            "x2": x2,
                            "y1": y1,
                            "y2": y2,
                            "conf":cls_conf,
                            "class_id":cls_id,
                            "class":class_name}
                frame_json["boxes"].append(box_json)
                # <class-name> <left> <top> <right> <bottom> [<difficult>]
                mAP_line = f"{class_name} {cls_conf} {x1} {y1} {x2} {y2}\n"
                mf.write(mAP_line)
        json_output.append(frame_json)

        ## Add frame to output video
        annotated_img = plot_boxes(img, boxes, None, class_names)
        annotated_img = np.array(annotated_img)
        video_writer.writeFrame(annotated_img)


    print(f"Writing per image annotations to {annotation_output_dir}")
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
