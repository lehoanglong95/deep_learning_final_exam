import argparse
import cv2
from gluoncv import model_zoo, data, utils
import mxnet
import sys

def get_args():
    parser = argparse.ArgumentParser("demo video")
    parser.add_argument("--model", type=str, default="ssd")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--input", type=str, default="test_videos/chi_street1_4k.mp4")
    parser.add_argument("--output", type=str, default="output_videos/ssd_512_resnet/chi_street1_4k_ssd.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    ctx = mxnet.cpu()
    if opt.model == "ssd":
        net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True, ctx=ctx)
    else:
        net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True, ctx=ctx)
    cap = cv2.VideoCapture(opt.input)
    out = cv2.VideoWriter(opt.output,  cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    count = 0
    while cap.isOpened():
        try:
            count += 1
            if count == 835:
                break
            flag, img = cap.read()
            img = mxnet.nd.array(img)
            x, img = data.transforms.presets.yolo.transform_test(img, short=512)
            class_IDs, scores, bounding_boxs = net(x)

            output_img = utils.viz.cv_plot_bbox(img, bounding_boxs[0], scores[0],
                                         class_IDs[0], class_names=net.classes)
            output_img = cv2.resize(output_img, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            # output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            out.write(output_img)
        except:
            break

    cap.release()
    out.release()


if __name__ == "__main__":
    opt = get_args()
    test(opt)
