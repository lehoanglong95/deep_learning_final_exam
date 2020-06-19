from gluoncv import model_zoo, data, utils
import cv2
import argparse

def get_args():
    parser = argparse.ArgumentParser("YOLO V3")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--input", type=str, default="test_images/pedestrian_imgs.jpg")
    parser.add_argument("--output", type=str, default="output_imgs/ssd_512_resnet50/pedestrian_prediction.jpg")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    opt = get_args()
    input_img = opt.input
    output_img = opt.output
    net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

    x, img = data.transforms.presets.yolo.load_test(input_img, short=512)

    class_IDs, scores, bounding_boxs = net(x)

    img = utils.viz.cv_plot_bbox(img, bounding_boxs[0], scores[0],
                                 class_IDs[0], class_names=net.classes)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_img, RGB_img)