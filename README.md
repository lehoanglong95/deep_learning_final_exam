#THIS IS SOURCE CODE FOR FINAL EXAM OF DEEP LEARNING COURSE.
#AUTHOR: Le Hoang Long - Sejong University
* Install python packages: pip install -r requirement.txt
* Images object detection:
    * YOLOv2: 
        * Download "whole_model_trained_yolo_voc" in [YOLO trained models](https://drive.google.com/drive/folders/1Ee6FHQTGuJpNRYSa8DtHWzu4yWNyc7sp).
        * Copy pretrain model to trained_models
        * python test_voc_images.py --input your_input_img --output your_output_folder
    * YOLOv3:
        * python demo_ssd.py --image_size 512 --input your_input_img --output your_output_folder
    * SSD512:
        * python demo_yolo.py --image_size 512 --input your_input_img --output your_output_folder
* Videos object detection:
    * YOLOv2:
        * python test_voc_video.py --input your_input_video --output your_output_folder
    * YOLOv3:
        * python demo_video.py --model yolov3 --input your_input_video --output your_output_folder
    * SSD512:
        * python demo_video.py --model ssd --input your_input_video --output your_output_folder       
        