import argparse
from yolo3.yolo import YOLO
from yolo3.video import detect_video
from PIL import Image
from yolo3 import utils
from enum import Enum


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        # try:
        image = Image.open(img)
        boxes, scores, classes = yolo.detect(image)
        yolo.class_names
        mod_image = utils.draw_results(image, boxes, scores, classes,
                                       yolo.class_names, yolo.colors)
        mod_image.show()
        # except Exception as ex:
        #     print('got exception', ex)

    yolo.close_session()


class InputType(Enum):
    image = 'image'
    video = 'video'

    def __str__(self):
        return self.value


if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' +
        YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' +
        YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' +
        YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' +
        str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--input_type', type=InputType, choices=list(InputType),
        help='the type of input to process: image or video'
    )

    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    args = parser.parse_args()

    if args.input_type == InputType.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        detect_img(YOLO(**vars(args)))
    elif args.input_type == InputType.video:
        detect_video(YOLO(**vars(args)), args.input, args.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
