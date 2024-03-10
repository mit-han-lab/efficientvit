# Code credit: [FastSAM Demo](https://huggingface.co/spaces/An-619/FastSAM).

import torch
import gradio as gr
from PIL import ImageDraw
# from utils.tools_gradio import fast_process
import copy
import argparse
import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Rectangle
from PIL import Image

from efficientvit.apps.utils import parse_unknown_args
from efficientvit.models.efficientvit.sam import EfficientViTSamAutomaticMaskGenerator, EfficientViTSamPredictor
from efficientvit.models.utils import build_kwargs_from_config
from efficientvit.sam_model_zoo import create_sam_model
from demo_sam_model import  draw_bbox ,draw_binary_mask,draw_scatter,load_image,cat_images,show_anns


parser = argparse.ArgumentParser(
    description="Host  EfficientViT-SAM as a local web service."
)

parser.add_argument(
    "--server-name",
    default="127.0.0.1",
    type=str,
    help="The server address that this demo will be hosted on."
)
parser.add_argument(
    "--port",
    default=8080,
    type=int,
    help="The port that this demo will be hosted on."
)

parser.add_argument("--model", default='xl1',type=str)


parser.add_argument("--weight_url", type=str, default=None)
parser.add_argument("--multimask", action="store_true")

# EfficientViTSamAutomaticMaskGenerator args
parser.add_argument("--pred_iou_thresh", type=float, default=0.8)
parser.add_argument("--stability_score_thresh", type=float, default=0.85)
parser.add_argument("--min_mask_region_area", type=float, default=100)

args, opt = parser.parse_known_args()
opt = parse_unknown_args(opt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

efficientvit_sam = create_sam_model(args.model, True, args.weight_url).cuda().eval()
efficientvit_sam_predictor = EfficientViTSamPredictor(efficientvit_sam)
efficientvit_mask_generator = EfficientViTSamAutomaticMaskGenerator(
    efficientvit_sam,
    pred_iou_thresh=args.pred_iou_thresh,
    stability_score_thresh=args.stability_score_thresh,
    min_mask_region_area=args.min_mask_region_area,
    **build_kwargs_from_config(opt, EfficientViTSamAutomaticMaskGenerator),
)


examples = [
    ["assets/fig/cat.jpg"],
    ["assets/fig/1.jpeg"],
    ["assets/fig/2.jpeg"],
    ["assets/fig/3.jpeg"],
    ["assets/fig/4.jpeg"],
    ["assets/fig/5.jpeg"],
    ["assets/fig/6.jpeg"],
    ["assets/fig/7.jpeg"],
]

# Description
title = "<center><strong><font size='8'> EfficientViT-SAM<font></strong> <a href='https://github.com/mit-han-lab/efficientvit'><font size='6'>[GitHub]</font></a> </center>"

description_s = """ # Instructions for everything mode

                1. Upload an image or click one of the provided examples.
                2. Click the Segment Everything button.
                3. The Reset button resets the image.

              """

description_p = """ # Instructions for point mode

                1. Upload an image or click one of the provided examples.
                2. Select the point type.
                3. Click once or multiple times on the image to indicate the object of interest.
                4. The Clear button clears all the points.
                5. The Reset button resets both points and the image.

              """

description_b = """ # Instructions for box mode

                1. Upload an image or click one of the provided examples.
                2. Click twice on the image (diagonal points of the box).
                3. The Clear button clears the box.
                4. The Reset button resets both the box and the image.

              """

css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

global_points = []
global_point_label = []
global_box = []
global_image = None
global_image_with_prompt = None
tmp_file = f".tmp_{time.time()}.png"

def draw_mask(
    image: np.ndarray,
    mask,
    tmp_name=".tmp.png",
) -> np.ndarray:
    dpi = 300
    oh, ow, _ = image.shape
    plt.close()
    plt.figure(1, figsize=(oh / dpi, ow / dpi))
    plt.imshow(image)
    show_anns(mask)
    plt.axis("off")
    plt.savefig(tmp_name, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    image = cv2.resize(load_image(tmp_name), dsize=(ow, oh))
    os.remove(tmp_name)
    plt.close()
    return image

def reset():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    global_image_with_prompt = None
    return None


def reset_all():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image = None
    global_image_with_prompt = None
    return None, None,None


def clear():
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []
    global_image_with_prompt = copy.deepcopy(global_image)
    return global_image


def on_image_upload(image, input_size=1024):
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    global_image = copy.deepcopy(image)
    global_image_with_prompt = copy.deepcopy(image)
    # print("Image changed")
    nd_image = np.array(global_image)
    efficientvit_sam_predictor.set_image(nd_image)

    return image

def image_resize(image,input_size=1024):
    global global_points
    global global_point_label
    global global_box
    global global_image
    global global_image_with_prompt
    global_points = []
    global_point_label = []
    global_box = []

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    global_image = copy.deepcopy(image)
    global_image_with_prompt = copy.deepcopy(image)
    # print("Image changed")
    nd_image = np.array(global_image)

    return image

def convert_box(xyxy):
    min_x = min(xyxy[0][0], xyxy[1][0])
    max_x = max(xyxy[0][0], xyxy[1][0])
    min_y = min(xyxy[0][1], xyxy[1][1])
    max_y = max(xyxy[0][1], xyxy[1][1])
    xyxy[0][0] = min_x
    xyxy[1][0] = max_x
    xyxy[0][1] = min_y
    xyxy[1][1] = max_y
    return xyxy


def segment_anything():
    global global_image_with_prompt
    image = global_image_with_prompt
    raw_image = np.array(image)
    masks = efficientvit_mask_generator.generate(raw_image)
    plots = draw_mask(raw_image,masks)
    return plots


def segment_with_points(
        label,
        evt: gr.SelectData,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=False,
):
    global global_points
    global global_point_label
    global global_image_with_prompt

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 5, (97, 217, 54) if label == "Positive" else (237, 34, 13)
    global_points.append([x, y])
    global_point_label.append(1 if label == "Positive" else 0)

    # print(f'global_points: {global_points}')
    # print(f'global_point_label: {global_point_label}')

    draw = ImageDraw.Draw(global_image_with_prompt)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = global_image_with_prompt

    raw_image = np.array(image)
    global_points_np = np.array(global_points)
    global_point_label_np = np.array(global_point_label)
    masks, _, _ = efficientvit_sam_predictor.predict(
        point_coords=global_points_np,
        point_labels=global_point_label_np,
        multimask_output=args.multimask,
    )
    
    plots = [
        draw_scatter(
            draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
            global_points_np,
            color=["g" if l == 1 else "r" for l in global_point_label_np],
            s=10,
            ew=0.25,
            tmp_name=tmp_file,
        )
        for binary_mask in masks
    ]
    plots = cat_images(plots, axis=1)
    return plots


def segment_with_box(
        evt: gr.SelectData,
        input_size=1024,
        better_quality=False,
        withContours=True,
        use_retina=True,
        mask_random_color=False,
):
    global global_box
    global global_image
    global global_image_with_prompt

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color, box_outline = 5, (97, 217, 54), 5
    box_color = (0, 255, 0)

    if len(global_box) == 0:
        global_box.append([x, y])
    elif len(global_box) == 1:
        global_box.append([x, y])
    elif len(global_box) == 2:
        global_image_with_prompt = copy.deepcopy(global_image)
        global_box = [[x, y]]

    # print(f'global_box: {global_box}')
    draw = ImageDraw.Draw(global_image_with_prompt)
    draw.ellipse(
        [(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)],
        fill=point_color,
    )
    image = global_image_with_prompt

    if len(global_box) == 2:
        global_box = convert_box(global_box)
        # xy = (global_box[0][0], global_box[0][1], global_box[1][0], global_box[1][1])
        # draw.rectangle(
        #     xy,
        #     outline=box_color,
        #     width=box_outline
        # )
        raw_image = np.array(image)
        global_box_np = np.array(global_box)
        global_box_np.reshape(1,-1)
        masks, _, _ = efficientvit_sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=global_box_np,
            multimask_output=args.multimask,
        )
        global_box_np =[global_box_np[0][0],global_box_np[0][1],global_box_np[1][0],global_box_np[1][1]]
        plots = [
            draw_bbox(
                draw_binary_mask(raw_image, binary_mask, (0, 0, 255)),
                [global_box_np],
                color="g",
                tmp_name=tmp_file,
            )
            for binary_mask in masks
        ]
        plots = cat_images(plots, axis=1)
        return plots
    return image

img_s = gr.Image(label="Input SAM", type="pil")
img_p = gr.Image(label="Input with points", type="pil")
img_b = gr.Image(label="Input with box", type="pil")

with gr.Blocks(css=css, title="EfficientSAM") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Tab("Everything mode") as tab_s:
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_s.render()
            with gr.Column(scale=1):
                with gr.Row():
                    with gr.Column():
                        sam_btn_s = gr.Button("Segment Everything", variant="primary")
                        reset_btn_s = gr.Button("Reset", variant="secondary")
                with gr.Row():
                    gr.Markdown(description_s)

        with gr.Row():
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[img_s],
                    outputs=[img_s],
                    examples_per_page=8,
                    fn=on_image_upload,
                    run_on_click=True
                )

    with gr.Tab("Point mode") as tab_p:
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_p.render()
            with gr.Column(scale=1):
                with gr.Row():
                    add_or_remove = gr.Radio(
                        ["Positive", "Negative"],
                        value="Positive",
                        label="Point Type"
                    )

                    with gr.Column():
                        clear_btn_p = gr.Button("Clear", variant="secondary")
                        reset_btn_p = gr.Button("Reset", variant="secondary")
                with gr.Row():
                    gr.Markdown(description_p)

        with gr.Row():
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[img_p],
                    outputs=[img_p],
                    examples_per_page=8,
                    fn=on_image_upload,
                    run_on_click=True
                )

    with gr.Tab("Box mode") as tab_b:
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                img_b.render()
            with gr.Row():
                with gr.Column():
                    clear_btn_b = gr.Button("Clear", variant="secondary")
                    reset_btn_b = gr.Button("Reset", variant="secondary")
                    gr.Markdown(description_b)

        with gr.Row():
            with gr.Column():
                gr.Markdown("Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[img_b],
                    outputs=[img_b],
                    examples_per_page=8,
                    fn=on_image_upload,
                    run_on_click=True
                )

    # with gr.Row():
    #     with gr.Column(scale=1):
    #         gr.Markdown(
    #             "<center><img src='https://visitor-badge.laobi.icu/badge?page_id=chongzhou/EfficientSAM' alt='visitors'></center>")
    img_s.upload(on_image_upload, img_s, [img_s])
    reset_btn_s.click(reset, outputs=[img_s])
    sam_btn_s.click(segment_anything,outputs=[img_s])
    tab_s.select(fn=reset_all, outputs=[img_s,img_p, img_b])

    img_p.upload(on_image_upload, img_p, [img_p])
    img_p.select(segment_with_points, [add_or_remove], img_p)

    clear_btn_p.click(clear, outputs=[img_p])
    reset_btn_p.click(reset, outputs=[img_p])
    tab_p.select(fn=reset_all, outputs=[img_s,img_p, img_b])

    img_b.upload(on_image_upload, img_b, [img_b])
    img_b.select(segment_with_box, outputs=[img_b])

    clear_btn_b.click(clear, outputs=[img_b])
    reset_btn_b.click(reset, outputs=[img_b])
    tab_b.select(fn=reset_all, outputs=[img_s,img_p, img_b])

demo.queue()
demo.launch(server_name=args.server_name, server_port=args.port)