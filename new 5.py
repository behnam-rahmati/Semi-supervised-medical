import os
import argparse
from PIL import Image
import torch
import cv2
import numpy as np
from functools import partial
from transformers import (AutoTokenizer, CLIPTextModelWithProjection)
from transformers import (AutoProcessor, CLIPVisionModelWithProjection)
import supervision as sv
from mmengine.runner import Runner
from mmengine.dataset import Compose
from mmengine.runner.amp import autocast
from mmengine.config import Config, DictAction
from mmdet.datasets import CocoDataset
from torchvision.ops import nms

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
MASK_ANNOTATOR = sv.MaskAnnotator()

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-image', help='output image file path', default='./output_image.png')
    args = parser.parse_args()
    return args

def generate_image_embeddings(prompt_image, vision_encoder, vision_processor, projector, device='cuda:0'):
    prompt_image = prompt_image.convert('RGB')
    inputs = vision_processor(images=[prompt_image], return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    image_outputs = vision_encoder(**inputs)
    img_feats = image_outputs.image_embeds.view(1, -1)
    img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
    if projector is not None:
        img_feats = projector(img_feats)
    return img_feats

def run_image(runner, vision_encoder, vision_processor, padding_token, image, text, prompt_image, add_padding, max_num_boxes, score_thr, nms_thr, output_image_path):
    image = image.convert('RGB')
    if prompt_image is not None:
        texts = [['object'], [' ']]
        projector = None
        if hasattr(runner.model, 'image_prompt_encoder'):
            projector = runner.model.image_prompt_encoder.projector
        prompt_embeddings = generate_image_embeddings(prompt_image, vision_encoder, vision_processor, projector)
        if add_padding == 'padding':
            prompt_embeddings = torch.cat([prompt_embeddings, padding_token], dim=0)
        prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(p=2, dim=-1, keepdim=True)
        runner.model.num_test_classes = prompt_embeddings.shape[0]
        runner.model.setembeddings(prompt_embeddings[None])
    else:
        runner.model.setembeddings(None)
        texts = [[t.strip()] for t in text.split(',')]
    data_info = dict(img_id=0, img=np.array(image), texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0), data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        if (prompt_image is not None) and ('texts' in data_batch['data_samples'][0]):
            del data_batch['data_samples'][0]['texts']
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]

    pred_instances = pred_instances.cpu().numpy()
    if 'masks' in pred_instances:
        masks = pred_instances['masks']
    else:
        masks = None
    detections = sv.Detections(xyxy=pred_instances['bboxes'],
                               class_id=pred_instances['labels'],
                               confidence=pred_instances['scores'],
                               mask=masks)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    if masks is not None:
        image = MASK_ANNOTATOR.annotate(image, detections)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = Image.fromarray(image)
    image.save(output_image_path)  # Save the output image
    print(f"Output saved to {output_image_path}")

def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline[0].type = 'mmdet.LoadImageFromNDArray'
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    # Initialize vision encoder
    clip_model = "/path/to/pretrained/open-ai-clip-vit-base-patch32"
    vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model)
    processor = AutoProcessor.from_pretrained(clip_model)
    device = 'cuda:0'
    vision_model.to(device)

    # Initialize text encoder
    tokenizer = AutoTokenizer.from_pretrained(clip_model)
    text_model = CLIPTextModelWithProjection.from_pretrained(clip_model)
    text_model.to(device)

    # Load input image (can be provided via argument)
    image_path = "./input_image.png"
    image = Image.open(image_path)

    # Set other parameters
    prompt_image = None  # Set your image prompt here if needed
    padding_token = torch.zeros(1, 512).to(device)  # Example padding token
    input_text = "cat, dog"  # Example text prompt
    add_padding = "none"
    max_num_boxes = 100
    score_thr = 0.5
    nms_thr = 0.7

    output_image_path = args.output_image  # Set output image path from args

    run_image(runner, vision_model, processor, padding_token, image, input_text, prompt_image, add_padding, max_num_boxes, score_thr, nms_thr, output_image_path)

if __name__ == '__main__':
    main()
