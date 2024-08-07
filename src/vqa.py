"""
Main entrypoint for Visual-Question Answering with a pretrained model
Example run: python src/vqa.py --model_path="./model_checkpoints/04-23_18-53-28/checkpoint-1000" --image_path="./data/images/000000581899.jpg" --user_question="What is in the image?"
"""

import argparse
import sys
import logging

import torch
from PIL import Image

from src.model.mutlimodal_model import MultimodalModelForConditionalGeneration
from src.processing import MultiModalProcessor
from utils.utils import get_available_device


logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        required=True,
                        help='Path to the pretrained model weights')

    parser.add_argument('--image_path',
                        required=True,
                        help='Path to the prompt image file')

    parser.add_argument('--user_question',
                        required=True,
                        help='The question to ask about the image to the model. Example: "What is in the image?"')

    args = parser.parse_args(sys.argv[1:])
    logging.info(f"Parameters received: {args}")

    logging.info("Loading pretrained model...")
    device = get_available_device()
    multimodal_model = MultimodalModelForConditionalGeneration.from_pretrained(args.model_path,
                                                                                     device_map="cpu",
                                                                                     torch_dtype=torch.bfloat16).eval()

    processor = MultiModalProcessor.from_pretrained(args.model_path)

    logging.info("Running model for VQA...")

    prompt = (f"<|im_start|>user <image>\n{args.user_question} <|im_end|>\n"
              f"<|im_start|>assistant\n")

    raw_image = Image.open(args.image_path)
    inputs = processor(prompt, raw_image, return_tensors='pt').to(device, torch_dtype=torch.bfloat16)

    output = multimodal_model.generate(**inputs,
                                             max_new_tokens=200,
                                             do_sample=False)
    logging.info(f"Model answer: {processor.decode(output[0][2:], skip_special_tokens=True)}")
