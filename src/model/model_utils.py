import torch
from transformers import AutoImageProcessor, AutoTokenizer

from utils.constants import IMAGE_TOKEN, PAD_TOKEN, LORA_CONFIG
from src.model.configuration import MultimodalConfig
from src.model.mutlimodal_model import MultimodalModelForConditionalGeneration
from src.processing import MultiModalProcessor


def build_model(text_model_id,
                vision_model_id,
                freeze_multimodal_projector=False,
                freeze_language_model=False,
                freeze_vision_model=False,
                device="cuda",
                use_bfloat16=True,
                load_in_4bit=False):
    """
    Build model and related components.
    """
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=False)

    tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})

    if tokenizer.pad_token is None or tokenizer.pad_token != PAD_TOKEN:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        tokenizer.pad_token = PAD_TOKEN

    tokenizer_len = len(tokenizer)

    multimodal_config = MultimodalConfig(vision_model_id=vision_model_id,
                                         text_model_id=text_model_id,
                                         tokenizer_len=tokenizer_len,
                                         lora_config=LORA_CONFIG,
                                         freeze_multimodal_projector=freeze_multimodal_projector,
                                         freeze_language_model=freeze_language_model,
                                         freeze_vision_model=freeze_vision_model,
                                         load_in_4bit=load_in_4bit)

    # Image processor
    image_processor = AutoImageProcessor.from_pretrained(vision_model_id)
    processor = MultiModalProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Language model
    multimodal_model = MultimodalModelForConditionalGeneration(multimodal_config).to(device)

    if use_bfloat16:
        multimodal_model = multimodal_model.to(torch.bfloat16)

    return dict(tokenizer=tokenizer,
                model=multimodal_model,
                processor=processor,
                config=multimodal_config)