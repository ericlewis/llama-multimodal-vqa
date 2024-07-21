import copy
import json
import os
from dataclasses import dataclass
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from utils.constants import IGNORE_INDEX
from dataset.data_utils import preprocess_multimodal


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 image_folder: str,
                 image_processor,
                 image_aspect_ratio,
                 tokenizer,
                 preprocess_func,
                 is_multimodal: bool = True):
        super(SupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.preprocess_func = preprocess_func
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.is_multimodal = is_multimodal
        self.data_path = data_path
        self.image_folder = image_folder
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know how to handle this!"

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.image_folder
            processor = self.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.image_aspect_ratio == 'keep':
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            elif self.image_aspect_ratio == 'pad':
                # Use the processor directly instead of process_images_clip
                image = processor(images=image, return_tensors='pt')['pixel_values'][0]
            else:
                raise ValueError(f"Invalid image_aspect_ratio: {self.image_aspect_ratio}")
        else:
            image = None

        data_dict = self.preprocess_func(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        
        if data_dict is None:
            # Handle the case where preprocessing failed
            print(f"Preprocessing failed for item {i}. Skipping this item.")
            # You might want to return a default item or skip to the next one
            # For now, let's return a dummy item
            return {"input_ids": torch.tensor([0]), "labels": torch.tensor([0])}

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict['image'] = image

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        
        max_length = min(input_ids.size(1), self.tokenizer.model_max_length)
        input_ids = input_ids[:, :max_length]
        labels = labels[:, :max_length]
        
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch