# Modified from https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/trainer.py

from typing import Dict, List, Optional, Union

import torch
import os
import logging
from transformers import Trainer
from torch.utils.data import Dataset
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Any
from trainer.base import FinetuneTrainer
from model import get_model

logger = logging.getLogger(__name__)

def forward_kl_loss_fn(
    teacher_logits,
    student_logits,
    input_ids,
    pad_token_id,
    temperature=1.0,
):
    """
    Forward KL: KL(teacher || student).
    Implementation detail: shift teacher/student by 1 for next-token prediction.
    """
    teacher_shift = teacher_logits[..., :-1, :].contiguous()
    student_shift = student_logits[..., :-1, :].contiguous()
    labels_shift = input_ids[..., 1:].contiguous()


    teacher_shift = teacher_shift.view(-1, teacher_shift.size(-1))
    student_shift = student_shift.view(-1, student_shift.size(-1))
    labels_shift = labels_shift.view(-1)

    mask = (labels_shift != pad_token_id) & (labels_shift != -100)

    teacher_scaled = teacher_shift[mask] / temperature
    student_scaled = student_shift[mask] / temperature

    teacher_probs = torch.softmax(teacher_scaled, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_scaled, dim=-1)
    student_log_probs = torch.log_softmax(student_scaled, dim=-1)

    kl_vals = teacher_probs * (teacher_log_probs - student_log_probs)
    # forward KL = sum over vocab: p_teacher * [log p_teacher - log p_student]
    return torch.sum(kl_vals) / max(mask.sum(), 1)


class DistillTrainer(FinetuneTrainer):
    def __init__(self, teacher_model, temperature, evaluators=None, template_args=None, *args, **kwargs):
        # teacher model, temperature
        # TODO: Load teacher model, set tokenizer
        model, tokenizer = get_model(teacher_model)
        self.teacher_model = model.cuda()
        self.tokenizer = tokenizer
        self.temperature = temperature
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        teacher_outputs = self.teacher_model(**inputs)

        loss = forward_kl_loss_fn(
            teacher_logits=teacher_outputs.logits,
            student_logits=outputs.logits,
            input_ids=inputs['input_ids'],
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=self.temperature,
        )
        
        return (loss, outputs) if return_outputs else loss
