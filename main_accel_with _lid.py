import argparse
import logging
import math
import os
import random

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator

from transformers import XLMWithLMHeadModel, AdamW, get_linear_schedule_with_warmup

import config, utils, dataset, new_dataset, XLM_MLM_LID

logger = logging.getLogger(__name__)


def main():
    # Initialize the accelerator
    accelerator = Accelerator()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(accelerator.state)
    
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        
    tokenizer = utils.train_tokenizer(tokenizer_train=False)
    
    train_dataset = new_dataset.LIDdataset(config.TRAIN_DATA_PATH_TXT, config.TRAIN_DATA_PATH_LID, tokenizer)
    
    eval_dataset = new_dataset.LIDdataset(config.VAL_DATA_PATH_TXT, config.VAL_DATA_PATH_LID, tokenizer)
    
    accelerator.print("Dataset prepared")
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.per_device_train_batch_size)

    eval_dataloader = DataLoader(eval_dataset, batch_size=config.per_device_eval_batch_size)
    
    train_dataloader, eval_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader
    )
    
    ## Without any involvement of LIDs in the loss
    model = XLMWithLMHeadModel(config=config.BASE_MODEL_CONFIG)
    
    ## Adding a new loss based on LID labels (to use more signal for training code-mix LLM)
    model = model_LID.XLM_model_lid(xlm_config=config.BASE_MODEL_CONFIG, pred_layer_config=config.LID_PRED_LAYER_CONFIG)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    
    
    ## Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(
        model, optimizer
    )
    
    accelerator.print("Modules prepared")
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    
    max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    total_batch_size = config.per_device_train_batch_size * accelerator.num_processes * config.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    
    for epoch in range(config.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            
            outputs = model(batch)
            
#             lang_labels = batch.pop("lang_labels", None)
#             outputs = model(batch, lang_labels)
            
            loss = outputs['total_loss']
            loss = loss / config.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        model.eval()
        losses = []
        xlm_losses = []
        for step, batch in enumerate(eval_dataloader):
            lang_labels = batch.pop("lang_labels", None)

            with torch.no_grad():
                outputs = model(batch, lang_labels)

            loss, xlm_loss = outputs['total_loss'], outputs['xlm_loss']
            losses.append(accelerator.gather(loss.repeat(config.per_device_eval_batch_size)))
            xlm_losses.append(accelerator.gather(xlm_loss.repeat(config.per_device_eval_batch_size)))

            
        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        mean_loss = torch.mean(losses)
        
        xlm_losses = torch.cat(xlm_losses)
        xlm_losses = losses[: len(eval_dataset)]
        perplexity = math.exp(torch.mean(xlm_losses))

        logger.info(f"epoch {epoch+1}: perplexity: {perplexity}, mean loss: {mean_loss.item()}")

        if config.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save({
                              'epoch': epoch,
                              'model_state_dict': unwrapped_model.state_dict(),
                              }, config.output_file)
#             unwrapped_model.save_pretrained(config.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
