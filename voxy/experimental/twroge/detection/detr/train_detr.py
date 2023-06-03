from experimental.twroge.detection.detr.detr import Detr
from experimental.twroge.detection.detr.coco_dataset import CocoDetection
from transformers import DetrFeatureExtractor
from torch.utils.data import DataLoader
import torch
import logging
import wandb
import json
import os
from pytorch_lightning import Trainer
from core.infra.cloud.gcs_utils import upload_to_gcs
import argparse

logging.getLogger().setLevel(logging.INFO)

def get_dataloaders(train_dataset, validation_dataset, feature_extractor):

    def collate_fn(batch):
      pixel_values = [item[0] for item in batch]
      encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
      labels = [item[1] for item in batch]
      batch = {}
      batch['pixel_values'] = encoding['pixel_values']
      batch['pixel_mask'] = encoding['pixel_mask']
      batch['labels'] = labels
      return batch

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, collate_fn=collate_fn, batch_size=1, num_workers=0)
    batch = next(iter(train_dataloader))
    return train_dataloader, validation_dataloader

def get_dataset(root_directory, annotations_file, feature_extractor):
    entire_dataset = CocoDetection(root_directory=root_directory, annotation_file=annotations_file, feature_extractor=feature_extractor)
    # for now the val set is the training set

    train_set_size = int(len(entire_dataset) * 0.8)
    valid_set_size = len(entire_dataset) - train_set_size

    train_dataset, validation_dataset =  torch.utils.data.random_split(entire_dataset, [train_set_size, valid_set_size])
    return train_dataset, validation_dataset, entire_dataset

def train_model(model, training, validation, args):
    trainer = Trainer(gpus=1, min_epochs=args.min_epochs, max_epochs=args.max_epochs, gradient_clip_val=args.grad_clip)
    trainer.fit(model, training, validation)

def save_model(model, name, local_path):
    LOCAL_PATH = os.path.join(local_path, name)
    #TODO add saving from torch jit
    torch.save(model.state_dict(), LOCAL_PATH)
    # upload file to GCS
    CLOUD_PATH = "gs://voxel-users/shared/detr/models/" + name
    logging.info(f"Saving model: {CLOUD_PATH} ")
    upload_to_gcs(CLOUD_PATH, LOCAL_PATH)


def setup_wandb(args):
    wandb.init(project='detr_032022', entity='voxel-wandb', name=f"eval-{args.identifier}", sync_tensorboard=True)


def main(args):
    DATASET_PATH = args.data
    ANNOTATIONS_PATH = args.annotations
    OUTPUT_PATH = args.output
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    logging.info(" [ Running DETR training ] ")
    logging.info("Constructing dataset from: ")
    logging.info(f"\t root: {DATASET_PATH}")
    logging.info(f"\t annotations: {ANNOTATIONS_PATH}")
    logging.info("Configuration: ")
    logging.info("\n" + json.dumps(args.__dict__ , indent=2))
    train_dataset, validation_dataset, full_dataset = get_dataset(DATASET_PATH, ANNOTATIONS_PATH, feature_extractor)
    train_dataloader, validation_dataloader = get_dataloaders(train_dataset, validation_dataset, feature_extractor)
    n_labels = len([i for i in full_dataset.coco.cats  if i >= 0]) + args.extra_neurons

    # just train detr using pytorch lightning
    model = Detr(lr=args.lr, lr_backbone=args.lr_backbone, weight_decay=args.weight_decay, n_labels=n_labels, optimizer=args.optimizer)

    logging.info("Training model" )
    train_model(model, train_dataloader, validation_dataloader, args)

    model_name = f"detr_trained_lr_{model.lr}_lr_backbone_{model.lr_backbone}_weight_decay_{model.weight_decay}_classes_{n_labels}_{args.optimizer}_id_{args.identifier}.pth"
    logging.info(f"Saving model: {model_name} ")
    save_model(model, model_name, OUTPUT_PATH)

def get_args():

    parser = argparse.ArgumentParser(description='Train DETR')
    parser.add_argument('--data',  required=True,
			help='The root directory for the image. These show up in the coco labels')
    parser.add_argument('--annotations', type=str,
			required=True,
			help='The annotations file in coco format')
    parser.add_argument('--output', type=str,
			required=True,
			help='The output directory to put the model')
    parser.add_argument('--lr', type=float, default=9e-5,
			help='Model learning rate')
    parser.add_argument('--lr-backbone', type=float, default=1e-5,
			help='CNN backbone learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
			help='model weight decay')
    parser.add_argument('--max-epochs', type=float, default=10,
			help='Max number of epochs to train')
    parser.add_argument('--min-epochs', type=float, default=7,
			help='Min epochs to train')
    parser.add_argument('--grad-clip', type=float, default=None,
			help='Gradient clip')
    parser.add_argument('--extra-neurons', type=int, default=0,
			help='Extra class labels')
    parser.add_argument('--identifier', type=str, default="",
			help='Extra string to add to class model name')
    parser.add_argument('--optimizer', type=str, default="AdamW" , choices = ["AdamW", "SGD"],
			help='Extra string to add to class model name')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    setup_wandb(args)
    main(args)
