
import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

sys.path.append('path to src folder')

from models.main_painAttnNet import PainAttnNet
from parser import ConfigParser
from utils.utils import generate_kfolds_index, load_data
from trainers.main_trainers import Trainer
from ssd_utils import calculate_teacher_student_loss_batch

SEED = 5012023
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)

def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def load_teacher_model(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

def at_loss(f_s, f_t):
    return ((f_s - f_t) ** 2).mean()

def train_knowledge_distillation(config,reps, fold_id, data_dir, teacher_model_path, results_dir, device, T, at_loss_weight, ce_loss_weight):
    batch_size = config["data_loader"]["args"]["batch_size"]

    logger = config.get_logger('trainers')

    teacher = PainAttnNet().to(device)
    logger.info(f"Loading teacher model from: {teacher_model_path}")

    if not os.path.exists(teacher_model_path):
        logger.warning(f"Teacher model not found at: {teacher_model_path}. Skipping fold {fold_id}.")
        return

    load_teacher_model(teacher, teacher_model_path)
    teacher.eval()

    def set_dropout_train(model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                # Check if the module belongs to the fusion model
                if "mscn" not in name and "encoderWrapper" not in name:  
                    module.train(True)  # Set only fusion model dropout layers to train
                    print("Setting dropout in teacher fusion model to train")
                else:
                    module.train(False)
    
    # def set_dropout_train(model):
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Dropout):
    #             module.train(True)  # Set only fusion model dropout layers to train
    #             print(f'Setting dropout in teacher module {name} fusion model to train')


    
    set_dropout_train(teacher) # Set dropout layers in teacher model to train

    student = PainAttnNet().to(device)
    student.apply(weights_init_normal)
    logger.info(student)

    label_converter = config['label_converter']

    trainable_params = filter(lambda p: p.requires_grad, student.parameters())
    optimizer = torch.optim.Adam(trainable_params)

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f'Data directory does not exist: {data_dir}')
    folds_data = generate_kfolds_index(data_dir, config["data_loader"]["args"]["num_folds"])

    data_loader, valid_data_loader, data_count = load_data(folds_data[fold_id][0], folds_data[fold_id][1], label_converter, batch_size)

    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    student.train()
    log_file_path = os.path.join(results_dir, f"fold_{fold_id}_training.log")
    with open(log_file_path, 'w') as log_file:
        for epoch in range(config["trainer"]["epochs"]):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                teacher_reps = []

                optimizer.zero_grad()

                with torch.no_grad():
                    for rep in range(reps):
                        teacher_features, teacher_logits = teacher.extract_features(inputs)

                        teacher_reps.append(teacher_features.view(teacher_features.size(0), -1))
                    

                student_features, student_logits = student.extract_features(inputs)


                student_features = student_features.view(student_features.size(0), -1)
                
                
                loss_at = calculate_teacher_student_loss_batch(student_features, teacher_reps,10) # KD loss between teacher and student features
                
                
                soft_targets = torch.nn.functional.softmax(teacher_logits / T, dim=-1)
                soft_prob = torch.nn.functional.log_softmax(student_logits / T, dim=-1)


                loss_kl = kl_loss(soft_prob, soft_targets) * (T ** 2) # KL divergence loss between teacher and student logits (optional)
                
                loss_ce = ce_loss(student_logits, labels) # Cross-entropy loss between student logits and ground truth labels
                
                # loss = at_loss_weight * loss_at + ce_loss_weight * loss_ce + (1 - ce_loss_weight) * loss_kl # Total loss w/ Logit based KD
                
                loss = at_loss_weight * loss_at + ce_loss_weight * loss_ce # Total loss w/o logit based KD

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(student_logits, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

            accuracy = correct_predictions / total_predictions
            avg_loss = running_loss / len(data_loader)
            log_file.write(f"Epoch {epoch+1}/{config['trainer']['epochs']}, Training Loss: {avg_loss}, Training Accuracy: {accuracy}\n")
            logger.info(f"Epoch {epoch+1}/{config['trainer']['epochs']}, Training Loss: {avg_loss}, Training Accuracy: {accuracy}")

            if valid_data_loader:
                student.eval()
                valid_loss = 0.0
                correct_valid_predictions = 0
                total_valid_predictions = 0
                with torch.no_grad():
                    for inputs, labels in valid_data_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        student_features, student_logits = student.extract_features(inputs)
                        loss_ce = ce_loss(student_logits, labels)
                        valid_loss += loss_ce.item()
                        _, predicted = torch.max(student_logits, 1)
                        correct_valid_predictions += (predicted == labels).sum().item()
                        total_valid_predictions += labels.size(0)
                valid_accuracy = correct_valid_predictions / total_valid_predictions
                avg_valid_loss = valid_loss / len(valid_data_loader)
                log_file.write(f"Epoch {epoch+1}/{config['trainer']['epochs']}, Validation Loss: {avg_valid_loss}, Validation Accuracy: {valid_accuracy}\n")
                logger.info(f"Epoch {epoch+1}/{config['trainer']['epochs']}, Validation Loss: {avg_valid_loss}, Validation Accuracy: {valid_accuracy}")

    trainer = Trainer(student, ce_loss, optimizer, config=config, data_loader=data_loader, fold_id=fold_id, valid_data_loader=valid_data_loader)
    validation_results = trainer.train()

    results_path = os.path.join(results_dir, f"results_fold_{fold_id}.npy")
    np.save(results_path, validation_results)
    return validation_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for distillation training arguments')
    parser.add_argument('-c', '--config', default='path to config.json', type=str, help='config file path (default: config.json)')
    parser.add_argument('-d', '--device', default='0', type=str, help='GPUs')
    parser.add_argument('-f', '--fold_id', default=0, type=int, help='fold_id')
    parser.add_argument('--teacher_model_path', type=str, required=False, help='Path to the pretrained teacher model')
    parser.add_argument('--results_dir', type=str, required=False, help='Directory to save the results')
    parser.add_argument('--at_loss_weight', type=float, default=0.5, help='Weight for the attention transfer loss')
    parser.add_argument('--ce_loss_weight', type=float, default=0.5, help='Weight for the cross-entropy loss')
    parser.add_argument('--T', type=float, default=2.0, help='Temperature for knowledge distillation')

    args = parser.parse_args()
    config = ConfigParser.from_args(parser, args.fold_id)
    data_dir = config["data_loader"]["args"]["data_dir"]
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    results_fold_dir = os.path.join(args.results_dir, f'fold_{args.fold_id}')
    os.makedirs(results_fold_dir, exist_ok=True)
    
    train_knowledge_distillation(config,30, args.fold_id, data_dir, args.teacher_model_path, args.results_dir, device,T=args.T, at_loss_weight=args.at_loss_weight, ce_loss_weight=args.ce_loss_weight)
