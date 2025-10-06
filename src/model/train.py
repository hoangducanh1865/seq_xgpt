import os
import sys
import json
import torch
import numpy as np
import warnings
import torch.nn.functional as F
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

project_path = os.path.abspath('')
if project_path not in sys.path:
    sys.path.append(project_path)
from src.utils import backend_model_info
from src.model.dataloader import DataManager
from src.model.model import ModelWiseCNNClassifier, ModelWiseTransformerClassifier, TransformerOnlyClassifier



class SupervisedTrainer:
    def __init__(self, data, model, en_labels, id2label, args):
        self.data = data
        self.model = model
        self.en_labels = en_labels
        self.id2label =id2label

        self.seq_len = args.seq_len
        self.num_train_epochs = args.num_train_epochs
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.warm_up_ratio = args.warm_up_ratio
        self.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Only create optimizer and scheduler if we have training data
        if self.data.train_dataloader is not None:
            self._create_optimizer_and_scheduler()
        else:
            self.optimizer = None
            self.scheduler = None

    def _create_optimizer_and_scheduler(self):
        num_training_steps = len(
            self.data.train_dataloader) * self.num_train_epochs // self.gradient_accumulation_steps
        no_decay = ["bias", "LayerNorm.weight"]

        named_parameters = self.model.named_parameters()
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in named_parameters
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                self.weight_decay,
            },
            {
                "params": [
                    p for n, p in named_parameters
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warm_up_ratio * num_training_steps,
            num_training_steps=num_training_steps)
    
    def load_pretrained_model(self, checkpoint_path):
        """Load pretrained model from checkpoint"""
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained model from: {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                print("Pretrained model loaded successfully!")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                raise e
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    def train(self, ckpt_name='linear_en.pt', final_ckpt_path=None):
        for epoch in trange(int(self.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_steps = 0
            self.optimizer.zero_grad()  # Initialize gradients
            
            # train
            for step, inputs in enumerate(
                    tqdm(self.data.train_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                        
                with torch.set_grad_enabled(True):
                    labels = inputs['labels']
                    output = self.model(inputs['features'], inputs['labels'])
                    logits = output['logits']
                    loss = output['loss']
                    
                    # Scale loss by gradient accumulation steps
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    tr_loss += loss.item() * self.gradient_accumulation_steps  # Unscale for logging
                    nb_tr_steps += 1
                    
                    # Update weights every gradient_accumulation_steps
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    
                    # Clear cache more frequently and delete intermediate tensors
                    del output, logits, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Handle remaining gradients if steps don't divide evenly
            if nb_tr_steps % self.gradient_accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            loss = tr_loss / nb_tr_steps
            print(f'epoch {epoch+1}: train_loss {loss}')
            # test
            self.test()
            print('*' * 120)
            
            # Save intermediate checkpoints during training
            torch.save(self.model.state_dict(), ckpt_name)
            self.model.to(self.device)

        # Save final checkpoint to output_models directory
        if final_ckpt_path:
            os.makedirs(os.path.dirname(final_ckpt_path), exist_ok=True)
            torch.save(self.model.state_dict(), final_ckpt_path)
            print(f"Final model checkpoint saved to: {final_ckpt_path}")
        
        # Load the final checkpoint
        final_path = final_ckpt_path if final_ckpt_path else ckpt_name
        state_dict = torch.load(final_path)
        self.model.load_state_dict(state_dict)

    def test(self, content_level_eval=False):
        self.model.eval()
        texts = []
        true_labels = []
        pred_labels = []
        total_logits = []
        
        with torch.no_grad():  # Wrap entire test in no_grad for better memory efficiency
            for step, inputs in enumerate(
                    tqdm(self.data.test_dataloader, desc="Iteration")):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.device)
                
                labels = inputs['labels']
                output = self.model(inputs['features'], inputs['labels'])
                logits = output['logits']
                preds = output['preds']
                
                # Move to CPU immediately and extend lists
                texts.extend(inputs['text'])
                pred_labels.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
                total_logits.extend(logits.cpu().tolist())
                
                # Delete intermediate variables and clear cache
                del output, logits, preds, labels
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # with open("", 'w') as f:
        #     f.write(json.dumps(total_logits[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(texts[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(true_labels[3], ensure_ascii=False) + '\n')
        #     f.write(json.dumps(pred_labels[3], ensure_ascii=False) + '\n')


        if content_level_eval:
            # content level evaluation
            print("*" * 8, "Content Level Evalation", "*" * 8)
            content_result = self.content_level_eval(texts, true_labels, pred_labels)
        else:
            # sent level evalation
            print("*" * 8, "Sentence Level Evalation", "*" * 8)
            sent_result = self.sent_level_eval(texts, true_labels, pred_labels)

        # word level evalation
        print("*" * 8, "Word Level Evalation", "*" * 8)
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        true_labels_1d = true_labels.reshape(-1)
        pred_labels_1d = pred_labels.reshape(-1)
        mask = true_labels_1d != -1
        true_labels_1d = true_labels_1d[mask]
        pred_labels_1d = pred_labels_1d[mask]
        accuracy = (true_labels_1d == pred_labels_1d).astype(np.float32).mean().item()
        print("Accuracy: {:.1f}".format(accuracy*100))
        pass
    
    def content_level_eval(self, texts, true_labels, pred_labels):
        '''from collections import Counter'''

        true_content_labels = []
        pred_content_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_label = np.array(true_label)
            pred_label = np.array(pred_label)
            mask = true_label != -1
            true_label = true_label[mask].tolist()
            pred_label = pred_label[mask].tolist()
            true_common_tag = self._get_most_common_tag(true_label)
            true_content_labels.append(true_common_tag[0])
            pred_common_tag = self._get_most_common_tag(pred_label)
            pred_content_labels.append(pred_common_tag[0])
        
        true_content_labels = [self.en_labels[label] for label in true_content_labels]
        pred_content_labels = [self.en_labels[label] for label in pred_content_labels]
        result = self._get_precision_recall_acc_macrof1(true_content_labels, pred_content_labels, 
                                                       save_plots=True, title_suffix="Content Level")
        return result

    def sent_level_eval(self, texts, true_labels, pred_labels):
        """
        """
        true_sent_labels = []
        pred_sent_labels = []
        for text, true_label, pred_label in zip(texts, true_labels, pred_labels):
            true_sent_label = self.get_sent_label(text, true_label)
            pred_sent_label = self.get_sent_label(text, pred_label)
            true_sent_labels.extend(true_sent_label)
            pred_sent_labels.extend(pred_sent_label)
        
        true_sent_labels = [self.en_labels[label] for label in true_sent_labels]
        pred_sent_labels = [self.en_labels[label] for label in pred_sent_labels]
        result = self._get_precision_recall_acc_macrof1(true_sent_labels, pred_sent_labels, 
                                                       save_plots=True, title_suffix="Sentence Level")
        return result

    def get_sent_label(self, text, label):
        import nltk
        sent_separator = nltk.data.load('tokenizers/punkt/english.pickle')
        sents = sent_separator.tokenize(text)

        offset = 0
        sent_label = []
        for sent in sents:
            start = text[offset:].find(sent) + offset
            end = start + len(sent)
            offset = end
            
            split_sentence = self.data.split_sentence
            end_word_idx = len(split_sentence(text[:end]))
            if end_word_idx > self.seq_len:
                break
            word_num = len(split_sentence(text[start:end]))
            start_word_idx = end_word_idx - word_num
            tags = label[start_word_idx:end_word_idx]
            most_common_tag = self._get_most_common_tag(tags)
            sent_label.append(most_common_tag[0])
        
        if len(sent_label) == 0:
            print("empty sent label list")
        return sent_label
    
    def _get_most_common_tag(self, tags):
        """most_common_tag is a tuple: (tag, times)"""
        from collections import Counter

        '''tags = [self.id2label[tag] for tag in tags]'''
        tags = [self.id2label[tag] for tag in tags if tag != -1]
        tags = ['-'.join(tag.split('-')[1:]) for tag in tags]
        tag_counts = Counter(tags)
        most_common_tag = tag_counts.most_common(1)[0]

        return most_common_tag

    def _get_precision_recall_acc_macrof1(self, true_labels, pred_labels, save_plots=False, title_suffix=""):
        # Get unique labels for proper indexing
        unique_labels = sorted(list(set(true_labels + pred_labels)))
        label_names = [list(self.en_labels.keys())[list(self.en_labels.values()).index(label)] for label in unique_labels]
        
        accuracy = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average='macro')
        weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')
        
        # Calculate MSE and MAE
        mse = mean_squared_error(true_labels, pred_labels)
        mae = mean_absolute_error(true_labels, pred_labels)
        
        print("\n" + "="*20 + f" {title_suffix} Results Summary" + "="*20)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Macro: {macro_f1:.4f}")
        print(f"F1-Weighted: {weighted_f1:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Detailed classification report
        print("\n📊 Classification Report (validation):")
        report = classification_report(true_labels, pred_labels, target_names=label_names, digits=4)
        print(report)
        
        if save_plots:
            # Confusion Matrix
            cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                       xticklabels=label_names, yticklabels=label_names)
            plt.title(f'Confusion Matrix - {title_suffix}')
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            plt.tight_layout()
            
            # Save plot
            plot_path = f'confusion_matrix_{title_suffix.lower().replace(" ", "_")}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Confusion matrix saved as: {plot_path}")
        
        precision = precision_score(true_labels, pred_labels, average=None, labels=unique_labels)
        recall = recall_score(true_labels, pred_labels, average=None, labels=unique_labels)
        precision_macro = precision_score(true_labels, pred_labels, average='macro')
        recall_macro = recall_score(true_labels, pred_labels, average='macro')
        
        result = {
            "precision": precision, "recall": recall, "accuracy": accuracy, 
            "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "precision_macro": precision_macro, "recall_macro": recall_macro,
            "mse": mse, "mae": mae
        }
        return result


def construct_bmes_labels(labels):
    prefix = ['B-', 'M-', 'E-', 'S-']
    id2label = {}
    counter = 0

    for label, id in labels.items():
        for pre in prefix:
            id2label[counter] = pre + label
            counter += 1
    
    return id2label

def split_dataset(data_path, train_path, test_path, train_ratio=0.9):
    file_names = [file_name for file_name in os.listdir(data_path) if file_name.endswith('.jsonl')]
    print('*'*32)
    print('The overall data sources:')
    print(file_names)
    file_paths = [os.path.join(data_path, file_name) for file_name in file_names]

    total_samples = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            samples = [json.loads(line) for line in f]
            total_samples.extend(samples)
    
    import random
    random.seed(0)
    random.shuffle(total_samples)

    split_index = int(len(total_samples) * train_ratio)
    train_data = total_samples[:split_index]
    test_data = total_samples[split_index:]

    def save_dataset(fpath, data_samples):
        with open(fpath, 'w', encoding='utf-8') as f:
            for sample in tqdm(data_samples):
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    save_dataset(train_path, train_data)
    save_dataset(test_path, test_data)
    print()
    print("The number of train dataset:", len(train_data))
    print("The number of test  dataset:", len(test_data))
    print('*'*32)
    pass

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', type=str) # ['three_labels', 'specific_labels']
    parser.add_argument('--dataset', type=str) # ['faid', 'hart', 'llm_detective']
    parser.add_argument('--model', type=str, default='Transformer')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train_mode', type=str, default='classify')
    parser.add_argument('--batch_size', type=int, default=8)  # Reduced from 32 to prevent OOM
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)  # Effective batch size = batch_size * gradient_accumulation_steps

    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--split_dataset', action='store_true')
    parser.add_argument('--data_path', type=str, default='')

    parser.add_argument('--num_train_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--warm_up_ratio', type=float, default=0.1)

    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_content', action='store_true')
    parser.add_argument('--load_pretrained', action='store_true')
    return parser.parse_args()

# python ./Seq_train/train.py --gpu=0 --split_dataset
# python ./Seq_train/train.py --gpu=0
if __name__ == "__main__":
    args = parse_args()
    
    # Select appropriate data files based on mode
    if args.load_pretrained:
        # For test-only mode, use the basic data files
        train_path = 'data/datasets/' + args.dataset + '/train_data.jsonl'  # Won't be used
        test_path = 'data/datasets/' + args.dataset + '/test_data.jsonl'
    else:
        # For training mode, use feature files if they exist, otherwise basic files
        train_features_path = 'data/datasets/' + args.dataset + '/train_data_with_features.jsonl'
        test_features_path = 'data/datasets/' + args.dataset + '/test_data_with_features.jsonl'
        
        if os.path.exists(train_features_path) and os.path.exists(test_features_path):
            train_path = train_features_path
            test_path = test_features_path
        else:
            train_path = 'data/datasets/' + args.dataset + '/train_data.jsonl'
            test_path = 'data/datasets/' + args.dataset + '/test_data.jsonl'
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # if args.split_dataset:
    #     print("Log INFO: split dataset...")
    #     split_dataset(data_path=args.data_path, train_path=train_path, test_path=test_path, train_ratio=args.train_ratio)
    #     sys.exit(1)
    
    if args.setup == 'three_labels':
        en_labels = {
            'human-text': 0,
            'deepseek-text': 1,
            'gemini-text': 1,
            'gpt-text': 1,
            'llama-text': 1,
            'human---deepseek-text': 2,
            'human---gemini-text': 2,
            'human---gpt-text': 2,
            'human---llama-text': 2
        }
    
    elif args.setup == 'specific_labels':
        if args.dataset == 'faid':
            en_labels = {
                'human-text': 0,
                'deepseek-text': 1,
                'gemini-text': 2,
                'gpt-text': 3,
                'llama-text': 4,
                'human---deepseek-text': 5,
                'human---gemini-text': 6,
                'human---gpt-text': 7,
                'human---llama-text': 8,
            }
        
        elif args.dataset == 'hart':
            en_labels = {
                'human-text': 0,
                'claude-text': 1,
                'gemini-text': 2,
                'gpt-text': 3,
                'human---claude-text': 4,
                'human---gemini-text': 5,
                'human---gpt-text': 6,
            }
        
        elif args.dataset == 'llm_detective':
            en_labels = {
                'human-text': 0,
                'gemma-text': 1,
                'llama3-text': 2,
                'mixtral-text': 3,
                'human---gemma-text': 4,
                'human---llama3-text': 5,
                'human---mixtral-text': 6,
            }

    id2label = construct_bmes_labels(en_labels)
    label2id = {v: k for k, v in id2label.items()}

    # Create DataManager - for test-only mode, we can pass None for train_path
    if args.load_pretrained:
        data = DataManager(train_path=None, test_path=test_path, batch_size=args.batch_size, max_len=args.seq_len, human_label='human-text', id2label=id2label, raw_text_mode=True)
    else:
        data = DataManager(train_path=train_path, test_path=test_path, batch_size=args.batch_size, max_len=args.seq_len, human_label='human-text', id2label=id2label)
    
    """linear classify"""
    if args.train_mode == 'classify':
        print('-' * 32 + 'classify' + '-' * 32)
        if args.model == 'CNN':
            print('-' * 32 + "CNN" + '-' * 32)
            classifier = ModelWiseCNNClassifier(id2labels=id2label)
            ckpt_name = 'cnn_checkpoint.pt'
            final_ckpt_name = f'cnn_{args.dataset}_{args.setup}_final.pt'
        elif args.model == 'RNN':
            print('-' * 32 + "RNN" + '-' * 32)
            classifier = TransformerOnlyClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = 'rnn_checkpoint.pt'
            final_ckpt_name = f'rnn_{args.dataset}_{args.setup}_final.pt'
        else:
            classifier = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=args.seq_len)
            ckpt_name = 'transformer_checkpoint.pt'
            final_ckpt_name = f'transformer_{args.dataset}_{args.setup}_final.pt'

        # Define paths for checkpoints
        final_ckpt_path = os.path.join('data', 'output_models', final_ckpt_name)
        
        trainer = SupervisedTrainer(data, classifier, en_labels, id2label, args)

        if args.do_test:    
            # Test-only mode: Load pretrained model for testing
            print("🔍 Testing pretrained model...")
            if args.load_pretrained and os.path.exists(final_ckpt_path):
                trainer.load_pretrained_model(final_ckpt_path)
                print(f"✅ Loaded pretrained model from: {final_ckpt_path}")
            elif os.path.exists(ckpt_name):
                state_dict = torch.load(ckpt_name)
                trainer.model.load_state_dict(state_dict)
                print(f"✅ Loaded checkpoint from: {ckpt_name}")
            else:
                print("❌ No model checkpoint found!")
                print(f"Looking for: {final_ckpt_path} or {ckpt_name}")
                sys.exit(1)
            
            trainer.test(content_level_eval=args.test_content)
        else:
            # Training mode: Train and then test
            print("🚀 Starting training...")
            # Load pretrained model if specified before training
            if args.load_pretrained and os.path.exists(final_ckpt_path):
                try:
                    trainer.load_pretrained_model(final_ckpt_path)
                    print(f"✅ Loaded pretrained model for fine-tuning: {final_ckpt_path}")
                except FileNotFoundError:
                    print(f"⚠️ Warning: Pretrained model not found at {final_ckpt_path}. Starting from scratch.")
            
            trainer.train(ckpt_name=ckpt_name, final_ckpt_path=final_ckpt_path)
            print("✅ Training completed!")
            
            print("🔍 Testing trained model...")
            trainer.test(content_level_eval=args.test_content)

    """contrastive training"""
    if args.train_mode == 'contrastive_learning':
        print('-' * 32 + 'contrastive_learning' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''

        trainer = SupervisedTrainer(data, classifier, loss_criterion = 'ContrastiveLoss')
        trainer.train(ckpt_name=ckpt_name)

    """classify after contrastive"""
    if args.train_mode == 'contrastive_classify':
        print('-' * 32 + 'contrastive_classify' + '-' * 32)
        if args.model == 'CNN':
            classifier = ModelWiseCNNClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
            saved_model = torch.load(ckpt_name)
            classifier.load_state_dict(saved_model.state_dict())
            ckpt_name = ''
        else:
            classifier = ModelWiseTransformerClassifier(class_num=backend_model_info.en_class_num)
            ckpt_name = ''
            saved_model = torch.load(ckpt_name)
            classifier.load_state_dict(saved_model.state_dict())
            ckpt_name = ''

        # trainer = SupervisedTrainer(data, classifier, train_mode='Contrastive_Classifier')
        trainer = SupervisedTrainer(data, classifier)
        trainer.train(ckpt_name=ckpt_name)
