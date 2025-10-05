import json
import os 
import transformers
from transformers import AutoTokenizer
import re
import gc


class TrainTestDataPreparer:
    def __init__(self):
        self.datasets_dir = 'data/datasets/'
        self.tokenizers = [
            # DeepSeek [0] OK
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            
            # Gemini-family [1] - Use T5 as alternative since Gemini is based on T5 OK
            "t5-base", 
            
            # GPT-family [2] OK
            "EleutherAI/gpt-j-6B",

            # LLaMA3-family [3] OK
            "meta-llama/Meta-Llama-3-8B",
            
            # LLaMA [4] OK
            "openlm-research/open_llama_3b", 

            # Mixtral [5] 
            "mistralai/Mixtral-8x7B-v0.1", 

            # Claude-family [6] - Use Llama as alternative since Claude tokenizer is similar to Llama tokenizer OK
            "meta-llama/Meta-Llama-3-8B"
        ]
        self._tokenizer_cache = {}
    
    def _get_first_sentence(self, text):
        """Extract the first sentence from text"""
        # Split by sentence-ending punctuation followed by space/newline
        sentences = re.split(r'[.!?]+\s+', text.strip())
        return sentences[0] if sentences else text
    
    def prepare(self):
        '''Loop over folders in folder self.dataset_dir, then loop over files (train.jsonl, test.jsonl and valid.jsonl), but just consider 2 files (train.jsonl and test.jsonl).
           Output file jsonl, each line has 3 fields: text, prompt_len and label.
           Where: field text take directly from the files, label take from folders' names in self.dataset_dir, field prompt_len by using Tokenizer in HuggingFace via library transformers respectively.
        '''
        for dataset in os.listdir(self.datasets_dir):
            dataset_dir = os.path.join(self.datasets_dir, dataset)
            if not os.path.isdir(dataset_dir):
                continue
                
            train_data_path = os.path.join(dataset_dir, 'train_data.jsonl')
            test_data_path = os.path.join(dataset_dir, 'test_data.jsonl')
            
            train_data = []
            test_data = []
            
            for model_text in os.listdir(dataset_dir):
                model_text_dir = os.path.join(dataset_dir, model_text)
                if not os.path.isdir(model_text_dir):
                    continue
                
                # Clear cache if it gets too large to prevent memory overflow
                if len(self._tokenizer_cache) > 2:
                    self._tokenizer_cache.clear()
                    gc.collect()
                
                for file in os.listdir(model_text_dir):
                    file_path = os.path.join(model_text_dir, file)
                    if not os.path.isfile(file_path):
                        continue
                    
                    if 'train' in file:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f.readlines():
                                line_data = json.loads(line.strip())
                                text = line_data['text']
                                label = model_text
                                
                                if model_text.lower() == 'human-text':
                                    output_data = {
                                        'text': text,
                                        'label': label
                                    }
                                elif 'human' not in model_text.lower():
                                    tokenizer_name = self._get_tokenizer_name(model_text)
                                    tokenizer = self._get_tokenizer(tokenizer_name)
                                    prompt_len = self._get_token_count(tokenizer, text, tokenizer_name)
                                    output_data = {
                                        'text': text,
                                        'prompt_len': prompt_len,
                                        'label': label
                                    }
                                else: # human + AI
                                    tokenizer_name = self._get_tokenizer_name(model_text)
                                    tokenizer = self._get_tokenizer(tokenizer_name)
                                    first_sentence = self._get_first_sentence(text)
                                    prompt_len = self._get_token_count(tokenizer, first_sentence, tokenizer_name)
                                    output_data = {
                                        'text': text,
                                        'prompt_len': prompt_len,
                                        'label': label
                                    }
                                train_data.append(output_data)
                                
                    elif 'test' in file:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f.readlines():
                                line_data = json.loads(line.strip())
                                text = line_data['text']
                                label = model_text
                                
                                if model_text.lower() == 'human-text':
                                    output_data = {
                                        'text': text,
                                        'label': label
                                    }
                                elif 'human' not in model_text.lower():
                                    tokenizer_name = self._get_tokenizer_name(model_text)
                                    tokenizer = self._get_tokenizer(tokenizer_name)
                                    prompt_len = self._get_token_count(tokenizer, text, tokenizer_name)
                                    output_data = {
                                        'text': text,
                                        'prompt_len': prompt_len,
                                        'label': label
                                    }
                                else: # human + AI
                                    tokenizer_name = self._get_tokenizer_name(model_text)
                                    tokenizer = self._get_tokenizer(tokenizer_name)
                                    first_sentence = self._get_first_sentence(text)
                                    prompt_len = self._get_token_count(tokenizer, first_sentence, tokenizer_name)
                                    output_data = {
                                        'text': text,
                                        'prompt_len': prompt_len,
                                        'label': label
                                    }
                                test_data.append(output_data)
            
            # Write train data
            with open(train_data_path, 'w', encoding='utf-8') as train_f:
                for data in train_data:
                    train_f.write(json.dumps(data) + '\n')
            
            # Write test data
            with open(test_data_path, 'w', encoding='utf-8') as test_f:
                for data in test_data:
                    test_f.write(json.dumps(data) + '\n')
            
            # Clear data from memory after writing
            del train_data, test_data
            gc.collect()

    def _get_tokenizer(self, tokenizer_name):
        if tokenizer_name not in self._tokenizer_cache:
            self._tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True
            )

        return self._tokenizer_cache[tokenizer_name]

    def _get_token_count(self, tokenizer, text, tokenizer_name):
        """Get token count based on tokenizer type"""
        if tokenizer_name == 'TheBloke/claude2-alpaca-13B-GGUF':
            # GGUF model uses built-in tokenizer
            tokens = tokenizer.tokenize(text[:1024])  # Limit text length
            return len(tokens)
        else:
            # Standard HuggingFace tokenizers
            # Truncate text if too long to avoid errors
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=512)
            return len(tokens)

    def _get_tokenizer_name(self, model_text):
        model_text_lower = model_text.lower()
        if 'deepseek' in model_text_lower:
            return self.tokenizers[0]
        elif 'gemini' in model_text_lower or 'gemma' in model_text_lower:
            return self.tokenizers[1]
        elif 'gpt' in model_text_lower:
            return self.tokenizers[2]
        elif 'llama3' in model_text_lower:
            return self.tokenizers[3]
        elif 'llama' in model_text_lower:
            return self.tokenizers[4]
        elif 'mixtral' in model_text_lower:
            return self.tokenizers[5]
        elif 'claude' in model_text_lower:
            return self.tokenizers[6]
        else:
            # Default tokenizer
            return self.tokenizers[0]


def main():
    train_test_data_preparer = TrainTestDataPreparer()
    train_test_data_preparer.prepare()


if __name__ == '__main__':
    main()