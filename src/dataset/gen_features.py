import random
import httpx
import msgpack
import threading
import time
import os
import argparse
import json
import scipy
import numpy as np
from sklearn.preprocessing import normalize
# from src.config.config import Config as config
from tqdm import tqdm

DEEPSEEK_API = "https://pentagonally-scalelike-delaney.ngrok-free.dev/inference"
T5_API = "https://207bbf789f92.ngrok-free.app/inference"
GPT_API = "https://5b8624fde175.ngrok-free.app/inference"
LLAMA3_API = "https://6eb20e338c8c.ngrok-free.appinference"
LLAMA_API = "https://6eb20e338c8c.ngrok-free.appinference"
MIXTRAL_API = ''
CLAUDE_API = ''


'''
def access_api(text, api_url, do_generate=False):
    """

    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "do_generate": do_generate,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content
'''
def access_api(text, api_url, do_generate=False):
    """
    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    try:
        with httpx.Client(timeout=60) as client:
            post_data = {
                "text": text,
                "do_generate": do_generate,
            }
            
            # Gửi msgpack data như server expect
            packed_data = msgpack.packb(post_data)
            prediction = client.post(api_url, data=packed_data, timeout=60)
            
            if prediction.status_code == 200:
                content = msgpack.unpackb(prediction.content)
                return content
            else:
                print(f"API Error {prediction.status_code}: {prediction.text[:100]}")
                return None
                
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def get_features(type, input_file, output_file):
    """
    get [losses, begin_idx_list, ll_tokens_list, label_int, label] based on raw lines
    """

    en_model_names = ['deepseek', 't5', 'gpt_j', 'llama']
    
    deepseek_api = DEEPSEEK_API 
    t5_api = T5_API
    gpt_j_api = GPT_API 
    claude_api = CLAUDE_API
    llama3_api = LLAMA3_API
    llama_api = LLAMA_API

    en_model_apis = [claude_api, t5_api, gpt_j_api]

    en_labels = {
        'human-text': 0,
        'claude-text': 1,
        'gemini-text': 2,
        'gpt-text': 3,
        # Mixed human-AI labels
        'human---claude-text': 4,
        'human---gemini-text': 5,
        'human---gpt-text': 6
    }

    # line = {'text': '', 'label': ''}
    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]
    # lines = lines[:10]

    print('input file:{}, length:{}'.format(input_file, len(lines)))

    with open(output_file, 'a', encoding='utf-8') as f:
        for data in tqdm(lines):
            line = data['text']
            label = data['label']
            prompt_len = data.get("prompt_len", len(line))

            losses = []
            begin_idx_list = []
            ll_tokens_list = []
            if type == 'en':
                model_apis = en_model_apis
                label_dict = en_labels
            elif type == 'cn':
                '''model_apis = cn_model_apis'''
                # label_dict = cn_labels
                pass

            label_int = label_dict[label]

            error_flag = False
            for api in model_apis:
                try:
                    loss, begin_word_idx, ll_tokens = access_api(line, api)
                    print(f"OK: {api}")
                except TypeError:
                    print("return NoneType, probably gpu OOM, discard this sample")
                    print(api)
                    error_flag = True
                    break
                losses.append(loss)
                begin_idx_list.append(begin_word_idx)
                ll_tokens_list.append(ll_tokens)
            # if oom, discard this sample
            if error_flag:
                continue

            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'll_tokens_list': ll_tokens_list,
                'label_int': label_int,
                'label': label,
                'text': line,
                'prompt_len': prompt_len
            }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def process_features(input_file, output_file, do_normalize=False):
    """
    Process features from raw features.

        raw_features: {losses, begin_idx_list, ll_tokens_list, label_int, label, text}
        ==>
        processed_features: {values, label_int, label}

        values = {losses, lt_zero_percents, std_deviations, pearson_list, spearmann_list}
    """

    # jsonl read
    with open(input_file, 'r') as f:
        raw_features = [json.loads(line) for line in f.readlines()]
    
    # json read
    # with open(input_file, 'r') as f:
    #     raw_features = json.load(f)

    # raw_features = raw_features[:10]
    # raw_features = json.load(open(input_file, 'r', encoding='utf-8'))
    print('input file:{}, length:{}'.format(input_file, len(raw_features)))

    with open(output_file, 'a', encoding='utf-8') as f:
        for raw_feature in tqdm(raw_features):
            losses = raw_feature['losses']
            begin_idx_list = raw_feature['begin_idx_list']
            ll_tokens_list = raw_feature['ll_tokens_list']
            label_int = raw_feature['label_int']
            label = raw_feature['label']
            text = raw_feature['text']


            # losses, begin_idx_list, ll_tokens_list, label_int, label = raw_feature
            #  python gen_features.py --process_features --input_file ../features/raw_features/en_alpaca_features.jsonl --output_file ../features/raw_processed_features/en_alpaca_processed_features.jsonl
            try:
                # ll_tokens_len_list = [len(ll_tokens) for ll_tokens in ll_tokens_list]
                # if ll_tokens_len_list.count(ll_tokens_len_list[0]) != len(ll_tokens_len_list):
                #     print(ll_tokens_len_list)

                # Align all vectors in ll_tokens_list
                # ll_tokens_list = np.array(ll_tokens_list)
                begin_idx_list = np.array(begin_idx_list)
                # Get the maximum value in begin_idx_list, which indicates where we need to truncate.
                max_begin_idx = np.max(begin_idx_list)
                # Truncate all vectors
                for idx, ll_tokens in enumerate(ll_tokens_list):
                    ll_tokens_list[idx] = ll_tokens[max_begin_idx:]
                # ll_tokens_list = ll_tokens_list[:, max_begin_idx:]

                # Get the length of all vectors and take the minimum
                min_len = np.min([len(ll_tokens) for ll_tokens in ll_tokens_list])
                # Align the lengths of all vectors
                for idx, ll_tokens in enumerate(ll_tokens_list):
                    ll_tokens_list[idx] = ll_tokens[:min_len]
                # ll_tokens_list = ll_tokens_list[:, :min_len]

                if do_normalize:
                    # print("normalize: {}".format(do_normalize))
                    # Normalize using L1 normalization
                    ll_tokens_list_normalized = normalize(ll_tokens_list, norm='l1')
                    # Convert back to Python lists
                    lls = ll_tokens_list_normalized.tolist()
                else:
                    # print("normalize: {}".format(do_normalize))
                    lls = ll_tokens_list


            except Exception as e:
                """
                [0, 0, 0, 0], too short, discard this sample
                """
                print(e)
                print("fail to process this sample, discard it, text:{}".format(text))
                print()
                continue

            try:
                lt_zero_percents = []
                std_deviations = []
                deviations = []
                pearson_list = []
                spearmann_list = []
                
                for i in range((len(lls))):
                    for j in range(i + 1, len(lls)):
                        # lls[i], ll[j]
                        deviation_ij = [li - lj for li, lj in zip(lls[i], lls[j])]
                        # `lt` means `less than`
                        deviation_lt_zero_ij = [d < 0 for d in deviation_ij]
                        lt_zero_pct_ij = sum(deviation_lt_zero_ij) / len(
                            deviation_lt_zero_ij)
                        std_ij = np.std(deviation_ij)
                        lt_zero_percents.append(lt_zero_pct_ij)
                        std_deviations.append(std_ij)
                        deviations.append(deviation_ij)
                        pearson = scipy.stats.pearsonr(lls[i], lls[j])[0]
                        spearmann = scipy.stats.spearmanr(lls[i], lls[j])[0]

                        pearson_list.append(pearson)
                        spearmann_list.append(spearmann)

                values = {'losses': losses,
                        'lt_zero_percents': lt_zero_percents,
                        'std_deviations': std_deviations,
                        'pearson_list': pearson_list,
                        'spearmann_list': spearmann_list}

                processed_feature = {'values': values,
                                    'label_int': label_int,
                                    'label': label,
                                    'text': text}

                f.write(json.dumps(processed_feature, ensure_ascii=False) + '\n')
            except:
                print("fail may due to speraman or pearson")
                print(text)
                print(lls[i], lls[j])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str) # ['train_data', 'test_data']
    parser.add_argument('--dataset', type=str) # ['faid', 'hart', 'llm_detectaive']
    # parser.add_argument("--add_loss", type=bool, default=True, help="when processing features, add loss")
    # parser.add_argument("--add_pct", type=bool, default=True, help="when processing features, add lt_zero_pct")
    # parser.add_argument("--add_std", type=bool, default=True, help="when processing features, add std")
    # parser.add_argument("--add_corr", type=bool, default=True, help="when processing features, add corr")

    parser.add_argument("--get_en_features", action="store_true", help="generate en logits and losses")
    parser.add_argument("--get_cn_features", action="store_true", help="generate cn logits and losses")
    parser.add_argument("--get_en_features_multithreading", action="store_true", help="multithreading generate en logits and losses")
    parser.add_argument("--get_cn_features_multithreading", action="store_true", help="multithreading generate cn logits and losses")
    parser.add_argument("--process_features", action="store_true", help="process the raw features")

    parser.add_argument("--do_normalize", action="store_true", help="normalize the features")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    input_file = 'data/datasets/' + args.dataset + '/' + args.data_type + '.jsonl'
    output_file = 'data/datasets/' + args.dataset + '/' + args.data_type + '_with_features.jsonl'

    if args.get_en_features:
        get_features(type='en', input_file=input_file, output_file=output_file)

    elif args.get_cn_features:
        get_features(type='cn', input_file=input_file, output_file=output_file)

    elif args.get_en_features_multithreading:
        en_input_files = ['supervised_learning/raw_data/en_gpt2_lines_all.jsonl',
                    'supervised_learning/raw_data/en_gptj_lines_all.jsonl',
                    'supervised_learning/raw_data/en_gptneo_lines_all.jsonl',
                    'supervised_learning/raw_data/en_human_lines_all.jsonl',
                    'supervised_learning/raw_data/en_llama_lines_all.jsonl']

        en_output_files = ['../features/supervised_learning_features/en_gpt2_features.jsonl',
                           '../features/supervised_learning_features/en_gptj_features.jsonl',
                           '../features/supervised_learning_features/en_gptneo_features.jsonl',
                           '../features/supervised_learning_features/en_human_features.jsonl',
                           '../features/supervised_learning_features/en_llama_features.jsonl']

        threads = []
        for i in range(len(en_input_files)):
            t = threading.Thread(target=get_features, args=('en', en_input_files[i], en_output_files[i]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    elif args.get_cn_features_multithreading:
        cn_input_files = ['raw_data/cn_human_lines.jsonl',
                          'raw_data/cn_gpt3re_lines.jsonl',
                          'raw_data/cn_gpt3sum_lines.jsonl',
                          'raw_data/cn_chatglm_lines.jsonl',
                          'raw_data/cn_wenzhong_lines.jsonl',
                          'raw_data/cn_damo_lines.jsonl',
                          'raw_data/cn_sky_text_lines.jsonl',
                          'aligned_data/cn_human_aligned_lines.jsonl',
                          'aligned_data/cn_gpt3re_aligned_lines.jsonl',
                          'aligned_data/cn_gpt3sum_aligned_lines.jsonl',
                          'aligned_data/cn_chatglm_aligned_lines.jsonl',
                          'aligned_data/cn_wenzhong_aligned_lines.jsonl',
                          'aligned_data/cn_damo_aligned_lines.jsonl',
                          'aligned_data/cn_sky_text_aligned_lines.jsonl']
        cn_output_files = ['../features/raw_features/cn_human_features.jsonl',
                           '../features/raw_features/cn_gpt3re_features.jsonl',
                           '../features/raw_features/cn_gpt3sum_features.jsonl',
                           '../features/raw_features/cn_chatglm_features.jsonl',
                           '../features/raw_features/cn_wenzhong_features.jsonl',
                           '../features/raw_features/cn_damo_features.jsonl',
                           '../features/raw_features/cn_sky_text_features.jsonl',
                           '../features/aligned_features/cn_human_aligned_features.jsonl',
                           '../features/aligned_features/cn_gpt3re_aligned_features.jsonl',
                           '../features/aligned_features/cn_gpt3sum_aligned_features.jsonl',
                           '../features/aligned_features/cn_chatglm_aligned_features.jsonl',
                           '../features/aligned_features/cn_wenzhong_aligned_features.jsonl',
                           '../features/aligned_features/cn_damo_aligned_features.jsonl',
                           '../features/aligned_features/cn_sky_text_aligned_features.jsonl']
        threads = []
        for i in range(len(cn_input_files)):
            t = threading.Thread(target=get_features, args=('cn', cn_input_files[i], cn_output_files[i]))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    elif args.process_features:
        
        print(args.do_normalize)
        process_features(input_file, output_file, args.do_normalize)

    else:
        print("please select an action")
