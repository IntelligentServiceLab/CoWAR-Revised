import time
import pickle
from data_preprocessing.preprocessing import get_preprocessed_data
from sbert_usage import sbert_representation

"""
    说明：本实验使用的Sentence-BERT的版本为all-mpnet-base-v2，该版本基于microsoft/mpnet-base的预训练模型，
    并在包含10亿条句子的数据集上进行微调，以优化性能。具体文件请自行前往Hugging Face进行下载，并将文件放置在sbert文件夹下，
    共计文件数为14。
    包括以下文件：
    - config.json
    - config_sentence_transformers.json
    - data_config.json
    - gitattributes
    - model_safetensors
    - modules.json
    - pytorch_model.bin
    - README.md
    - sentence_bert_config.json
    - special_tokens_map.json
    - tokenizer.json
    - tokenizer_config.json
    - train_script.py
    - vocab.txt
"""

def batch_sbert_representation(api_description_dict, batch_size=32):
    """
    按批次处理 API 描述，避免内存溢出
    """
    list_apis = list(api_description_dict.keys())
    list_descriptions = list(api_description_dict.values())

    total_apis = len(list_apis)
    api_embedding_dict = {}

    for i in range(0, total_apis, batch_size):
        batch_apis = list_apis[i : i + batch_size]
        batch_descriptions = list_descriptions[i : i + batch_size]

        # 处理当前批次
        batch_dict = dict(zip(batch_apis, batch_descriptions))
        batch_embeddings = sbert_representation(batch_dict)

        # 合并结果
        api_embedding_dict.update(batch_embeddings)

        print(f"Processed {i + len(batch_apis)} / {total_apis} APIs")

    return api_embedding_dict


def get_api_representation():
    mashup_dict, valid_api_dict = get_preprocessed_data()
    # 通过遍历已经经过预处理的 API 数据，获取其中的索引值作为 key ，描述信息为 value
    api_description_dict = {
        api_info[0]: api_info[1]
        for api_info in valid_api_dict.values()
    }

    api_representation_dict = batch_sbert_representation(api_description_dict, batch_size=32)

    # 保存数据
    with open('../data/api_representation_dict_sbert.pkl', 'wb') as f_save:
        pickle.dump(api_representation_dict, f_save)


if __name__ == "__main__":
    start_time = time.time()
    get_api_representation()
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")



