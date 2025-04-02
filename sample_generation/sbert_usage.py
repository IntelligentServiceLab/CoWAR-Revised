from transformers import AutoTokenizer, AutoModel
import torch


"""
    Sentence BERT模型表征过程
"""

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sbert_representation(api_description_dict:dict):

    list_apis, list_descriptions = zip(*api_description_dict.items())  # 直接解包字典

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained('../sbert')   # sentence-transformers/all-mpnet-base-v2版本
    model = AutoModel.from_pretrained('../sbert').to(device)

    encoded_input = tokenizer(list_descriptions, padding=True, truncation=True, return_tensors='pt').to(device)

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu()
    api_embedding_dict = dict(zip(list_apis, sentence_embeddings))

    return api_embedding_dict


if __name__ == "__main__":
    api_description_dict = {
        1: "This is an example sentence.",
        2: "Each sentence is converted.",
        3: "hello, world"
    }

    api_embedding_dict = sbert_representation(api_description_dict)
    print(api_embedding_dict)
