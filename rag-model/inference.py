from transformers import AutoTokenizer, AutoModel
import torch

def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return tokenizer, model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        import json
        return json.loads(request_body)
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model_components):
    tokenizer, model = model_components
    inputs = tokenizer(input_data['inputs'], return_tensors='pt', truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.squeeze().tolist()

def output_fn(prediction, response_content_type):
    import json
    return json.dumps(prediction)
