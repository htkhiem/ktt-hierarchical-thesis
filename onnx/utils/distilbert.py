import transformers

def get_pretrained():
    encoder = base_encoder
    encoder.load_state_dict(base_encoder_state)
    return encoder

def init():
    global tokenizer
    global base_encoder
    global base_encoder_state
    tokenizer = transformers.DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    base_encoder = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
    base_encoder_state = base_encoder.state_dict()

if __name__ == "__main__":
    pass
