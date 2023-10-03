from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
from baby_llama.data.dataloader import CLMDataset

def getfromtext(
    data_path: Path,        
    tokenizer: AutoTokenizer,
    tokenizer_args: dict
    ) -> CLMDataset:
    """
    Reads the txt data from the data_path and returns it in an appropriate format (CLMDataset).
    """
    data = data_path.read_text().split("\n")
    data = [i for i in data if i]  # Remove empty strings
    
    data_max_length = [""]
    for d in tqdm(data):
        sentence_token = tokenizer(data_max_length[-1] + d)
        if len(sentence_token["input_ids"]) < tokenizer_args['max_length']:
            data_max_length[-1] = data_max_length[-1] + "\n" + d
        else:
            data_max_length.append(d)    
            
            
    logger.info(f"Combining samples based on max len: from {len(data)} to {len(data_max_length)} samples.")
                    
    return CLMDataset(data=data_max_length, tokenizer=tokenizer, tokenizer_args=tokenizer_args)