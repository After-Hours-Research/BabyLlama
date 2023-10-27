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
    data = data_path.read_text().split("\n\n")
    data = [i for i in data if i]  # Remove empty strings
    return CLMDataset(data=data, tokenizer=tokenizer, tokenizer_args=tokenizer_args)
