from loguru import logger
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
import hydra
from baby_llama.config import DatasetConfig

@hydra.main(config_path="config/dataset", config_name="tinyshakespeare.yaml", version_base=None)
def main(cfg: DatasetConfig) -> None:   
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(pattern=" ", behavior="removed")
    
    trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "<s>", "</s>"])

    tokenizer.train([cfg.path], trainer)
    
    encodings = tokenizer.encode("CORIOLANUS: \n It is apart \n That I shall blush in acting, and might well \n Be taken from the people.")
    print(f"IDS: {encodings.ids}")
    print(f"Tokens: {encodings.tokens}")
    
    logger.info(f"Tokenizer created with vocab size: {tokenizer.get_vocab_size()}")
    
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    pretrained_tokenizer.save_pretrained(cfg.tokenizer_path)
    
    # tokenizer.save(f"{cfg.tokenizer_path}config.json")
    # tokenizer.model.save(f"{cfg.tokenizer_path}")

if __name__ == "__main__":
    main()


