from loguru import logger
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
from transformers import PreTrainedTokenizerFast
import hydra
from baby_llama.config import DatasetConfig

@hydra.main(config_path="config/dataset", config_name="tinyshakespeare.yaml", version_base=None)
def main(cfg: DatasetConfig) -> None:   
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    # tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="Ġ", add_prefix_space=False)
    # tokenizer.decoder = decoders.Metaspace(replacement="Ġ", add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
    
    tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 2), ("[EOS]", 3)],
    )

    tokenizer.train([cfg.path], trainer)
    
    encodings = tokenizer.encode("CORIOLANUS: \n It is apart \n That I shall blush in acting, and might well \n Be taken from the people.")
    decodings = tokenizer.decode(encodings.ids)
    print(f"Token Ids: {encodings.ids}")
    print(f"Encoded Tokens : {encodings.tokens}")
    print(f"Decoded Tokens: {decodings}")
    
    logger.info(f"Tokenizer created with vocab size: {tokenizer.get_vocab_size()}")
    
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    pretrained_tokenizer.save_pretrained(cfg.tokenizer_path)


if __name__ == "__main__":
    main()


