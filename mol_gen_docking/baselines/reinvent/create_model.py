import argparse
from collections import Counter
from typing import Generator, List

from datasets import Dataset, load_dataset
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
)


def get_training_corpus(dataset: Dataset) -> Generator[List[str], None, None]:
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["smiles"]


def create_reinvent_model(
    N_voc: int = 60,
    num_hidden_layers: int = 6,
    num_attention_heads: int = 24,
    head_dim: int = 32,
) -> None:
    # Load dataset and train a new tokenizer
    ds = load_dataset("jarod0411/zinc10M", split="train").select(range(1000000))
    print("Dataset loaded of length: {}".format(len(ds)))

    # Use BPE model with CharLevel pre-tokenizer to avoid ByteLevelBPE's fixed 256 base tokens
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"\(|\)"), behavior="isolated")

    trainer = trainers.BpeTrainer(
        vocab_size=N_voc,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(get_training_corpus(ds), trainer=trainer)
    tokenizer.decoder = decoders.BPEDecoder()

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_input_names=["input_ids", "attention_mask"],
        clean_up_tokenization_spaces=False,
    )
    fast_tok.add_special_tokens({"pad_token": "<pad>"})
    fast_tok.add_special_tokens({"bos_token": "<s>"})
    fast_tok.add_special_tokens({"eos_token": "</s>"})
    fast_tok.add_special_tokens({"unk_token": "<unk>"})

    print("#-#" * 20)
    print("Computing token occurrences from training set...")

    token_counts: Counter[str] = Counter()
    for row in tqdm(ds.select(range(1000))):
        smiles = row["smiles"]
        tokenized = fast_tok(smiles)
        token_counts.update(tokenized["input_ids"])
    id_to_token = fast_tok.get_vocab()
    token_id_to_string = {v: k for k, v in id_to_token.items()}
    sorted_vocab = sorted(
        [
            (token_id_to_string.get(tok_id, f"<UNK_{tok_id}>"), count)
            for tok_id, count in token_counts.items()
        ],
        key=lambda x: x[1],
        reverse=True,
    )
    max_occ = max(count for _, count in sorted_vocab) if sorted_vocab else 1

    tok_vocab_display = "\n".join(
        [
            "    - {} ({})".format(tok, occ)
            + " " * (40 - len("    - {} ({})".format(tok, occ)))
            + "|"
            + "#" * int(occ / max_occ * 100)
            for tok, occ in sorted_vocab
        ]
    )
    print(f"Tokenizer vocab (sorted by occurrence):\n{tok_vocab_display}")

    print("\nExample tokenizations:")
    for row in ds.select(range(10)):
        smiles = row["smiles"]
        tokenized = fast_tok("<s>" + smiles + "</s>")
        de_tokenized = fast_tok.decode(tokenized["input_ids"])
        print(smiles + "->" + str(tokenized["input_ids"]) + "->" + de_tokenized)
    print("#-#" * 20)

    config = Qwen3Config().__dict__
    config["vocab_size"] = len(fast_tok)
    config["num_hidden_layers"] = num_hidden_layers  # 6
    config["layer_types"] = ["full_attention"] * num_hidden_layers
    config["num_attention_heads"] = num_attention_heads  # 24
    config["head_dim"] = head_dim  # 32

    config["num_key_value_heads"] = config["num_attention_heads"]
    config["hidden_size"] = config["num_attention_heads"] * config["head_dim"]
    config["intermediate_size"] = int(config["hidden_size"] * 2.7)
    print(config)
    cfg = Qwen3Config(**config)
    model = Qwen3ForCausalLM(cfg)
    n_params = model.num_parameters(only_trainable=True)

    if n_params // 1e6 > 0:
        n_params = f"{int(n_params // 1e6)}M"
    else:
        n_params = f"{int(n_params // 1e3)}K"

    NAME = f"reinvent_{n_params}"
    NAME += f"_{N_voc}"
    print("#-#" * 20)
    print(f"Model name : {NAME}")
    print("#-#" * 20)

    assert model.forward(**fast_tok("CCC", return_tensors="pt")) is not None

    fast_tok.save_pretrained(f"./{NAME}")
    fast_tok.push_to_hub(f"Franso/{NAME}")
    model.push_to_hub(f"Franso/{NAME}")

    print(AutoTokenizer.from_pretrained(f"Franso/{NAME}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a REINVENT model with custom architecture"
    )

    parser.add_argument(
        "--vocab_size", type=int, default=48, help="Vocabulary size for the tokenizer"
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=6,
        help="Number of hidden layers in the model",
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=24, help="Number of attention heads"
    )
    parser.add_argument(
        "--head_dim", type=int, default=32, help="Dimension of each attention head"
    )

    args = parser.parse_args()

    create_reinvent_model(
        N_voc=args.vocab_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.head_dim,
    )
