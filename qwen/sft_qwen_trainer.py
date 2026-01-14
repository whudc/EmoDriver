import logging
import math
import os
import sys
import random
from dataclasses import dataclass, field
from itertools import chain
from datasets import disable_caching
disable_caching()
from typing import Optional,List,Union

import datasets
import evaluate
import torch
import numpy as np
from datasets import load_dataset
from peft import (  # noqa: E402
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
)
from qwen.model import Qwen3ForCausalLM, MapPeftModelForCausalLM
from qwen.trainer import CustomTrainer as Trainer
from qwen.mapping import get_peft_model

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    dataset_cache: Optional[str] = field(default=None, metadata={"help": "Path to the dataset cache"})
    ckpt_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    target_modules: Optional[str] = field(
        default='q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj',
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    load_in_bits: Optional[int] = field(default=8)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    
    add_special_tokens: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Add special tokens to the tokenizer. If there are multiple sequences, they should be separated by ',' without space."
            )
        },
    )
    
    resize_token_embeddings: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to resize the token embeddings matrix of the model. "
                "Useful when adding new tokens to the vocabulary."
            )
        },
    )
    
    layers_to_transform: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "List of layer indices to transform. "
            )
        }
    )
    
    map_input_size: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "The input size of the map encoder. "
            )
        },
    )
    
    freeze_map_adapter: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to freeze map adapter. "
            )
        }
    )
    
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_on_inputs: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_files: Optional[List[str]]  = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_files is None and self.validation_files is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_files is not None:
                extension = self.train_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_files is not None:
                extension = self.validation_files[0].split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
                
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            print('+++++++++++++++++save call back++++++++++++++++')
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            kwargs["model"].save_pretrained(checkpoint_folder, state_dict=kwargs["model"].state_dict())

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            return control


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # pdb.set_trace()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # training_args.gradient_accumulation_steps = 10


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if True:
        data_files = {}
        dataset_args = {}
        if data_args.train_files is not None:
            data_files["train"] = data_args.train_files
        if data_args.validation_files is not None:
            data_files["validation"] = data_args.validation_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            **dataset_args,
        )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": None,
    }
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, **config_kwargs)

    # 如果配置里没有 rope_parameters
    if not hasattr(config, "rope_parameters"):
        config.rope_parameters = {}
        config.rope_parameters['rope_type'] = 'default'
    
    tokenizer_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": None,
        "padding_side":'left'
    }
    if model_args.ckpt_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.ckpt_path, **tokenizer_kwargs)
        print('!!!!!!!! Loading tokenizer from {}'.format(model_args.ckpt_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    # tokenizer.pad_token = tokenizer.eos_token
    # 自动确保特殊 token 一致
    # if tokenizer.bos_token is None:
    #     tokenizer.bos_token = "<|im_start|>"
    # if tokenizer.eos_token is None:
    #     tokenizer.eos_token = "<|im_end|>"
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token  # 常规共享
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

    if model_args.add_special_tokens is not None:
        # 用户输入的特殊 token
        user_tokens = model_args.add_special_tokens.split(',')

        # 获取当前 tokenizer 已经占用的 extra token 数量
        existing_extra_tokens = tokenizer.additional_special_tokens or []
        used_indices = []
        for tok in existing_extra_tokens:
            if tok.startswith('<|extra_') and tok.rstrip('|>').split('_')[-1].isdigit():
                used_indices.append(int(tok.rstrip('|>').split('_')[-1]))
        used_indices = set(used_indices)

        # 分配可用的 extra token
        mapped_tokens = []
        next_idx = 0
        for _ in user_tokens:
            while next_idx in used_indices:
                next_idx += 1
            if next_idx > 204:
                raise ValueError("Exceeded <|extra_0|> to <|extra_204|> range.")
            mapped_tokens.append(f'<|extra_{next_idx}|>')
            used_indices.add(next_idx)
            next_idx += 1

        # 添加到 tokenizer
        special_tokens = {'additional_special_tokens': mapped_tokens}
        if not model_args.ckpt_path:
            tokenizer.add_special_tokens(special_tokens)
        else:
            print('!!!!!!!!!! special tokens will not be added again, check if this is intended !!!!!!!!!!!!')

        # 保存映射
        special_token_ids = tokenizer.convert_tokens_to_ids(mapped_tokens)
        special_token_dict = dict(zip(user_tokens, special_token_ids))
        config.special_token_dict = special_token_dict
        print("Special tokens mapping:", special_token_dict)

    
    ##############################
    config.map_insize = model_args.map_input_size
    if model_args.layers_to_transform is not None:
        model_args.layers_to_transform = [int(num) for num in model_args.layers_to_transform.strip().split(',')]
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules = list(model_args.target_modules.split(',')),
        fan_in_fan_out = False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=model_args.layers_to_transform
    )
    # print(lora_config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    print(torch_dtype)
    print('==================')
    print(int(os.environ.get("LOCAL_RANK")))
    # torch_dtype = torch.float16
    model = Qwen3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=None,
        torch_dtype=torch_dtype,
        load_in_8bit=True if model_args.load_in_bits==8 else False,
        quantization_config=bnb_config if model_args.load_in_bits==4 else None,
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},
        trust_remote_code=True,
    )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size and model_args.resize_token_embeddings:
        print('resize_token_embeddings from {} to {}'.format(embedding_size, len(tokenizer)))
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=2)

    
    if model_args.load_in_bits==8:
        model = prepare_model_for_kbit_training(model)
    elif model_args.load_in_bits==4:
        try:
            model = prepare_model_for_kbit_training(model)
        except:
            print('!!!!!!!!!! prepare_model_for_kbit_training failed, continue anyway !!!!!!!!!!!!')
            import pdb; pdb.set_trace()
    
    column_names = list(raw_datasets["train"].features)
    input_column_name = 'input'
    target_column_name = 'target'
    map_column_name = 'map_info'

    def tokenize(prompt, cutoff_len=data_args.block_size, padding=False, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=padding,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        result["labels"] = result["input_ids"].copy()

        return result
    
    def generate_and_tokenize_prompt(data_point):
        input_text = data_point[input_column_name]
        if 'Final Answer' not in input_text:
            input_text = data_point[input_column_name]
        input_text = input_text.replace('<map>','<map></map>')
        target_text = data_point[target_column_name]
        full_prompt = input_text + target_text
        try:
            map_info = data_point[map_column_name]
        except:
            map_info = None
        if map_info is None or map_info == 'null':
            map_feats = None
            map_masks = None
        else:
            map_info = np.load(map_info, allow_pickle=True)
            map_feats = map_info['encoding'].squeeze(0)
            map_masks = map_info['mask'].squeeze(0)
        
        tokenized_full_prompt = tokenize(full_prompt, add_eos_token=True)
        tokenized_input_text = tokenize(input_text, add_eos_token=True)
        
        input_text_len = len(tokenized_input_text["input_ids"])
        tokenized_full_prompt["labels"] = [-100] * input_text_len + tokenized_full_prompt["labels"][input_text_len:]
        if map_feats is not None:
            tokenized_full_prompt["map_feats"] = torch.from_numpy(map_feats).squeeze(0)
            tokenized_full_prompt["map_masks"] = torch.from_numpy(map_masks).squeeze(0)
        
        return tokenized_full_prompt
    with training_args.main_process_first(desc="dataset map tokenization"):
        if model_args.dataset_cache is not None:
            try:
                from datasets import load_from_disk
                tokenized_datasets = load_from_disk(model_args.dataset_cache)
                print(f"!!!!!!!!!!  ----------------- Loading dataset from {model_args.dataset_cache}")
            except FileNotFoundError:
                tokenized_datasets = raw_datasets.map(
                    generate_and_tokenize_prompt,
                    batched=False,
                    remove_columns=column_names,
                    num_proc=32
                )
                tokenized_datasets.save_to_disk(model_args.dataset_cache)
                logging.info(f"Saving dataset to {model_args.dataset_cache}")
        else:
            tokenized_datasets = raw_datasets.map(
                    generate_and_tokenize_prompt,
                    batched=False,
                    remove_columns=column_names,
                    num_proc=32
                )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 2048:
            block_size = 2048
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set, has {len(train_dataset[index]['attention_mask'])} token.")
        train_dataset = train_dataset.shuffle(seed=training_args.seed)

    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("qwen/metric.py")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    if model_args.ckpt_path:
        model = MapPeftModelForCausalLM.from_pretrained(model, model_args.ckpt_path, is_trainable=True,  config=lora_config)
        print('!!!!!!!!!!!!!!!!!! loading ckpt from {}'.format(model_args.ckpt_path))
    else:
        model = get_peft_model(model, lora_config)
    # freeze map adapter
    if model_args.freeze_map_adapter:
        for name, param in model.named_parameters():
            if 'map_adapter' in name:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            if 'map_adapter' in name:
                param.requires_grad = True
    model.print_trainable_parameters()
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            resume_from_checkpoint = training_args.resume_from_checkpoint
            checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
            if not os.path.exists(checkpoint_name):
                checkpoint_name = os.path.join(
                    resume_from_checkpoint, "adapter_model.safetensors"
                )  # only LoRA model - LoRA config above has to fit
                resume_from_checkpoint = (
                    False  # So the trainer won't try loading its state
                )
            # The two files above have a different name depending on how they were saved, but are actually the same.
            if os.path.exists(checkpoint_name):
                print(f"Restarting from {checkpoint_name}")
                adapters_weights = torch.load(checkpoint_name)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                print(f"Checkpoint {checkpoint_name} not found")
            # checkpoint = Fa
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def load_data(json_path):
    import json
    f = json.load(open(json_path,'r'))
    return f  

def padding_token(input_ids_list, padding_id, padding_side='left'):
    max_length = max([len(i) for i in input_ids_list])
    for i in range(len(input_ids_list)):
        if padding_side == 'left':
            input_ids_list[i] = [padding_id] * (max_length - len(input_ids_list[i])) + input_ids_list[i].tolist()
        elif padding_side == 'right':
            input_ids_list[i] = input_ids_list[i].tolist() + [padding_id] * (max_length - len(input_ids_list[i]))
        else:
            raise ValueError('padding_side must be left or right!')
    return torch.tensor(input_ids_list)
  
if __name__ == "__main__":
    main()