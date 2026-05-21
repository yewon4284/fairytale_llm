"""
train_s5.py
kanana-safeguard-8b QLoRA 파인튜닝 — S5(아동 성착취) 카테고리 강화

필요 패키지:
  pip install peft trl bitsandbytes datasets accelerate

사용법:
  python finetune/train_s5.py
  python finetune/train_s5.py --dataset finetune/s5_dataset.json --output finetune/kanana-s5-adapter
"""

import argparse
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ── 상수 ──────────────────────────────────────────────────────────────────────
BASE_MODEL = "kakaocorp/kanana-safeguard-8b"
PROMPT_TEMPLATE = "<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
RESPONSE_TEMPLATE = "<start_of_turn>model\n"  # completion 시작 구분자


def load_dataset(path: str, val_ratio: float = 0.1):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    random.shuffle(raw)

    records = []
    for item in raw:
        text = item["text"].strip()
        label = item["label"].strip()
        formatted = f"<{label}>"          # <SAFE> or <UNSAFE-S5>
        full = PROMPT_TEMPLATE.format(text=text) + formatted
        records.append({"text": full})

    val_n = max(1, int(len(records) * val_ratio))
    return (
        Dataset.from_list(records[val_n:]),   # train
        Dataset.from_list(records[:val_n]),   # val
    )


def build_model_and_tokenizer(base_model: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        # Gemma 계열 어텐션 레이어
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main(args):
    print(f"[1/3] 데이터셋 로딩: {args.dataset}")
    train_ds, val_ds = load_dataset(args.dataset, args.val_ratio)
    print(f"  train {len(train_ds)}개 / val {len(val_ds)}개")

    print(f"[2/3] 모델 로딩: {BASE_MODEL}")
    model, tokenizer = build_model_and_tokenizer(BASE_MODEL)

    # completion 부분만 loss 계산 (prompt 제외)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
        optim="paged_adamw_8bit",
        max_seq_length=256,
        dataset_text_field="text",
    )

    print("[3/3] 파인튜닝 시작")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        args=sft_config,
    )

    trainer.train()

    adapter_path = Path(args.output) / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\n어댑터 저장 완료: {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="finetune/s5_dataset.json")
    parser.add_argument("--output", default="finetune/kanana-s5-adapter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    main(parser.parse_args())
