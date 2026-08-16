"""
train_s5.py
kanana-safeguard-8b QLoRA 파인튜닝 — S5(아동 성착취) 카테고리 강화

필요 패키지:
  pip install peft trl bitsandbytes datasets accelerate

사용법:
  python finetune/train_s5.py
  python finetune/train_s5.py --dataset finetune/s5_dataset.json --output finetune/kanana-s5-adapter

과적합 방지 하이퍼파라미터(epoch/lr/dropout/val_ratio/EarlyStopping)는 장서연님이 paper_jsy
브랜치에서 동일 규모 데이터셋으로 실험 중 5 epoch/lr 2e-4에서 eval_loss가 거의 0으로 수렴하는
과적합을 확인하고 수정한 값을 반영함 (2026-08-16).
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# ── 상수 ──────────────────────────────────────────────────────────────────────
BASE_MODEL = "kakaocorp/kanana-safeguard-8b"
PROMPT_TEMPLATE = "<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
RESPONSE_TEMPLATE = "<start_of_turn>model\n"  # completion 시작 구분자
GRAD_ACCUM_STEPS = 4


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


def build_model_and_tokenizer(base_model: str, lora_r: int, lora_alpha: int, lora_dropout: float):
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
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
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
    model, tokenizer = build_model_and_tokenizer(
        BASE_MODEL, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    # completion 부분만 loss 계산 (prompt 제외)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # trl 0.19.1의 SFTConfig는 warmup_ratio 필드가 없고 warmup_steps만 받음
    # (설치된 버전에서 __dataclass_fields__로 실측 확인, 2026-08-16) — 직접 스텝 수로 환산.
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * GRAD_ACCUM_STEPS))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"  스텝/epoch={steps_per_epoch}, 총 스텝={total_steps}, warmup_steps={warmup_steps}")

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        # 과적합 직전(최저 eval_loss) 체크포인트를 최종 결과로 채택 + 조기 종료
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
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
        # eval_loss가 patience 이상 연속으로 개선 안 되면 조기 종료 (과적합 방지)
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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
    parser.add_argument("--epochs", type=int, default=3,
                         help="5 epoch에서 eval_loss가 거의 0으로 수렴(과적합)했던 것을 확인해 기본값을 낮춤")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.15,
                         help="val 표본이 너무 작으면 과적합 감지가 부정확해서 기본값을 올림")
    parser.add_argument("--lr", type=float, default=5e-5,
                         help="기존 2e-4는 200여개 소규모 데이터셋+8B 모델엔 과적합을 유발할 만큼 공격적이었음")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                         help="기존 0.05보다 올려서 과적합 억제")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                         help="총 학습 스텝의 이 비율만큼 웜업. SFTConfig가 warmup_ratio를 직접 "
                              "받지 않는 trl 버전이 있어 여기서 warmup_steps로 환산해 전달함")
    main(parser.parse_args())
