"""Seed the ChromaDB knowledge base with ML failure mode entries.

Run from the project root:
    python scripts/seed_kb.py

Idempotent: only inserts entries whose ID is not already in the collection.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from backend.kb.chroma_store import get_store

KB_ENTRIES: list[dict] = [
    # ------------------------------------------------------------------
    # OOM / Hardware  (5 entries)
    # ------------------------------------------------------------------
    {
        "id": "oom_001",
        "symptom": "CUDA out of memory during forward pass",
        "diagnosis": "Batch size too large for GPU VRAM",
        "fix": "Reduce batch size by 50% or enable gradient checkpointing",
        "code_snippet": (
            "model = torch.utils.checkpoint.checkpoint_sequential(model, segments, input)"
        ),
        "citation": "PyTorch docs: torch.utils.checkpoint",
        "domain": "oom",
    },
    {
        "id": "oom_002",
        "symptom": "CUDA out of memory during backward pass",
        "diagnosis": "Gradient accumulation buffers exceeding VRAM",
        "fix": "Use gradient accumulation instead of large batches",
        "code_snippet": (
            "optimizer.zero_grad()\n"
            "loss = loss / accum_steps\n"
            "loss.backward()\n"
            "if step % accum_steps == 0:\n"
            "    optimizer.step()"
        ),
        "citation": "Ott et al., 2018 - Scaling Neural MT",
        "domain": "oom",
    },
    {
        "id": "oom_003",
        "symptom": "Expected object of device type cuda but got device type cpu",
        "diagnosis": "Input tensors not moved to GPU",
        "fix": "Move all inputs to device explicitly",
        "code_snippet": (
            "inputs = inputs.to(device)\n"
            "labels = labels.to(device)"
        ),
        "citation": "PyTorch Common Errors",
        "domain": "oom",
    },
    {
        "id": "oom_004",
        "symptom": "NCCL error in distributed training",
        "diagnosis": "Process group not initialized before DDP",
        "fix": "Initialize process group before model wrap",
        "code_snippet": (
            "dist.init_process_group(backend='nccl')\n"
            "# then wrap:\n"
            "model = DistributedDataParallel(model)"
        ),
        "citation": "PyTorch Distributed Training Docs",
        "domain": "oom",
    },
    {
        "id": "oom_005",
        "symptom": "GPU utilization low despite training under 30 percent",
        "diagnosis": "Data loading bottleneck, not compute bottleneck",
        "fix": "Increase DataLoader workers and enable pin_memory",
        "code_snippet": (
            "DataLoader(dataset, num_workers=8, pin_memory=True)"
        ),
        "citation": "PyTorch Performance Tuning Guide",
        "domain": "oom",
    },
    # ------------------------------------------------------------------
    # Training Instability  (5 entries)
    # ------------------------------------------------------------------
    {
        "id": "instability_001",
        "symptom": "Loss spikes to Inf or NaN after initially decreasing",
        "diagnosis": "Exploding gradients or learning rate too high",
        "fix": "Clip gradients and reduce LR by 10x",
        "code_snippet": (
            "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)"
        ),
        "citation": "Goodfellow et al., Deep Learning Book Ch.8",
        "domain": "instability",
    },
    {
        "id": "instability_002",
        "symptom": "Loss decreases very slowly in early epochs slope greater than negative 0.01",
        "diagnosis": "Learning rate too low or poor initialization",
        "fix": "Increase LR by 5x or use LR warmup",
        "code_snippet": (
            "scheduler = get_linear_schedule_with_warmup(\n"
            "    optimizer,\n"
            "    num_warmup_steps=100,\n"
            "    num_training_steps=total\n"
            ")"
        ),
        "citation": "Smith, 2018 - Cyclical LR",
        "domain": "instability",
    },
    {
        "id": "instability_003",
        "symptom": "Training loss decreasing but validation loss increasing from epoch 3",
        "diagnosis": "Overfitting, model capacity too high for dataset",
        "fix": "Add dropout, reduce model size, or add weight decay",
        "code_snippet": (
            "nn.Dropout(0.3)\n"
            "optimizer = AdamW(params, weight_decay=0.01)"
        ),
        "citation": "Srivastava et al., 2014 - Dropout",
        "domain": "instability",
    },
    {
        "id": "instability_004",
        "symptom": "Loss oscillates without converging",
        "diagnosis": "Learning rate too high for batch size",
        "fix": "Use linear LR scaling rule: LR = base_lr * (batch_size / 256)",
        "code_snippet": (
            "optimizer = SGD(params, lr=0.1 * batch_size / 256, momentum=0.9)"
        ),
        "citation": "Goyal et al., 2017 - Accurate Large Minibatch SGD",
        "domain": "instability",
    },
    {
        "id": "instability_005",
        "symptom": "Loss NaN from first step",
        "diagnosis": "Numerical instability in loss function or bad input data",
        "fix": "Check for NaN/Inf in inputs; use label smoothing",
        "code_snippet": (
            "assert not torch.isnan(inputs).any()\n"
            "criterion = LabelSmoothingLoss(smoothing=0.1)"
        ),
        "citation": "PyTorch Numerical Stability Docs",
        "domain": "instability",
    },
    # ------------------------------------------------------------------
    # Shape Mismatch  (5 entries)
    # ------------------------------------------------------------------
    {
        "id": "shape_001",
        "symptom": "RuntimeError Expected input batch_size to match target batch_size",
        "diagnosis": "Final layer output dim doesn't match num_classes",
        "fix": "Check classifier head output features equals num_classes",
        "code_snippet": (
            "nn.Linear(hidden_dim, num_classes)  # verify num_classes matches dataset"
        ),
        "citation": "PyTorch nn.Module Docs",
        "domain": "shape_mismatch",
    },
    {
        "id": "shape_002",
        "symptom": "RuntimeError mat1 and mat2 shapes cannot be multiplied",
        "diagnosis": "Incorrect flatten before Linear layer",
        "fix": "Use nn.Flatten() or calculate correct input features",
        "code_snippet": (
            "x = torch.flatten(x, 1)\n"
            "# or\n"
            "nn.AdaptiveAvgPool2d((1, 1))  # before Linear"
        ),
        "citation": "PyTorch Common Errors",
        "domain": "shape_mismatch",
    },
    {
        "id": "shape_003",
        "symptom": "RuntimeError the size of tensor a must match tensor b at non-singleton dimension",
        "diagnosis": "Broadcasting failure in loss or attention",
        "fix": "Explicitly reshape tensors before operations",
        "code_snippet": (
            "x = x.unsqueeze(1).expand_as(y)"
        ),
        "citation": "PyTorch Broadcasting Semantics Docs",
        "domain": "shape_mismatch",
    },
    {
        "id": "shape_004",
        "symptom": "Expected 4D input but got 3D input",
        "diagnosis": "Missing batch dimension",
        "fix": "Add batch dimension with unsqueeze",
        "code_snippet": (
            "x = x.unsqueeze(0)  # adds batch dim"
        ),
        "citation": "PyTorch Tensor Docs",
        "domain": "shape_mismatch",
    },
    {
        "id": "shape_005",
        "symptom": "Index out of bounds in embedding layer",
        "diagnosis": "vocab_size in config smaller than actual vocabulary",
        "fix": "Set vocab_size = tokenizer.vocab_size + special_tokens",
        "code_snippet": (
            "embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)"
        ),
        "citation": "HuggingFace Tokenizer Docs",
        "domain": "shape_mismatch",
    },
    # ------------------------------------------------------------------
    # Transformer / Attention  (6 entries)
    # ------------------------------------------------------------------
    {
        "id": "transformer_001",
        "symptom": "Attention scores are NaN after first few steps with padding tokens",
        "diagnosis": "Padding mask not applied before softmax causing -inf to propagate",
        "fix": "Apply additive mask of -1e9 for padding positions before softmax",
        "code_snippet": (
            "# additive mask: 0 for real tokens, -1e9 for padding\n"
            "mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9\n"
            "scores = scores + mask\n"
            "attn_weights = F.softmax(scores, dim=-1)"
        ),
        "citation": "Vaswani et al., 2017 - Attention Is All You Need",
        "domain": "instability",
    },
    {
        "id": "transformer_002",
        "symptom": "RuntimeError expected all tensors on same device in encoder decoder cross attention",
        "diagnosis": "Encoder output and decoder input on different devices in seq2seq model",
        "fix": "Move encoder outputs to the same device as decoder inputs explicitly",
        "code_snippet": (
            "encoder_outputs = encoder_outputs.to(decoder_input.device)"
        ),
        "citation": "PyTorch Common Errors",
        "domain": "oom",
    },
    {
        "id": "transformer_003",
        "symptom": "Loss NaN only during validation not training with transformer model",
        "diagnosis": "Dropout not disabled during eval — model.eval() not called",
        "fix": "Always call model.eval() and torch.no_grad() before validation loop",
        "code_snippet": (
            "model.eval()\n"
            "with torch.no_grad():\n"
            "    outputs = model(inputs)\n"
            "    val_loss = criterion(outputs, labels)"
        ),
        "citation": "PyTorch Training Best Practices",
        "domain": "instability",
    },
    {
        "id": "transformer_004",
        "symptom": "Positional encoding shape mismatch sequence length dimension error",
        "diagnosis": "Positional encoding buffer shorter than input sequence length",
        "fix": "Generate positional encodings up to max_seq_len and slice to actual length",
        "code_snippet": (
            "# in PositionalEncoding forward:\n"
            "x = x + self.pe[:, :x.size(1), :]  # slice to actual seq_len"
        ),
        "citation": "PyTorch Transformer Tutorial",
        "domain": "shape_mismatch",
    },
    {
        "id": "transformer_005",
        "symptom": "past_key_values shape error during autoregressive generation inference",
        "diagnosis": "KV cache dimensions inconsistent with current step batch size or heads",
        "fix": "Ensure batch size and num_heads are consistent; clear cache between unrelated sequences",
        "code_snippet": (
            "# reset cache between independent inference calls\n"
            "past_key_values = None\n"
            "outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)\n"
            "past_key_values = outputs.past_key_values"
        ),
        "citation": "HuggingFace Generation Docs",
        "domain": "shape_mismatch",
    },
    {
        "id": "transformer_006",
        "symptom": "Multi-head attention output all zeros or constant after first epoch",
        "diagnosis": "Query and key projections initialized identically causing attention collapse",
        "fix": "Use Xavier or Kaiming init on projection weights; ensure separate weight matrices",
        "code_snippet": (
            "nn.init.xavier_uniform_(self.q_proj.weight)\n"
            "nn.init.xavier_uniform_(self.k_proj.weight)\n"
            "nn.init.xavier_uniform_(self.v_proj.weight)"
        ),
        "citation": "Glorot & Bengio, 2010 - Understanding difficulty of training deep networks",
        "domain": "instability",
    },
    # ------------------------------------------------------------------
    # Mixed Precision / autocast  (4 entries)
    # ------------------------------------------------------------------
    {
        "id": "mixed_precision_001",
        "symptom": "Loss becomes exactly 0 after a few steps with autocast enabled gradients all zero",
        "diagnosis": "Loss scale underflow in GradScaler — gradients being zeroed silently",
        "fix": "Check scaler.get_scale() each step; if scale drops below 1.0 disable AMP or tune init_scale",
        "code_snippet": (
            "scaler = torch.cuda.amp.GradScaler(init_scale=2**16)\n"
            "with torch.autocast(device_type='cuda'):\n"
            "    loss = criterion(model(inputs), labels)\n"
            "scaler.scale(loss).backward()\n"
            "scaler.step(optimizer)\n"
            "scaler.update()\n"
            "# monitor: print(scaler.get_scale())"
        ),
        "citation": "PyTorch Automatic Mixed Precision Guide",
        "domain": "instability",
    },
    {
        "id": "mixed_precision_002",
        "symptom": "RuntimeError expected scalar type Float but found Half in loss or metric computation",
        "diagnosis": "Loss function or metric receives float16 tensor outside autocast context",
        "fix": "Cast model output to float32 before computing loss outside autocast block",
        "code_snippet": (
            "with torch.autocast(device_type='cuda'):\n"
            "    logits = model(inputs)  # float16 inside\n"
            "loss = criterion(logits.float(), labels)  # cast before loss"
        ),
        "citation": "PyTorch AMP Best Practices",
        "domain": "instability",
    },
    {
        "id": "mixed_precision_003",
        "symptom": "NaN loss only with autocast on large transformer but not with float32",
        "diagnosis": "Softmax overflow in float16 — attention logits exceed float16 max before scaling",
        "fix": "Scale dot-product attention by sqrt(d_k) and use flash attention or scaled_dot_product_attention",
        "code_snippet": (
            "# PyTorch 2.0+\n"
            "attn_output = F.scaled_dot_product_attention(\n"
            "    query, key, value, attn_mask=mask, dropout_p=0.1\n"
            ")"
        ),
        "citation": "Dao et al., 2022 - FlashAttention",
        "domain": "instability",
    },
    {
        "id": "mixed_precision_004",
        "symptom": "Training with torch.compile and autocast raises graph break or triton error",
        "diagnosis": "torch.compile and autocast require consistent dtype annotations across the graph",
        "fix": "Wrap compile outside autocast and set dtype explicitly in model forward",
        "code_snippet": (
            "model = torch.compile(model)\n"
            "# inside training loop:\n"
            "with torch.autocast('cuda', dtype=torch.bfloat16):\n"
            "    loss = model(inputs)"
        ),
        "citation": "PyTorch torch.compile Docs",
        "domain": "instability",
    },
    # ------------------------------------------------------------------
    # Data Pipeline  (5 entries)
    # ------------------------------------------------------------------
    {
        "id": "data_001",
        "symptom": "Training accuracy suspiciously high from epoch 1 near 95 percent immediately",
        "diagnosis": "Label leakage — target labels visible in input features or data split contaminated",
        "fix": "Audit feature pipeline; ensure train/val split happens before any feature engineering",
        "code_snippet": (
            "# correct split order\n"
            "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)\n"
            "# fit scaler ONLY on train, transform both\n"
            "scaler.fit(X_train)\n"
            "X_train = scaler.transform(X_train)\n"
            "X_val = scaler.transform(X_val)"
        ),
        "citation": "Kaufman et al., 2012 - Leakage in Data Mining",
        "domain": "instability",
    },
    {
        "id": "data_002",
        "symptom": "Validation accuracy high but test accuracy low model works in notebook not production",
        "diagnosis": "Normalization mismatch — model trained with different mean/std than inference data",
        "fix": "Save and reuse the exact training normalization statistics at inference time",
        "code_snippet": (
            "# save with model checkpoint\n"
            "torch.save({'model': model.state_dict(),\n"
            "            'mean': train_mean, 'std': train_std}, 'checkpoint.pt')\n"
            "# at inference\n"
            "ckpt = torch.load('checkpoint.pt')\n"
            "transform = Normalize(mean=ckpt['mean'], std=ckpt['std'])"
        ),
        "citation": "CS231n - Data Preprocessing",
        "domain": "instability",
    },
    {
        "id": "data_003",
        "symptom": "Loss plateaus from epoch 1 with no improvement DataLoader shuffle False",
        "diagnosis": "Training data not shuffled — model sees same batch order every epoch causing local optima",
        "fix": "Set shuffle=True in training DataLoader; never shuffle validation loader",
        "code_snippet": (
            "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n"
            "val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)"
        ),
        "citation": "PyTorch DataLoader Docs",
        "domain": "instability",
    },
    {
        "id": "data_004",
        "symptom": "Loss decreasing but accuracy stuck near majority class percentage class imbalance",
        "diagnosis": "Class imbalance — model collapses to predicting majority class",
        "fix": "Use weighted loss, oversample minority class, or use focal loss",
        "code_snippet": (
            "# weighted cross-entropy\n"
            "class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)\n"
            "criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))\n"
            "# or focal loss alpha parameter"
        ),
        "citation": "Lin et al., 2017 - Focal Loss for Dense Object Detection",
        "domain": "instability",
    },
    {
        "id": "data_005",
        "symptom": "Training loss much lower than validation loss from epoch 1 even with dropout",
        "diagnosis": "Train and validation sets have different preprocessing or augmentation applied to val",
        "fix": "Apply augmentation only to training set; apply only normalization to validation set",
        "code_snippet": (
            "train_transform = transforms.Compose([\n"
            "    transforms.RandomHorizontalFlip(),\n"
            "    transforms.ColorJitter(),\n"
            "    transforms.ToTensor(),\n"
            "    transforms.Normalize(mean, std),\n"
            "])\n"
            "val_transform = transforms.Compose([\n"
            "    transforms.ToTensor(),\n"
            "    transforms.Normalize(mean, std),  # normalization only\n"
            "])"
        ),
        "citation": "PyTorch torchvision.transforms Docs",
        "domain": "instability",
    },
    # ------------------------------------------------------------------
    # LR Scheduler Misuse  (5 entries)
    # ------------------------------------------------------------------
    {
        "id": "scheduler_001",
        "symptom": "UserWarning detected call of lr_scheduler.step before optimizer.step",
        "diagnosis": "LR scheduler stepped before optimizer in PyTorch >= 1.1 causing incorrect LR",
        "fix": "Always call optimizer.step() before scheduler.step() each iteration",
        "code_snippet": (
            "# correct order every training step\n"
            "loss.backward()\n"
            "optimizer.step()\n"
            "scheduler.step()  # AFTER optimizer"
        ),
        "citation": "PyTorch LR Scheduler Docs",
        "domain": "instability",
    },
    {
        "id": "scheduler_002",
        "symptom": "Learning rate drops to near zero after first epoch cosine annealing scheduler",
        "diagnosis": "CosineAnnealingLR T_max set to number of epochs but scheduler called per batch",
        "fix": "Set T_max = num_epochs * steps_per_epoch when stepping per batch",
        "code_snippet": (
            "steps_per_epoch = len(train_loader)\n"
            "scheduler = CosineAnnealingLR(\n"
            "    optimizer,\n"
            "    T_max=num_epochs * steps_per_epoch  # total steps, not epochs\n"
            ")"
        ),
        "citation": "PyTorch CosineAnnealingLR Docs",
        "domain": "instability",
    },
    {
        "id": "scheduler_003",
        "symptom": "Loss spikes periodically at regular epoch intervals with cyclic learning rate",
        "diagnosis": "Cyclical LR cycle length too short causing LR to spike training loss repeatedly",
        "fix": "Increase cycle length to at least 4-8 epochs; use warmup_steps >= 5% of total steps",
        "code_snippet": (
            "scheduler = OneCycleLR(\n"
            "    optimizer,\n"
            "    max_lr=1e-3,\n"
            "    steps_per_epoch=len(train_loader),\n"
            "    epochs=num_epochs,\n"
            "    pct_start=0.3  # 30% warmup\n"
            ")"
        ),
        "citation": "Smith & Topin, 2019 - Super-Convergence",
        "domain": "instability",
    },
    {
        "id": "scheduler_004",
        "symptom": "Learning rate does not change despite using ReduceLROnPlateau scheduler",
        "diagnosis": "ReduceLROnPlateau monitors wrong metric or patience too high",
        "fix": "Pass validation loss explicitly to scheduler.step(); reduce patience to 3-5 epochs",
        "code_snippet": (
            "val_loss = validate(model, val_loader)\n"
            "scheduler.step(val_loss)  # must pass the monitored metric\n"
            "# check current LR:\n"
            "print(optimizer.param_groups[0]['lr'])"
        ),
        "citation": "PyTorch ReduceLROnPlateau Docs",
        "domain": "instability",
    },
    {
        "id": "scheduler_005",
        "symptom": "Model fails to converge with warmup scheduler learning rate stays at warmup value",
        "diagnosis": "Warmup scheduler not chained with decay scheduler after warmup phase ends",
        "fix": "Use SequentialLR to chain warmup scheduler with cosine/linear decay",
        "code_snippet": (
            "warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)\n"
            "decay  = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)\n"
            "scheduler = SequentialLR(\n"
            "    optimizer,\n"
            "    schedulers=[warmup, decay],\n"
            "    milestones=[warmup_steps]\n"
            ")"
        ),
        "citation": "PyTorch SequentialLR Docs",
        "domain": "instability",
    },
]


def main() -> None:
    store = get_store()
    existing = store.existing_ids()

    new_entries = [e for e in KB_ENTRIES if e["id"] not in existing]

    if not new_entries:
        print(f"Knowledge base already up to date ({store.count()} docs). Nothing to add.")
        return

    print(f"Found {len(existing)} existing docs. Adding {len(new_entries)} new entries...")
    store.add_documents(new_entries)
    print(f"Done. Knowledge base now has {store.count()} documents.")


if __name__ == "__main__":
    main()
