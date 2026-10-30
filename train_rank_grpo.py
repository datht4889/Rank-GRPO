import os
import wandb
from datasets import load_from_disk
from trl import GRPOConfig, RankGRPOTrainer
import argparse

from libs.data import load_catalog
from libs.utils import StepLRSchedulerCallback
from libs.reward_funcs import make_reward_func, make_reward_func_individual
from libs.logs import setup_environment, setup_wandb


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Rank-GRPO with user feedback as verifiable reward.\n\n"
            "This script post-trains an SFT model using offline reinforcement learning (GRPO), "
            "where rank-level rewards are derived from user preference data."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    core = parser.add_argument_group("Core setup")
    core.add_argument(
        "--train_path",
        required=True,
        help=(
            "Path to the training dataset directory (load_from_disk). "
            "Only the training split is loaded — validation is performed via saved checkpoints."
        ),
    )
    core.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model name or path (Hugging Face format).",
    )
    core.add_argument(
        "--sft_checkpoint",
        type=int,
        default=1500,
        help="Checkpoint ID for the supervised fine-tuned (SFT) reference policy.",
    )
    core.add_argument(
        "--reward_func",
        default="exp_inf",
        choices=["exp_inf", "log_decay"],
        help=(
            "Reward shaping function:\n"
            " - exp_inf: per-rank independent exponential/indicator reward.\n"
            " - log_decay: rank-weighted logarithmic decay reward."
        ),
    )
    core.add_argument(
        "--sft_model_path",
        default=None,
        help="Explicit path to the SFT checkpoint. Defaults to ./results/{model_name}/checkpoint-{sft_checkpoint}.",
    )
    core.add_argument(
        "--catalog_path",
        default="reddit_catalog/gt_catalog.pkl",
        help="Path to the ground-truth catalog pickle (used for hit/match validation).",
    )

    opt = parser.add_argument_group("Optimization hyperparameters")
    opt.add_argument("--lr", type=float, default=1e-6, help="Initial learning rate for policy update.")
    opt.add_argument("--adam_beta1", type=float, default=0.9, help="Adam optimizer β₁ parameter.")
    opt.add_argument("--adam_beta2", type=float, default=0.99, help="Adam optimizer β₂ parameter.")
    opt.add_argument("--kl_beta", type=float, default=1e-3, help="KL-divergence penalty coefficient.")
    opt.add_argument("--gradient_accumulation_steps", type=int, default=12, help="Gradient accumulation steps.")
    opt.add_argument("--optim", default="paged_adamw_8bit", help="Optimizer type (paged_adamw_8bit recommended).")
    opt.add_argument("--opsd_coef", type=float, default=0.0,
                     help="Rank-OPSD: weight of the privileged-hint self-distillation advantage (0 = plain Rank-GRPO).")
    opt.add_argument("--opsd_scope", default="dead", choices=["dead", "all"],
                     help="Rank-OPSD: add the distillation advantage only on dead (group, rank) cells or everywhere.")

    opt.add_argument("--cf_coef", type=float, default=0.0,
                     help="CF partial credit: weight of the centred EASE item score added to the advantage (0 = off).")
    opt.add_argument("--cf_scope", default="all", choices=["all", "hit"],
                     help="CF partial credit on every (group, rank) cell or only on cells with at least one hit.")
    opt.add_argument("--cf_path", default="cf/ease_l1000.npz", help="EASE matrix from scripts/build_cf_ease.py.")
    opt.add_argument("--luffy_gold", type=int, default=0,
                     help="LUFFY gold guidance: max gold items written into one list per group (0 = off; exp_inf only).")
    opt.add_argument("--luffy_scope", default="dead", choices=["dead", "all"],
                     help="LUFFY: guide only groups where no list hits at any rank, or every group with a dead rank.")
    opt.add_argument("--luffy_gamma", type=float, default=0.1, help="LUFFY policy shaping f(p) = p / (p + gamma).")

    sched = parser.add_argument_group("Training schedule")
    sched.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size per GPU.")
    sched.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size per GPU.")
    sched.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    sched.add_argument("--mu", type=int, default=1, help="Number of GRPO iterations (μ=1 = strictly on-policy).")
    sched.add_argument("--max_steps", type=int, default=-1, help="Stop after N optimizer steps (-1 = use num_train_epochs).")
    sched.add_argument("--num_train_samples", type=int, default=None, help="Use only the first N training rows (for smoke tests).")

    gen = parser.add_argument_group("Model / generation parameters")
    gen.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt token length.")
    gen.add_argument("--max_completion_length", type=int, default=1024, help="Maximum generation token length.")
    gen.add_argument("--num_generations", type=int, default=8, help="Number of sampled completions per prompt.")
    gen.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")

    vllm = parser.add_argument_group("vLLM inference backend")
    vllm.add_argument("--use_vllm", action="store_true", help="Use vLLM as the inference backend.")
    vllm.add_argument("--vllm_mode", default="colocate", help="vLLM deployment mode: colocate or separate.")
    vllm.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.5,
        help="Fraction of GPU memory allocated to vLLM inference.",
    )

    ### For Qwen2.5-0.5B, please modify this to 2
    vllm.add_argument("--vllm_tensor_parallel_size", type=int, default=4, help="vLLM tensor parallel size.")

    log = parser.add_argument_group("Logging / checkpointing")
    log.add_argument("--output_dir", default=None, help="Output directory. Defaults to ./results/grpo/{model_name}_lr{lr}_kl{kl_beta}_mu{mu}.")
    log.add_argument("--wandb_project", default="rank_grpo", help="Weights & Biases project name.")
    log.add_argument("--logging_steps", type=int, default=10, help="Logging frequency in steps.")
    log.add_argument("--save_strategy", default="steps", help="Checkpoint save strategy: steps or epoch.")
    log.add_argument("--save_steps", type=int, default=200, help="Save model every N steps.")
    log.add_argument("--save_total_limit", type=int, default=None, help="Keep at most N checkpoints.")
    log.add_argument("--save_only_model", action="store_true", help="Save model weights only (no optimizer state, cannot resume).")
    log.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint.")

    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility.")
    misc.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision.")
    misc.add_argument("--fp16", action="store_true", help="Enable float16 mixed precision (GPUs without bf16, e.g. T4).")
    misc.add_argument("--verbose", action="store_true", help="Enable verbose LR scheduler output.")

    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = setup_environment(args.wandb_project)

    # Load datasets and catalog
    gt_catalog = load_catalog(args.catalog_path)
    train_dataset = load_from_disk(os.path.join(args.train_path, "train"))
    if args.num_train_samples is not None:
        train_dataset = train_dataset.select(range(min(args.num_train_samples, len(train_dataset))))

    # Reward function selection
    if args.reward_func == "exp_inf":
        reward_func = make_reward_func_individual(rec_num=20, gt_catalog=gt_catalog)
    elif args.reward_func == "log_decay":
        reward_func = make_reward_func(rec_num=20, gt_catalog=gt_catalog)
    else:
        raise ValueError(f"{args.reward_func} not implemented!")

    # Define model paths and W&B run
    sft_model_path = args.sft_model_path or f"./results/{args.model_name}/checkpoint-{args.sft_checkpoint}"
    output_dir = args.output_dir or f"./results/grpo/{args.model_name}_lr{args.lr}_kl{args.kl_beta}_mu{args.mu}"
    run_name = setup_wandb(accelerator, output_dir, args.model_name, args.sft_checkpoint, args.seed, args.wandb_project)

    # ---------------- Learning rate scheduler ----------------
    # According to the paper setup: constant LR for small model, decay for larger models.
    if "0.5B" in args.model_name:
        schedule = [(8000, 1e-6)]  # constant for Qwen2.5-0.5B
    else:
        schedule = [(8000, 1e-7)]  # decay for Llama-3.2-1B/3B
    callback = StepLRSchedulerCallback(schedule=schedule, verbose=args.verbose)

    # ---------------- GRPO Configuration ----------------
    config = GRPOConfig(
        importance_sampling_level="item",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        beta=args.kl_beta,
        epsilon=0.06,
        epsilon_high=0.08,
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        num_iterations=args.mu,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        optim=args.optim,
        lr_scheduler_type="constant",
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        gradient_checkpointing=args.gradient_checkpointing,
        run_name=run_name,
    )

    # ---------------- Trainer ----------------
    trainer = RankGRPOTrainer(
        model=sft_model_path,
        reward_funcs=reward_func,
        args=config,
        train_dataset=train_dataset,
        callbacks=[callback], 
    )
    trainer.opsd_coef, trainer.opsd_scope = args.opsd_coef, args.opsd_scope
    if args.cf_coef:
        from libs.cf_reward import CFScorer
        trainer.cf_coef, trainer.cf_scope = args.cf_coef, args.cf_scope
        trainer.cf_scorer = CFScorer(args.cf_path, gt_catalog)
        accelerator.print(f"CF partial credit on: coef {args.cf_coef}, scope {args.cf_scope}, {args.cf_path}")
    if args.luffy_gold:
        if args.reward_func != "exp_inf":
            raise ValueError("LUFFY gold guidance reads 0/1 hits per rank, so it needs --reward_func exp_inf")
        trainer.luffy_gold, trainer.luffy_scope, trainer.luffy_gamma = args.luffy_gold, args.luffy_scope, args.luffy_gamma
        accelerator.print(f"LUFFY gold guidance on: {args.luffy_gold} gold/list, scope {args.luffy_scope}, "
                          f"gamma {args.luffy_gamma}")
    if args.opsd_coef:
        accelerator.print(f"Rank-OPSD on: coef {args.opsd_coef}, scope {args.opsd_scope}")

    accelerator.print("🚀 Training …")
    trainer.train(resume_from_checkpoint=args.resume)
    accelerator.print("✅ Done")


if __name__ == "__main__":
    main()