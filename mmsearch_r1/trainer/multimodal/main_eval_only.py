"""
Lightweight evaluation-only entrypoint for MMSearch-R1.

Compared with `main_ppo.py`, this script:
1) skips the training loop entirely;
2) builds only rollout workers by default (no critic / ref policy);
3) runs one validation pass and exits.
"""

import json
import os
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf, open_dict
from verl.utils.fs import copy_to_local
from verl.utils import hf_processor, hf_tokenizer

from mmsearch_r1.trainer.multimodal.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_eval_only(config)


def _prepare_eval_only_config(config):
    """Force a minimal setup suitable for evaluation-only runs."""
    OmegaConf.resolve(config)
    with open_dict(config):
        # Ensure eval-only behavior regardless of inherited defaults.
        config.trainer.val_only = True
        config.trainer.val_before_train = True
        config.trainer.test_freq = -1
        config.trainer.save_freq = -1
        config.trainer.total_epochs = 1

        # Keep eval logs local/console by default.
        if "logger" not in config.trainer or config.trainer.logger is None:
            config.trainer.logger = ["console"]

        # RayPPOTrainer still builds train dataloader in __init__.
        # Reuse val file to avoid requiring a separate train file for eval-only use.
        if "train_files" not in config.data or config.data.train_files is None:
            config.data.train_files = config.data.val_files

        # Disable critic path in trainer internals for a lighter worker graph.
        # (GAE requires critic; GRPO does not.)
        config.algorithm.adv_estimator = "grpo"


def run_eval_only(config, compute_score=None):
    _prepare_eval_only_config(config)

    if not ray.is_initialized():
        env_vars = {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
        passthrough_env_keys = [
            "SERPER_API_KEY",
            "OPENROUTER_API_KEY",
            "QWEN_API_KEY",
            "DASHSCOPE_API_KEY",
            "QWEN_API_BASE",
            "QWEN_SUMMARY_MODEL",
            "OPENROUTER_HTTP_REFERER",
            "OPENROUTER_X_TITLE",
            "MMSEARCH_TEXT_SEARCH_TOP_K",
            "MMSEARCH_IMAGE_SEARCH_TOP_K",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        ]
        # Prevent Ray from overwriting CUDA_VISIBLE_DEVICES (see main_ppo.py)
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        for key in passthrough_env_keys:
            value = os.getenv(key)
            if value:
                env_vars[key] = value

        ray.init(include_dashboard=False, runtime_env={"env_vars": env_vars})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)
def main_task(config, compute_score=None):
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)

    if config.actor_rollout_ref.actor.strategy == "fsdp":
        from verl.single_controller.ray import RayWorkerGroup
        from mmsearch_r1.workers.multimodal.fsdp_workers import ActorRolloutRefWorker

        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError

        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == "naive":
        from mmsearch_r1.workers.multimodal.reward import MMSearchR1_RewardManager

        reward_manager_cls = MMSearchR1_RewardManager
    elif reward_manager_name == "prime":
        from verl.workers.reward_manager import PrimeRewardManager

        reward_manager_cls = PrimeRewardManager
    else:
        raise NotImplementedError

    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score)
    val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()

    # Keep behavior consistent with training entrypoint: load latest checkpoint when enabled.
    trainer.global_steps = 0
    trainer._load_checkpoint()
    val_metrics = trainer._validate()

    pprint({"eval_only_metrics": val_metrics})
    metrics_save_path = config.trainer.get("val_metrics_save_path", None)
    if metrics_save_path:
        metrics_dir = os.path.dirname(metrics_save_path)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_save_path, "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, ensure_ascii=False, indent=2)
        print(f"Validation metrics saved to: {metrics_save_path}")


if __name__ == "__main__":
    main()
