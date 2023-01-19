SCVI_DEFAULT_MODEL_PARAMS = {
    "use_gpu": None,
    "train_size": 0.9,
    "validation_size": None,
    "batch_size": 128,
    "latent_distribution": "normal",
    "n_hidden": 128,
    "n_latent": 10,
    "n_layers": 1,
    "dropout_rate": 0.1,
    "dispersion": "gene",
    "gene_likelihood": "zinb",
}
SCVI_MODEL_PARAMS = list(SCVI_DEFAULT_MODEL_PARAMS.keys())

SCVI_PLAN_PARAMS = [
    "lr",
    "weight_decay",
    "eps",
    "optimizer",
    "n_steps_kl_warmup",
    "n_epochs_kl_warmup",
    "reduce_lr_on_plateau",
    "lr_factor",
    "lr_patience",
    "lr_threshold",
    "lr_scheduler_metric",
    "lr_min",
]

SCVI_TRAIN_PARAMS = [
    "gpus",
    "benchmark",
    "flush_logs_every_n_steps",
    "check_val_every_n_epoch",
    "max_epochs",
    "default_root_dir",
    "checkpoint_callback",
    "num_sanity_val_steps",
    "weights_summary",
    "early_stopping",
    "early_stopping_monitor",
    "early_stopping_min_delta",
    "early_stopping_patience",
    "early_stopping_mode",
    "progress_bar_refresh_rate",
    "simple_progress_bar",
    "logger",
]
