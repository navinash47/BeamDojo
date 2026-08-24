"""RSL-RL PPO runner cfg for BeamDojo (paper MLP [512, 216, 128], double critic)."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BeamDojoPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10_000
    save_interval = 100
    experiment_name = "beamdojo_stage1"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    empirical_normalization = False
    logger = "tensorboard"
    wandb_project = "beamdojo"
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticDouble",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 216, 128],
        critic_hidden_dims=[512, 216, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPODoubleCritic",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.02,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BeamDojoStage2PPORunnerCfg(BeamDojoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "beamdojo_stage2"


@configclass
class BeamDojoG1PPORunnerCfg(BeamDojoPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "beamdojo_g1_stage1"
        self.algorithm.entropy_coef = 0.008
