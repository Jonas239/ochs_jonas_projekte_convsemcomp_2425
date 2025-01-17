import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from vizdoom import gymnasium_wrapper, ScreenResolution, ScreenFormat
import os

def check_cuda():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS):", device)
    elif torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    return device

device = check_cuda()

def initialize_ppo_model(env, device):
    model = PPO(
        "MultiInputPolicy", 
        env, 
        verbose=1,
        batch_size=512,
        n_epochs=40, 
        device=device,
        learning_rate=1e-5, 
        n_steps=1024, 
        gamma=0.96,
        gae_lambda=0.95,
        ent_coef=0.2, 
        vf_coef=0.75,
        clip_range=0.2,
        max_grad_norm=0.5,
    )
    print("Initialized PPO model.")
    return model

def train_single_env(env_name, total_timesteps, difficulty, model_dir="models/ppo", model_filename="hellbot.zip"):
    
    os.makedirs(model_dir, exist_ok=True)

    def make_env():
        env = gym.make(env_name)
        env.unwrapped.game.set_window_visible(False)
        env.unwrapped.game.set_render_hud(False)
        env.unwrapped.game.set_render_crosshair(False)
        env.unwrapped.game.set_render_decals(False)
        env.unwrapped.game.set_render_particles(False)
        env.unwrapped.game.set_render_corpses(False)
        env.unwrapped.game.set_screen_resolution(ScreenResolution.RES_320X240)
        env.unwrapped.game.set_screen_format(ScreenFormat.RGB24)
        env.unwrapped.game.set_sound_enabled(False)
        env.unwrapped.game.set_ticrate(20)
        env.unwrapped.game.set_episode_timeout(2000)
        env.unwrapped.game.set_doom_skill(difficulty) 

        return env

    vec_env = SubprocVecEnv([make_env for _ in range(12)])
    vec_env = VecTransposeImage(vec_env)

    model_path = os.path.join(model_dir, model_filename)

    if os.path.exists(model_path):
        print(f"Loading existing model from: {model_path}")
        model = PPO.load(model_path, env=vec_env)
    else:
        print("No existing model found. Initializing a new model.")
        model = initialize_ppo_model(vec_env, device)

    try:
        print(f"Starting training on {env_name} with difficulty {difficulty}...")
        model.learn(total_timesteps=total_timesteps)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving the model...")
    finally:
        model.save(model_path)
        print(f"Model saved at: {model_path}")
       

if __name__ == "__main__":
    env_name = "VizdoomDefendCenter-v0"
    total_timesteps = 200_000
    model_dir = "models/ppo"
    model_filename = "hellbot.zip"

    train_single_env(
            env_name=env_name,
            total_timesteps=total_timesteps,
            difficulty=0,
            model_dir=model_dir,
            model_filename=model_filename
        )
