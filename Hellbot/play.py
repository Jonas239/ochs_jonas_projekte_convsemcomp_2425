from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from vizdoom import gymnasium_wrapper, ScreenResolution, ScreenFormat
import os

class ActionPaddedEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def action(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return min(action, self.action_space.n - 1)
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            return [min(a, max_val) for a, max_val in zip(action, self.action_space.nvec)]
        return action

def playtest_model(model_path, env_name, max_episodes=5):
    """
    Playtest a trained PPO model in the specified VizDoom environment.

    :param model_path: Path to the trained model file.
    :param env_name: Name of the VizDoom environment.
    :param max_episodes: Number of episodes to playtest.
    """
    def make_env():
        env = gym.make(env_name, render_mode='human')
        env.unwrapped.game.set_render_hud(False)
        env.unwrapped.game.set_render_crosshair(False)
        env.unwrapped.game.set_render_decals(False)
        env.unwrapped.game.set_render_particles(False)
        env.unwrapped.game.set_render_corpses(False)
        env.unwrapped.game.set_screen_resolution(ScreenResolution.RES_800X600)
        env.unwrapped.game.set_screen_format(ScreenFormat.RGB24)
        env.unwrapped.game.set_ticrate(20)
        env.unwrapped.game.set_episode_timeout(1000)
        env.unwrapped.game.set_doom_skill(0)
        return ActionPaddedEnv(env)

    env = DummyVecEnv([make_env])

    model = PPO.load(model_path)

    total_rewards = []
    for episode in range(max_episodes):
        obs = env.reset()
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {episode + 1} Reward: {total_reward}")
        total_rewards.append(total_reward)

    print(f"Average Reward: {sum(total_rewards) / len(total_rewards)}")
    env.close()

# Run playtest
model_path = os.path.join(os.path.dirname(__file__), "models/ppo/hellbot.zip")
playtest_model(model_path, "VizdoomDefendCenter-v0", max_episodes=10)
