import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ailive_envs.walking import HumanoidEnv
from ailive_envs.standingup import HumanoidStandupEnv
from ailive_envs.crawling import AiliveHumanoidCrawlingEnv
import os
import json
import numpy as np
import random

# Configuration
AGENT_NAME = "zero"
SKILL_NAME = "walking"
SESSION_ID = 0

BASE_PATH = f"./public/sessions/{AGENT_NAME}/{SKILL_NAME}"
TENSORBOARD_PATH = os.path.join(BASE_PATH, "tensorboard")
MODELS_PATH = os.path.join(BASE_PATH, "models")
OBS_PATH = os.path.join(BASE_PATH, "obs")
SAVE_INTERVAL = 5_000_000  # Save every 5 million steps
TOTAL_TIMESTEPS = 100_000_000  # Total training timesteps
MODEL = None
STEPS_TRAINED = 0

def export_obs(saved_steps_trained: int, replay_count: int = 10) -> None:
    """
    Export observations to a JSON file after replaying with saved models.

    Args:
        saved_steps_trained (int): The step count corresponding to the model to use.
        replay_count (int): Number of replays for generating observations.
    """
    model_path = os.path.join(MODELS_PATH, f"{saved_steps_trained}.zip")
    obs_file_path = os.path.join(OBS_PATH, f"{saved_steps_trained}.json")
    obs_values = []

    for i in range(replay_count):
        random_seed = i
        np.random.seed(random_seed)
        random.seed(random_seed)

        env = gym.make("Humanoid-v5", max_episode_steps=5000)
        model = PPO.load(model_path, seed=random_seed)

        observations = []
        obs, _ = env.reset(seed=random_seed)
        done = False

        observations.append(obs[:22].tolist())
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            observations.append(obs[:45].tolist())
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        obs_values.append(observations)
        env.close()

    os.makedirs(os.path.dirname(obs_file_path), exist_ok=True)
    with open(obs_file_path, "w") as file:
        json.dump(obs_values, file)

    print(f"Exported observations ({replay_count} times): {obs_file_path}")

def save_model() -> None:
    """
    Save the current model and export its observations.
    """
    global MODEL, STEPS_TRAINED
    if MODEL:
        os.makedirs(MODELS_PATH, exist_ok=True)
        model_path = os.path.join(MODELS_PATH, f"{STEPS_TRAINED}.zip")
        MODEL.save(model_path)
        print(f"Model saved at {model_path}.")
        export_obs(STEPS_TRAINED)

def getEnv(skill: str) -> classmethod:
    if skill == "walking":
        return HumanoidEnv
    elif skill == "standingup":
        return HumanoidStandupEnv
    elif skill == "crawling":
        return AiliveHumanoidCrawlingEnv
    
def main() -> None:
    """
    Main function to train and save the PPO model for Humanoid-v5.
    """
    global MODEL, STEPS_TRAINED

    # Create the environment
    env = make_vec_env(getEnv(SKILL_NAME))

    # Initialize the PPO model
    print(f"TensorBoard logging at {TENSORBOARD_PATH}")
    MODEL = PPO("MlpPolicy", env, verbose=0, tensorboard_log=TENSORBOARD_PATH)

    # Load the latest saved model if available
    if os.path.exists(MODELS_PATH):
        saved_models = [f for f in os.listdir(MODELS_PATH) if f.endswith(".zip")]
        if saved_models:
            latest_model = max(saved_models, key=lambda f: int(f.split(".")[0]))
            model_path = os.path.join(MODELS_PATH, latest_model)
            MODEL.set_parameters(model_path)
            STEPS_TRAINED = int(latest_model.split(".")[0])
            print(f"Loaded model from {model_path}. Resuming at step {STEPS_TRAINED}.")
    else:
        print("No saved models found. Starting fresh training.")

    # Train the model
    print(f"Starting training for a total of {TOTAL_TIMESTEPS} timesteps.")
    while STEPS_TRAINED < TOTAL_TIMESTEPS:
        MODEL.learn(total_timesteps=SAVE_INTERVAL, reset_num_timesteps=True,
                    tb_log_name=f"{AGENT_NAME}_{SKILL_NAME}_{STEPS_TRAINED}")
        STEPS_TRAINED += SAVE_INTERVAL
        save_model()

    env.close()
    print("Training complete!")


if __name__ == "__main__":
    main()
