import os
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.snake_env_cnn import SnakeEnvCnn

# --- 1. Définition du Cerveau Custom (Adapté au 30x30) ---
class CustomCNN(BaseFeaturesExtractor):
    """
    CNN personnalisé pour traiter des grilles 30x30.
    L'architecture par défaut (NatureCNN) réduit trop la taille,
    ce qui cause le crash sur les petites images.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] # Devrait être 1

        self.cnn = nn.Sequential(
            # Layer 1 : On garde les détails (Kernel 4, Stride 1)
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            # Layer 2 : On continue d'analyser sans trop réduire
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            # Aplatissement pour passer aux décisions
            nn.Flatten(),
        )

        # Calcul automatique de la taille de sortie pour connecter la suite
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# --- 2. Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "checkpoints/PPO_CNN")
LOG_DIR = os.path.join(script_dir, "logs")
TIMESTEPS = 3000000 
SAVE_FREQ = 200000

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- 3. Création Environnement & Modèle ---
env = SnakeEnvCnn()

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODELS_DIR,
    name_prefix="snake_cnn"
)

# On indique à PPO d'utiliser notre CustomCNN
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=LOG_DIR,
    learning_rate=0.0003,
    policy_kwargs=policy_kwargs, # <--- C'est ici qu'on injecte le fix
    batch_size=128,    # Ajusté pour Mac M1/M2
    gamma=0.99
)

print("-----------------------------------------")
print("Lancement de l'entraînement V3 (Custom CNN)...")
print("Architecture adaptée aux grilles 30x30.")
print("-----------------------------------------")

model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

model.save(f"{MODELS_DIR}/snake_cnn_final")
print("Terminé.")