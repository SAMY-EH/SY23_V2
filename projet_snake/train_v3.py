import os
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env # <--- Pour la vectorisation
from stable_baselines3.common.vec_env import SubprocVecEnv # <--- Vrai parallélisme
from envs.snake_env_cnn import SnakeEnvCnn

# --- 1. Définition du Cerveau Custom (Inchangé) ---
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# --- CONFIGURATION MULTI-CPU ---
# Sur un M2, vous avez 8 coeurs. En garder un ou deux pour le système est bien.
# 4 à 8 environnements est idéal. Au-delà, le gain diminue.
N_ENVS = 8  
TIMESTEPS = 5000000 
SAVE_FREQ = 200000 

# --- BLOCK MAIN OBLIGATOIRE SUR MAC ---
if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(script_dir, "checkpoints/PPO_CNN_MULTI")
    LOG_DIR = os.path.join(script_dir, "logs")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"--- Démarrage sur {N_ENVS} environnements en parallèle ---")

    # 2. Création de l'environnement vectorisé
    # Cela va lancer N_ENVS processus Python indépendants
    env = make_vec_env(
        SnakeEnvCnn, 
        n_envs=N_ENVS, 
        vec_env_cls=SubprocVecEnv # Utilise plusieurs coeurs CPU
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1), # Ajustement de la fréquence car ça va plus vite
        save_path=MODELS_DIR,
        name_prefix="snake_cnn"
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # 3. Le Modèle
    # device="auto" va essayer d'utiliser le GPU (MPS) ou le CPU.
    # Souvent sur les petits CNN, le CPU est aussi rapide, mais le M2 gère bien le MPS.
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=LOG_DIR,
        learning_rate=0.0003,
        policy_kwargs=policy_kwargs,
        batch_size=256, # On augmente le batch car on a plus de données
        n_steps=1024,   # Nombre de pas PAR environnement avant update (1024 * 8 données totales)
        gamma=0.99,
        device="auto"   # Laisse SB3 choisir (souvent CPU sur Mac pour RL, ce qui est OK)
    )

    print("Entraînement lancé... (Regardez le Moniteur d'activité, vos coeurs vont chauffer !)")
    
    model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
    
    model.save(f"{MODELS_DIR}/snake_cnn_final")
    print("Terminé.")
    env.close() # Important de fermer les processus