import os
import glob
import time
from stable_baselines3 import PPO
from envs.snake_env_cnn import SnakeEnvCnn

# IMPORTANT : On doit redéfinir la classe CustomCNN ici aussi
# pour que Python puisse charger le modèle sauvegardé.
# (Dans un vrai projet, on mettrait cette classe dans un fichier 'utils.py' importé partout)
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

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
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

# --- Le reste est standard ---

def main():
    # 1. Trouver le dernier modèle CNN
    list_of_files = glob.glob('SY23_V2/projet_snake/checkpoints/PPO_CNN/*.zip')
    if not list_of_files:
        print("Pas de modèle CNN trouvé !")
        return
    # Prend le fichier le plus récent
    latest_model = max(list_of_files, key=os.path.getctime)
    print(f"Chargement de : {latest_model}")

    # 2. Créer l'environnement CNN
    env = SnakeEnvCnn(render_mode="human")

    # 3. Charger le modèle
    # Pas besoin de passer policy_kwargs ici, SB3 le retrouve dans le fichier zip
    model = PPO.load(latest_model, env=env)

    obs, _ = env.reset()
    
    print("Début de la démo CNN...")
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Ralentir un peu pour admirer
        # Si vous voulez voir la vitesse réelle de l'IA, commentez cette ligne
        time.sleep(0.05)
        
        if done:
            print("Perdu ! Reset.")
            time.sleep(1)
            obs, _ = env.reset()

if __name__ == "__main__":
    main()