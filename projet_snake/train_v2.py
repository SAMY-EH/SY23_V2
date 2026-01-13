import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.snake_env import SnakeEnv

# --- CONFIGURATION v2 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(script_dir, "checkpoints/PPO_v2")
LOG_DIR = os.path.join(script_dir, "logs")
TIMESTEPS = 2000000  # 2 Millions de pas (environ 20-30 min sur Mac M1/M2)
SAVE_FREQ = 100000   # Sauvegarder une copie du cerveau tous les 100k pas

# Création des dossiers
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 1. L'environnement (Toujours sans rendu pour la vitesse)
env = SnakeEnv()

# 2. Le Callback (Votre demande de "Logging")
# Cela va créer des fichiers : PPO_v2/rl_model_100000_steps.zip, rl_model_200000_steps.zip, etc.
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODELS_DIR,
    name_prefix="snake_v2"
)

# 3. Architecture du Réseau de Neurones (Custom)
# Par défaut c'est [64, 64]. On passe à [128, 128] pour plus d'intelligence.
policy_kwargs = dict(net_arch=[128, 128])

# 4. Le Modèle v2 (Avec des hyperparamètres tunés)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    tensorboard_log=LOG_DIR,
    learning_rate=0.0003,    # Vitesse d'apprentissage standard
    gamma=0.99,              # Importance du futur (0.99 = long terme)
    policy_kwargs=policy_kwargs # Notre plus gros cerveau
)

print("-----------------------------------------")
print(f"Lancement de l'entraînement v2 ({TIMESTEPS} pas)...")
print(f"Checkpoints sauvegardés tous les {SAVE_FREQ} pas dans {MODELS_DIR}")
print("Pour suivre les courbes en direct, lancez TensorBoard dans un autre terminal.")
print("-----------------------------------------")

# 5. On lance l'entraînement AVEC le callback
model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)

print("Entraînement v2 terminé !")