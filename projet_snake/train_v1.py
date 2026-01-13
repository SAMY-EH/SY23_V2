import os
from stable_baselines3 import PPO
from envs.snake_env import SnakeEnv

# 1. Création des dossiers pour sauvegarder les modèles et les logs
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "checkpoints/PPO")
log_dir = os.path.join(script_dir, "logs")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 2. Création de l'environnement
# IMPORTANT : On ne met PAS 'render_mode="human"' ici pour que l'entraînement soit ultra-rapide.
env = SnakeEnv() 

# 3. Initialisation du Modèle (L'IA)
# MlpPolicy : Petit réseau de neurones standard (parfait pour nos 11 chiffres)
# verbose=1 : Pour voir les infos de progression dans le terminal
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

print("-----------------------------------------")
print("Démarrage de l'entraînement...")
print("L'IA va jouer environ 100 000 coups.")
print("-----------------------------------------")

# 4. Lancement de l'apprentissage
# TIMESTEPS est le nombre de 'pas' (frames) que l'IA va jouer au total.
TIMESTEPS = 100000 
model.learn(total_timesteps=TIMESTEPS)

# 5. Sauvegarde du modèle final
model.save(f"{models_dir}/snake_final")

print("-----------------------------------------")
print("Entraînement terminé !")
print(f"Modèle sauvegardé dans : {models_dir}/snake_final.zip")
print("Vous pouvez maintenant lancer 'test_play.py' pour voir le résultat.")
print("-----------------------------------------")