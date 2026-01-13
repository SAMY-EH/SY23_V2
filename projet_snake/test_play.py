import os
import glob
from stable_baselines3 import PPO
from envs.snake_env import SnakeEnv
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
def lister_modeles(dossier_racine=os.path.join(script_dir, "checkpoints")):
    """Récupère tous les fichiers .zip dans les sous-dossiers"""
    liste_fichiers = []
    # On cherche récursivement dans tous les dossiers
    for root, dirs, files in os.walk(dossier_racine):
        for file in files:
            if file.endswith(".zip"):
                path_complet = os.path.join(root, file)
                liste_fichiers.append(path_complet)
    
    # On trie par ordre alphabétique (souvent chronologique si les noms sont bien faits)
    # On peut inverser pour avoir les plus récents en premier si on veut
    liste_fichiers.sort()
    return liste_fichiers

def main():
    print("--- SÉLECTION DU MODÈLE ---")
    
    # 1. Trouver les modèles
    fichiers = lister_modeles()
    
    if not fichiers:
        print("Aucun modèle trouvé dans le dossier 'checkpoints' !")
        print("Avez-vous lancé l'entraînement ?")
        return

    # 2. Afficher le menu
    for i, f in enumerate(fichiers):
        # On affiche juste le nom du fichier pour que ce soit lisible, pas tout le chemin
        nom_lisible = f.replace("checkpoints/", "") 
        print(f"[{i}] {nom_lisible}")
    
    # 3. Choix de l'utilisateur
    choix = input(f"\nEntrez le numéro du modèle (0-{len(fichiers)-1}) : ")
    
    try:
        index = int(choix)
        model_path = fichiers[index]
    except (ValueError, IndexError):
        print("Choix invalide. Chargement du dernier modèle par défaut.")
        model_path = fichiers[-1]

    print(f"\nChargement de : {model_path}")
    print("--------------------------------")

    # 4. Lancement du jeu
    # On active le mode "human" pour voir le jeu
    env = SnakeEnv(render_mode="human")
    
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Erreur lors du chargement : {e}")
        return

    obs, _ = env.reset()
    score_actuel = 0
    
    print("Jeu lancé ! (Ctrl+C pour quitter)")
    
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Ralentir un tout petit peu pour que ce soit agréable à regarder
        # (Sinon l'IA joue trop vite pour l'oeil humain)
        time.sleep(0.05) 
        
        if reward > 0: # Si on mange
            score_actuel += 1
            
        if done:
            print(f"Fin de partie. Score : {score_actuel}")
            obs, _ = env.reset()
            score_actuel = 0
            time.sleep(1) # Pause avant de rejouer

if __name__ == "__main__":
    main()