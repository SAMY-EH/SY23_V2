# 🐍 Snake AI - Apprentissage par Renforcement

Un projet d'intelligence artificielle qui apprend à jouer au jeu Snake en utilisant le **Reinforcement Learning** avec l'algorithme **PPO** (Proximal Policy Optimization).

![Python](https://img.shields.io/badge/Python-3.14-blue)
![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.7.1-green)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.2.3-orange)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-red)

---

## 📋 Table des matières

- [Description](#-description)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Environnements](#-environnements)
- [Architecture du modèle](#-architecture-du-modèle)
- [Résultats](#-résultats)

---

## 📖 Description

Ce projet implémente un agent d'apprentissage par renforcement capable de jouer au jeu Snake de manière autonome. L'agent apprend à :
- 🍎 Trouver et manger les pommes
- 🧱 Éviter les murs
- 🐍 Ne pas se mordre la queue

L'apprentissage utilise l'algorithme **PPO** de la librairie Stable-Baselines3, reconnu pour sa stabilité et ses bonnes performances.

---

## 📁 Structure du projet

```
projet_snake/
├── envs/                       # Environnements Gymnasium
│   ├── __init__.py
│   ├── snake_env.py           # Env V1 : Observation = vecteur 11 valeurs
│   └── snake_env_cnn.py       # Env V2 : Observation = grille 30x30 (CNN)
├── checkpoints/               # Modèles sauvegardés (.zip)
│   └── PPO/
├── logs/                      # Logs TensorBoard
├── train_v1.py               # Entraînement basique (100k steps)
├── train_v2.py               # Entraînement avancé (2M steps)
├── train_v3.py               # Entraînement CNN
├── test_play.py              # Visualiser l'IA jouer
├── check_env.py              # Vérifier l'environnement
├── requirements.txt          # Dépendances Python
└── README.md
```

---

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/SAMY-EH/SY23P.git
cd SY23P/projet_ultra_secret/projet_snake
```

### 2. Créer un environnement virtuel
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# ou
.venv\Scripts\activate     # Windows
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

> ⚠️ **macOS** : Si pygame ne s'installe pas, installez d'abord SDL2 :
> ```bash
> brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf sdl2_gfx
> ```

---

## 🎮 Utilisation

### Entraîner l'IA

```bash
# Entraînement basique (100 000 steps) - ~5 min
python train_v1.py

# Entraînement avancé (500 000 steps) - ~20 min
python train_v2.py

# Entraînement avec CNN (plus lent mais potentiellement meilleur)
python train_v3.py
```

### Voir l'IA jouer

```bash
python test_play.py
```

Un menu s'affiche pour choisir le modèle à charger parmi ceux disponibles dans `checkpoints/`.

### Visualiser les logs d'entraînement

```bash
tensorboard --logdir=logs
```
Puis ouvrir http://localhost:6006 dans un navigateur.

---

## 🌍 Environnements

### `SnakeEnv` (snake_env.py)
- **Observation** : Vecteur de 11 valeurs binaires
  - 3 valeurs : Danger (tout droit, droite, gauche)
  - 4 valeurs : Direction actuelle (G, D, H, B)
  - 4 valeurs : Position relative de la pomme (G, D, H, B)
- **Actions** : 4 (Gauche, Droite, Haut, Bas)
- **Récompenses** :
  - +10 : Manger une pomme
  - -10 : Collision (mur ou queue)

### `SnakeEnvCnn` (snake_env_cnn.py)
- **Observation** : Image 30x30 en niveaux de gris
  - 0 : Case vide
  - 80 : Corps du serpent
  - 180 : Tête
  - 255 : Pomme
- **Réseau** : CNN (CnnPolicy) pour traiter l'image

---

## 🧠 Architecture du modèle

```
┌─────────────────────────────────────────────────────┐
│                    PPO Agent                        │
├─────────────────────────────────────────────────────┤
│  Policy Network (MlpPolicy)                         │
│  ├── Input Layer (11 neurons)                       │
│  ├── Hidden Layer 1 (64 neurons, ReLU)             │
│  ├── Hidden Layer 2 (64 neurons, ReLU)             │
│  └── Output Layer (4 neurons, Softmax)             │
├─────────────────────────────────────────────────────┤
│  Value Network                                      │
│  ├── Shared layers with Policy                      │
│  └── Output (1 neuron - state value)               │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Résultats

| Version | Steps | Score Moyen | Temps |
|---------|-------|-------------|-------|
| V1 (MLP) | 100k | ~5-10 | 5 min |
| V2 (MLP) | 500k | ~15-25 | 20 min |
| V3 (CNN) | 1M | ~20-30 | 1h+ |

> Les résultats peuvent varier selon les hyperparamètres et la configuration matérielle.

---

## 🛠️ Technologies utilisées

- **[Gymnasium](https://gymnasium.farama.org/)** : Framework pour environnements RL
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** : Algorithmes RL (PPO, DQN, A2C...)
- **[PyTorch](https://pytorch.org/)** : Backend deep learning
- **[Pygame](https://www.pygame.org/)** : Rendu graphique du jeu
- **[TensorBoard](https://www.tensorflow.org/tensorboard)** : Visualisation des métriques

---

## 📝 Auteur

**Samy E et Willen A** - Projet SY23 - Janvier 2026

---

## 📜 Licence

Ce projet est réalisé dans le cadre d'un cours universitaire.
