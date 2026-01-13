# 🐍 Snake AI - Apprentissage par Renforcement

Un projet d'intelligence artificielle qui apprend à jouer au jeu Snake en utilisant le **Reinforcement Learning** avec l'algorithme **PPO** (Proximal Policy Optimization).

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Stable Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-green)
![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0+-orange)
![Pygame](https://img.shields.io/badge/Pygame-2.6+-red)

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
│   ├── snake_env.py           # Env V1 : Observation = vecteur 11 valeurs (MLP)
│   └── snake_env_cnn.py       # Env V2 : Observation = grille 30x30 (CNN)
├── checkpoints/               # Modèles sauvegardés (.zip)
│   ├── PPO/                   # Modèles MLP
│   └── PPO_CNN/               # Modèles CNN
├── logs/                      # Logs TensorBoard
├── train_v1.py               # Entraînement basique MLP (100k steps)
├── train_v2.py               # Entraînement avancé MLP (500k steps)
├── train_v3.py               # Entraînement CNN avec parallélisation
├── train_colab.ipynb         # Notebook pour Google Colab (GPU)
├── test_play.py              # Visualiser l'IA MLP jouer
├── test_play_cnn.py          # Visualiser l'IA CNN jouer
├── check_env.py              # Vérifier l'environnement
├── requirements.txt          # Dépendances Python
└── README.md
```

---

## 🚀 Installation

### 1. Cloner le projet
```bash
git clone https://github.com/SAMY-EH/SY23_V2.git
cd SY23_V2/projet_snake
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
# Entraînement MLP basique (100 000 steps) - ~5 min
python train_v1.py

# Entraînement MLP avancé (500 000 steps) - ~20 min
python train_v2.py

# Entraînement CNN avec parallélisation (plus lent mais plus général)
python train_v3.py
```

### Voir l'IA jouer

```bash
# Version MLP (vecteur 11 valeurs)
python test_play.py

# Version CNN (grille 30x30)
python test_play_cnn.py
```

Un menu s'affiche pour choisir le modèle à charger parmi ceux disponibles dans `checkpoints/`.

### Visualiser les logs d'entraînement

```bash
tensorboard --logdir=logs
```
Puis ouvrir http://localhost:6006 dans un navigateur.

### Entraînement sur Google Colab (GPU)

1. Ouvrir `train_colab.ipynb` sur Google Colab
2. Activer le GPU : `Exécution > Modifier le type d'exécution > T4 GPU`
3. Exécuter les cellules dans l'ordre

---

## 🌍 Environnements

### `SnakeEnv` (snake_env.py) - Version MLP

| Caractéristique | Description |
|-----------------|-------------|
| **Observation** | Vecteur de 11 valeurs binaires |
| **Espace** | `Box(0, 1, shape=(11,), dtype=int8)` |
| **Actions** | 4 : Gauche, Droite, Haut, Bas |

**Détail du vecteur d'observation :**
- 3 valeurs : Danger (tout droit, droite, gauche)
- 4 valeurs : Direction actuelle (G, D, H, B)
- 4 valeurs : Position relative de la pomme (G, D, H, B)

**Récompenses :**
- `+10` : Manger une pomme
- `-10` : Collision (mur ou queue)

---

### `SnakeEnvCnn` (snake_env_cnn.py) - Version CNN

| Caractéristique | Description |
|-----------------|-------------|
| **Observation** | Grille 30x30 en niveaux de gris |
| **Espace** | `Box(0, 255, shape=(1, 30, 30), dtype=uint8)` |
| **Actions** | 4 : Gauche, Droite, Haut, Bas |

**Valeurs de la grille :**
- `0` : Case vide (noir)
- `80` : Corps du serpent (gris foncé)
- `180` : Tête du serpent (gris clair)
- `255` : Pomme (blanc)

**Récompenses avec reward shaping :**
- `+20` : Manger une pomme
- `-10` : Collision (mur ou queue)
- `+1` : Se rapprocher de la pomme
- `-1` : S'éloigner de la pomme

---

## 🧠 Architecture du modèle

### Version MLP (train_v1.py, train_v2.py)

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

### Version CNN (train_v3.py)

```
┌─────────────────────────────────────────────────────┐
│                 Custom CNN Extractor                │
├─────────────────────────────────────────────────────┤
│  Conv2D(1 → 32, kernel=3, stride=2) + ReLU         │
│  Conv2D(32 → 64, kernel=3, stride=2) + ReLU        │
│  Conv2D(64 → 64, kernel=3, stride=1) + ReLU        │
│  Flatten → Linear → 128 features                   │
├─────────────────────────────────────────────────────┤
│  Policy Head: 128 → 4 (actions)                    │
│  Value Head: 128 → 1 (state value)                 │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Résultats

| Version | Environnement | Steps | Score Moyen | Temps |
|---------|---------------|-------|-------------|-------|
| V1 (MLP) | SnakeEnv | 100k | ~5-10 | ~5 min |
| V2 (MLP) | SnakeEnv | 500k | ~15-25 | ~20 min |
| V3 (CNN) | SnakeEnvCnn | 1M+ | ~20-30 | 1h+ (GPU recommandé) |

> Les résultats peuvent varier selon les hyperparamètres et la configuration matérielle.

---

## 🛠️ Technologies utilisées

- **[Gymnasium](https://gymnasium.farama.org/)** : Framework pour environnements RL
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** : Algorithmes RL (PPO)
- **[PyTorch](https://pytorch.org/)** : Backend deep learning
- **[Pygame](https://www.pygame.org/)** : Rendu graphique du jeu
- **[TensorBoard](https://www.tensorflow.org/tensorboard)** : Visualisation des métriques

---

## 📝 Auteurs

**Samy E. et Willen A.** - Projet SY23 - UTC - Janvier 2026

---

## 📜 Licence

Ce projet est réalisé dans le cadre d'un cours universitaire (SY23 - UTC).