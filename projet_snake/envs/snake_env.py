import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

# Constantes du jeu
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
BLOCK_SIZE = 20  # Taille d'une case (le serpent fait 20x20 pixels)
SPEED = 20       # Vitesse de rendu pour l'humain (pas pour l'IA)

# Couleurs modernes
WHITE = (255, 255, 255)
BLACK = (15, 15, 25)
DARK_GRAY = (30, 30, 40)
RED = (255, 80, 80)
ORANGE = (255, 165, 0)
GREEN = (76, 175, 80)
BLUE1 = (66, 165, 245)
BLUE2 = (33, 150, 243)
CYAN = (0, 188, 212)
YELLOW = (255, 235, 59)

class SnakeEnv(gym.Env):
    """
    Environnement Custom pour Snake compatible avec OpenAI Gym / Stable Baselines 3
    """
    metadata = {'render_modes': ['human'], 'render_fps': SPEED}

    def __init__(self, render_mode=None):
        super(SnakeEnv, self).__init__()
        
        # Dimensions de la grille
        self.w = WINDOW_WIDTH
        self.h = WINDOW_HEIGHT
        self.render_mode = render_mode
        
        # Init Pygame seulement si nécessaire
        self.window = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        # ACTION SPACE : 
        # 0: Gauche, 1: Droite, 2: Haut, 3: Bas
        self.action_space = spaces.Discrete(4)
        
        # OBSERVATION SPACE : Vecteur de 11 valeurs (0 ou 1)
        # [Danger Tout Droit, Danger Droite, Danger Gauche, 
        #  Direction G, Direction D, Direction H, Direction B,
        #  Pomme G, Pomme D, Pomme H, Pomme B]
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # État initial du serpent (au centre)
        self.direction = 1 # Commence vers la Droite
        self.head = [self.w/2, self.h/2]
        self.snake = [self.head, 
                      [self.head[0]-BLOCK_SIZE, self.head[1]],
                      [self.head[0]-(2*BLOCK_SIZE), self.head[1]]]
        
        self.score = 0
        self.frame_iteration = 0
        self._place_food()
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        self.frame_iteration += 1
        
        # 1. Mouvement
        # Empêcher le retournement immédiat (ex: aller à gauche si on va à droite)
        # C'est optionnel, mais ça aide l'IA au début.
        self._move(action)
        
        # 2. Vérifier État (Game Over ?)
        game_over = False
        reward = 0
        
        # Condition de défaite : Collision mur ou soi-même
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10 # Punition forte
            return self._get_observation(), reward, game_over, False, {}
            
        # 3. Manger la Pomme
        if self.head == self.food:
            self.score += 1
            reward = 10 # Récompense forte
            self._place_food()
        else:
            # Si on ne mange pas, on retire la queue (mouvement normal)
            self.snake.pop()
            
        # 4. Petit Reward Shaping (Optionnel)
        # On peut donner une petite récompense si on se rapproche de la pomme, 
        # mais attention aux boucles infinies. Pour l'instant, restons simples.
        
        # 5. Rendu Visuel (si demandé)
        if self.render_mode == "human":
            self._render_frame()
            
        return self._get_observation(), reward, game_over, False, {}

    def _get_observation(self):
        # C'est ici que l'IA "voit". On construit le vecteur de 11 valeurs.
        
        # Coordonnées des points autour de la tête
        head = self.snake[0]
        point_l = [head[0] - BLOCK_SIZE, head[1]]
        point_r = [head[0] + BLOCK_SIZE, head[1]]
        point_u = [head[0], head[1] - BLOCK_SIZE]
        point_d = [head[0], head[1] + BLOCK_SIZE]
        
        # Directions actuelles (Booléens)
        dir_l = self.direction == 0
        dir_r = self.direction == 1
        dir_u = self.direction == 2
        dir_d = self.direction == 3

        state = [
            # Danger Tout Droit
            (dir_r and self._is_collision(point_r)) or 
            (dir_l and self._is_collision(point_l)) or 
            (dir_u and self._is_collision(point_u)) or 
            (dir_d and self._is_collision(point_d)),

            # Danger à Droite (relativement à la tête)
            (dir_u and self._is_collision(point_r)) or 
            (dir_d and self._is_collision(point_l)) or 
            (dir_l and self._is_collision(point_u)) or 
            (dir_r and self._is_collision(point_d)),

            # Danger à Gauche (relativement à la tête)
            (dir_d and self._is_collision(point_r)) or 
            (dir_u and self._is_collision(point_l)) or 
            (dir_r and self._is_collision(point_u)) or 
            (dir_l and self._is_collision(point_d)),
            
            # Direction actuelle
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Position de la Pomme
            self.food[0] < self.head[0],  # Pomme est à Gauche
            self.food[0] > self.head[0],  # Pomme est à Droite
            self.food[1] < self.head[1],  # Pomme est en Haut
            self.food[1] > self.head[1]   # Pomme est en Bas
        ]
        
        return np.array(state, dtype=np.int8)

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = [x, y]
        # Vérifier que la pomme n'est pas SUR le serpent
        if self.food in self.snake:
            self._place_food()

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Collision Mur
        if pt[0] > self.w - BLOCK_SIZE or pt[0] < 0 or pt[1] > self.h - BLOCK_SIZE or pt[1] < 0:
            return True
        # Collision Soi-même
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        # Mise à jour de la direction (interdire le 180°)
        # 0: Gauche, 1: Droite, 2: Haut, 3: Bas
        clock_wise = [0, 1, 2, 3] # Ordre arbitraire, ici mapping direct
        
        if action == 0 and self.direction != 1: self.direction = 0
        elif action == 1 and self.direction != 0: self.direction = 1
        elif action == 2 and self.direction != 3: self.direction = 2
        elif action == 3 and self.direction != 2: self.direction = 3
            
        # Calcul de la nouvelle tête
        x = self.head[0]
        y = self.head[1]
        
        if self.direction == 1: x += BLOCK_SIZE
        elif self.direction == 0: x -= BLOCK_SIZE
        elif self.direction == 3: y += BLOCK_SIZE
        elif self.direction == 2: y -= BLOCK_SIZE
            
        self.head = [x, y]
        self.snake.insert(0, self.head) # On ajoute la tête au début

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.w, self.h))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 48)
            self.small_font = pygame.font.Font(None, 32)
            pygame.display.set_caption("🐍 Snake AI Training 🐍")
        
        # Fond
        pygame.event.pump()
        self.window.fill(BLACK)
        
        # Grille légère en arrière-plan (optionnel, pour voir les cases)
        grid_color = (50, 50, 70)
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.window, grid_color, (x, 0), (x, self.h), 1)
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.window, grid_color, (0, y), (self.w, y), 1)
        
        # Dessiner la Pomme (avec un effet de glow)
        self._draw_apple()
        
        # Dessiner le Serpent avec dégradé de couleur
        self._draw_snake()
        
        # Afficher le Score
        self._draw_score()
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_apple(self):
        """Dessiner la pomme avec un effet visuel amélioré"""
        x, y = int(self.food[0]), int(self.food[1])
        
        # Lueur autour de la pomme
        glow_radius = BLOCK_SIZE // 2 + 3
        pygame.draw.circle(self.window, (255, 100, 0, 50), (x + BLOCK_SIZE//2, y + BLOCK_SIZE//2), glow_radius)
        
        # Pomme principale (dégradé simulé)
        pygame.draw.rect(self.window, RED, pygame.Rect(x+2, y+2, BLOCK_SIZE-4, BLOCK_SIZE-4), border_radius=4)
        pygame.draw.rect(self.window, ORANGE, pygame.Rect(x+3, y+3, BLOCK_SIZE-6, BLOCK_SIZE-6), border_radius=3)
        
        # Brillance
        pygame.draw.circle(self.window, YELLOW, (x + 7, y + 7), 3)

    def _draw_snake(self):
        """Dessiner le serpent avec dégradé de couleur"""
        snake_length = len(self.snake)
        
        for i, pt in enumerate(self.snake):
            x, y = int(pt[0]), int(pt[1])
            
            # Couleur dégradée : cyan pour la tête, bleu pour la queue
            ratio = i / max(snake_length - 1, 1)
            color = (
                int(BLUE1[0] + (CYAN[0] - BLUE1[0]) * (1 - ratio)),
                int(BLUE1[1] + (CYAN[1] - BLUE1[1]) * (1 - ratio)),
                int(BLUE1[2] + (CYAN[2] - BLUE1[2]) * (1 - ratio))
            )
            
            # Corps du serpent (arrondi pour plus joli)
            pygame.draw.rect(self.window, color, pygame.Rect(x+1, y+1, BLOCK_SIZE-2, BLOCK_SIZE-2), border_radius=3)
            
            # Tête du serpent (plus grande et brillante)
            if i == 0:
                pygame.draw.rect(self.window, CYAN, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE), border_radius=4)
                pygame.draw.circle(self.window, WHITE, (x + 6, y + 6), 2)
                pygame.draw.circle(self.window, WHITE, (x + 14, y + 6), 2)

    def _draw_score(self):
        """Afficher le score et les informations en haut à droite"""
        score_text = self.font.render(f"Score: {self.score}", True, GREEN)
        length_text = self.small_font.render(f"Length: {len(self.snake)}", True, CYAN)
        frame_text = self.small_font.render(f"Frame: {self.frame_iteration}", True, WHITE)
        
        # Position en haut à droite
        panel_width = 180
        panel_x = self.w - panel_width - 10
        
        # Fond semi-transparent pour la lisibilité
        pygame.draw.rect(self.window, DARK_GRAY, (panel_x, 5, panel_width, 90), border_radius=5)
        pygame.draw.rect(self.window, CYAN, (panel_x, 5, panel_width, 90), 2, border_radius=5)
        
        self.window.blit(score_text, (panel_x + 10, 10))
        self.window.blit(length_text, (panel_x + 10, 45))
        self.window.blit(frame_text, (panel_x + 10, 70))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()