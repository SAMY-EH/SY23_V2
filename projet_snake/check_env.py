from envs.snake_env import SnakeEnv
import random

# Initialiser l'environnement
env = SnakeEnv(render_mode="human")
obs, _ = env.reset()

for _ in range(50):
    # Action aléatoire (0 à 3)
    random_action = env.action_space.sample() 
    obs, reward, done, truncated, info = env.step(random_action)
    
    if done:
        obs, _ = env.reset()

env.close()