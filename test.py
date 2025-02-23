from stable_baselines3 import PPO
from simulation.env import RealisticSchedulingEnv
from models.transformer_model import GraphTransformerModel
from utils.graph_utils import build_pyg_data
import torch

# Lade dein Modell
model = PPO.load("./logs/best_model.zip")

# Falls die Umgebung nötig ist, erstelle sie erneut:
transformer_model = GraphTransformerModel(in_channels=3, hidden_channels=16, out_channels=8, heads=2)
pyg_data = build_pyg_data(G, conflict_edges)
env = RealisticSchedulingEnv(transformer_model, pyg_data, data_json["jobs"])

# Testlauf mit dem geladenen Modell
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

print(f"Finale Performance (Makespan): {env.simulation.current_makespan}")