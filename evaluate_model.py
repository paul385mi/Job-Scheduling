
# Interpretation und mögliche Probleme
# 1.	Warum bleibt der Reward konstant?
# 	•	Mögliche Ursache: Das Modell hat sich an eine suboptimale Strategie gewöhnt und exploriert nicht weiter.
# 	•	Lösung: Erhöhe n_steps, gamma oder füge eine explorationsfördernde Mechanik hinzu (z. B. entropy_coef).
# 2.	Warum bleibt der Reward negativ?
# 	•	Mögliche Ursache: Die Belohnungsfunktion ist so gestaltet, dass der Agent nicht wirklich profitiert.
# 	•	Lösung: Falls du Kosten minimierst, könnte es helfen, eine positive Belohnung für gute Entscheidungen hinzuzufügen.
# 3.	Warum gibt es keinen weiteren Anstieg?
# 	•	Mögliche Ursache: Der Optimierungsprozess hat ein Plateau erreicht.
# 	•	Lösung: Versuch alternative Optimierungsstrategien, wie learning_rate-Decay oder eine andere gamma-Wert.


import numpy as np
import matplotlib.pyplot as plt
import os

def load_evaluation_data(file_path):
    """Lädt die evaluations.npz Datei und gibt die gespeicherten Metriken zurück."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Datei {file_path} wurde nicht gefunden.")
    
    eval_data = np.load(file_path)
    print("Verfügbare Keys in der Datei:", eval_data.files)
    return eval_data

def plot_evaluation_metrics(eval_data):
    """Visualisiert die Entwicklung der Rewards über die Trainingsschritte."""
    timesteps = eval_data["timesteps"]
    rewards = eval_data["results"]  # Mean Rewards aus der Evaluierung
    
    # Berechne statistische Werte
    mean_rewards = rewards.mean(axis=1)
    min_rewards = rewards.min(axis=1)
    max_rewards = rewards.max(axis=1)
    
    # Plotten der Reward-Entwicklung
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_rewards, label="Durchschnittlicher Reward", color="blue")
    plt.fill_between(timesteps, min_rewards, max_rewards, color="blue", alpha=0.2, label="Min/Max Reward")
    plt.xlabel("Trainingsschritte")
    plt.ylabel("Reward")
    plt.title("Entwicklung der Reward-Funktion über die Zeit")
    plt.legend()
    plt.grid(True)
    plt.show()

def print_statistics(eval_data):
    """Gibt statistische Kennwerte der Evaluationsmetriken aus."""
    rewards = eval_data["results"]
    mean_rewards = rewards.mean(axis=1)
    
    print("\nStatistische Analyse der Rewards:")
    print(f"Minimaler Reward: {mean_rewards.min():.2f}")
    print(f"Maximaler Reward: {mean_rewards.max():.2f}")
    print(f"Durchschnittlicher Reward: {mean_rewards.mean():.2f}")

def main():
    file_path = "./logs/evaluations.npz"
    eval_data = load_evaluation_data(file_path)
    plot_evaluation_metrics(eval_data)
    print_statistics(eval_data)

if __name__ == "__main__":
    main()