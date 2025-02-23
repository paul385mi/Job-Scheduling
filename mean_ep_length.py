import numpy as np
import matplotlib.pyplot as plt

def plot_evaluation_metrics(npz_path):
    """
    Lädt die Evaluationsdaten aus einer .npz-Datei und visualisiert wichtige Metriken.
    
    Erwartete Schlüssel in der npz-Datei:
      - "timesteps": Array mit den Zeitschritten der Evaluierung.
      - "results": Array mit den durchschnittlichen Belohnungen (oder Ergebnissen) pro Evaluierung.
      - "ep_lengths": Array mit der durchschnittlichen Episodenlänge.
    """
    # Lade das .npz-Archiv
    data = np.load(npz_path)
    print("Gefundene Keys in der npz-Datei:", list(data.keys()))
    
    # Verwende den "timesteps"-Schlüssel, falls vorhanden
    if 'timesteps' in data:
        timesteps = data['timesteps']
    else:
        first_key = list(data.keys())[0]
        timesteps = np.arange(len(data[first_key]))
    
    # Verwende "results" als durchschnittliche Belohnung
    mean_reward = data.get('results', None)
    if mean_reward is None:
        print("results-Daten (als mean_reward) nicht vorhanden.")
    
    # Verwende "ep_lengths" als durchschnittliche Episodenlänge
    mean_ep_length = data.get('ep_lengths', None)
    if mean_ep_length is None:
        print("ep_lengths-Daten (als mean_ep_length) nicht vorhanden.")
    
    # Erstelle das Plot
    plt.figure(figsize=(12, 6))
    
    if mean_reward is not None:
        plt.subplot(1, 2, 1)
        plt.plot(timesteps, mean_reward, marker='o', linestyle='-', color='blue')
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Reward")
        plt.title("Durchschnittliche Belohnung über Zeitschritte")
        plt.grid(True)
    else:
        print("Keine 'results'-Daten zum Plotten der Belohnung gefunden.")
    
    if mean_ep_length is not None:
        plt.subplot(1, 2, 2)
        plt.plot(timesteps, mean_ep_length, marker='o', linestyle='-', color='orange')
        plt.xlabel("Timesteps")
        plt.ylabel("Mean Episode Length")
        plt.title("Durchschnittliche Episodenlänge über Zeitschritte")
        plt.grid(True)
    else:
        print("Keine 'ep_lengths'-Daten zum Plotten der Episodenlänge gefunden.")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Passe den Pfad zur npz-Datei an, falls nötig.
    npz_path = "./logs/evaluations.npz"
    plot_evaluation_metrics(npz_path)