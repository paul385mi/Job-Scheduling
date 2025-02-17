"""
Hauptskript für die Produktionsplanungssimulation

Dieses Skript implementiert eine KI-gestützte Produktionsplanung mit folgenden Hauptkomponenten:
- Generierung und Verarbeitung von Produktionsdaten
- Aufbau und Visualisierung eines Produktionsgraphen
- Training eines PPO (Proximal Policy Optimization) Agenten
- Durchführung einer kontinuierlichen Produktionssimulation

Der Prozess nutzt Graph Transformer und Reinforcement Learning für optimale Scheduling-Entscheidungen.
"""
from stable_baselines3 import PPO
from pathlib import Path

from data_handlers.json_handler import load_data_from_json
from data_handlers.data_generator import ProductionConfig, generate_test_data
from utils.graph_utils import build_graph, add_machine_conflicts, build_pyg_data
from models.transformer_model import GraphTransformerModel
from simulation.env import RealisticSchedulingEnv
from visualization.graph_viz import visualize_production_graph, print_step_report
import matplotlib.pyplot as plt

def main():
    # Initialisierung der Verzeichnispfade
    # base_dir: Hauptverzeichnis des Projekts (wo sich dieses Skript befindet)
    # data_dir: Verzeichnis für die Datenspeicherung (hier gleich base_dir)
    base_dir = Path(__file__).parent
    data_dir = base_dir
    
    # Generierung der Testdaten mit spezifischer Konfiguration
    # - n_jobs: Anzahl der zu planenden Aufträge (50)
    # - min_ops: Minimale Anzahl an Operationen pro Auftrag (2)
    # - max_ops: Maximale Anzahl an Operationen pro Auftrag (5)
    config = ProductionConfig(
        n_jobs=50,
        min_ops=2,
        max_ops=5
    )
    # Erstellt JSON-Datei mit Testdaten und gibt den Pfad zurück
    json_filepath = generate_test_data(data_dir, config)
    
    # Laden der generierten Testdaten aus der JSON-Datei
    data_json = load_data_from_json(json_filepath)

    
    # Aufbau des Produktionsgraphen
    # - G: NetworkX-Graph-Objekt mit Operationen als Knoten
    # - op_to_job: Mapping von Operationen zu ihren zugehörigen Aufträgen
    G, op_to_job = build_graph(data_json)
    # Hinzufügen von Maschinenkonfliktkanten
    # Diese Kanten verbinden Operationen, die nicht gleichzeitig auf derselben Maschine ausgeführt werden können
    conflict_edges = add_machine_conflicts(G, op_to_job)
    
    # Visualisierung des Produktionsgraphen
    # - Normale Kanten zeigen Abhängigkeiten zwischen Operationen
    # - Konfliktkanten (anders gefärbt) zeigen Maschinenkonflikt
    fig, ax = visualize_production_graph(G, conflict_edges)
    plt.show()
    
    # Konvertierung des NetworkX-Graphen in PyTorch Geometric (PyG) Format
    # Dies ist notwendig für die Verarbeitung durch den Graph Transformer
    # - x: Knotenfeatures (z.B. Operationszeiten, Prioritäten)
    # - edge_index: Kantenliste im COO-Format
    # - edge_attr: Kantenattribute (z.B. Kantentyp: normal/Konflikt)
    pyg_data = build_pyg_data(G, conflict_edges)
    print("PyG Data Objekt:")
    print("x (Knotenfeatures):", pyg_data.x)
    print("edge_index:", pyg_data.edge_index)
    print("edge_attr:", pyg_data.edge_attr)
    
    # Initialisierung des Graph Transformer Modells
    # Parameter:
    # - in_channels: Anzahl der Eingabefeatures pro Knoten (3)
    # - hidden_channels: Größe der versteckten Schichten (16)
    # - out_channels: Größe der Ausgabe pro Knoten (8)
    # - heads: Anzahl der Attention Heads (2)
    transformer_model = GraphTransformerModel(
        in_channels=3,
        hidden_channels=16, 
        out_channels=8,
        heads=2
    )
    # Erste Verarbeitung der Graphdaten durch den Transformer
    # Erzeugt Knotenembeddings für die weitere Verarbeitung
    transformer_output = transformer_model(pyg_data.x, pyg_data.edge_index)
    print("Output des Graph Transformers (Knotenembeddings):")
    print(transformer_output)
    
    # Einrichtung der Simulationsumgebung und Training des PPO-Agenten
    # Die Umgebung kombiniert den Transformer mit den Produktionsdaten
    env = RealisticSchedulingEnv(transformer_model, pyg_data, data_json["jobs"])
    obs = env.reset()
    print("Initialer Zustand:", obs)
    
    # Initialisierung und Training des PPO (Proximal Policy Optimization) Agenten
    # - MlpPolicy: Verwendet ein Multi-Layer Perceptron als Policy-Netzwerk
    # - 10000 Zeitschritte für das Training
    ppo_model = PPO("MlpPolicy", env, verbose=1)
    ppo_model.learn(total_timesteps=10000)
    
    # Durchführung der kontinuierlichen Produktionssimulation
    print("Starte kontinuierliche Simulation, um alle Jobs zu verarbeiten:")
    done = False
    obs = env.reset()
    
    # Hauptsimulationsschleife
    while not done:
        # Vorhersage der nächsten Aktion durch den trainierten PPO-Agenten
        action, _states = ppo_model.predict(obs)
        # Ausführung der Aktion in der Umgebung
        obs, reward, done, info = env.step(action)
        
        # Sammeln von Informationen über den aktuellen Simulationsschritt
        last_history = env.get_history()[-1][2]
        # Erfassung neu fertiggestellter Aufträge mit ihren Fertigstellungszeiten
        # und Prioritäten für die Berichterstattung
        new_finished = {
            job: (ft, next(
                (j.get("Priorität", "n/a") 
                 for j in data_json["jobs"] if j["Name"] == job),
                "n/a"
            ))
            for job, ft in env.simulation.job_finish_times.items()
            if job in env.reported_finished_jobs
        }
        # Ausgabe eines detaillierten Berichts für den aktuellen Simulationsschritt
        print_step_report(env, action, reward, info, last_history, new_finished)

if __name__ == "__main__":
    main()
