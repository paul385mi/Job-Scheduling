# Dieses Modul implementiert eine Gym-Umgebung für die Produktionsplanung und -steuerung

# Import der benötigten Bibliotheken
import gym  # Grundlegende Gym-Funktionalitäten
from gym import spaces  # Definitionen für Aktions- und Zustandsräume
import numpy as np  # Numerische Berechnungen
import simpy  # Ereignisbasierte Simulation
from simulation.production_sim import OperationBasedProductionSimulation  # Eigene Simulationsklasse
import torch  # Deep Learning Framework

class RealisticSchedulingEnv(gym.Env):
    '''
    Eine angepasste Gym-Umgebung für realistische Produktionsplanung.
    Implementiert das OpenAI Gym-Interface für Reinforcement Learning.
    '''
    def __init__(self, transformer_model, pyg_data, jobs_data):
        '''
        Initialisiert die Umgebung mit dem Transformer-Modell und den Produktionsdaten.
        
        Parameter:
        - transformer_model: Vortrainiertes Graph-Transformer-Modell
        - pyg_data: PyTorch Geometric Datensatz mit Graphstruktur
        - jobs_data: Liste der Produktionsaufträge mit Details
        '''
        super(RealisticSchedulingEnv, self).__init__()
        # Speichern der Modell- und Datenkonfiguration
        self.transformer_model = transformer_model
        self.pyg_data = pyg_data
        self.jobs_data = jobs_data
        
        # Definition des Zustandsraums (8D Embedding + 3D Metriken = 11 Dimensionen)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)
        # Definition des Aktionsraums: [Scheduling-Modus (3), Anzahl Jobs (20)]
        self.action_space = spaces.MultiDiscrete([3, 20])
        
        # Initialisierung der Verlaufsverfolgung
        self.history = []
        self.simulation_class = OperationBasedProductionSimulation
        self.reported_finished_jobs = set()

    def reset(self):
        '''
        Setzt die Umgebung in den Ausgangszustand zurück.
        
        Returns:
        - Initialer Zustand der Umgebung
        '''
        # Zurücksetzen aller Verlaufsdaten
        self.history = []
        self.reported_finished_jobs = set()
        
        # Neue Simulationsumgebung erstellen
        self.env = simpy.Environment()
        self.simulation = self.simulation_class(self.env, self.jobs_data)
        
        # Kurze Initialisierungsphase
        self.env.run(until=0.001)
        self.update_state()
        
        # Ersten Historieneintrag erstellen
        self.history.append((self.env.now, (-1, -1), [], {}))
        return self.state

    def update_state(self):
        '''
        Aktualisiert den Zustand der Umgebung basierend auf:
        1. Graph-Embeddings aus dem Transformer
        2. Aktuelle Simulationsmetriken
        '''
        # Transformer-Modell in Evaluierungsmodus setzen
        self.transformer_model.eval()
        
        # Graph-Embeddings berechnen
        with torch.no_grad():
            out = self.transformer_model(self.pyg_data.x, self.pyg_data.edge_index)
        global_embedding = out.mean(dim=0).numpy()
        
        # Erweiterte Metriken sammeln
        current_makespan = self.simulation.current_makespan
        machine_utilization = len([m for m in self.simulation.machines.values() if m.count > 0]) / len(self.simulation.machines)
        waiting_jobs_count = len(self.simulation.waiting_jobs)
        
        # Kombinierte Metriken
        sim_metrics = np.array([current_makespan, machine_utilization, waiting_jobs_count], dtype=np.float32)
        
        # Gesamtzustand aus Embedding und Metriken erstellen
        self.state = np.concatenate((global_embedding, sim_metrics))

    def step(self, action):
        '''
        Führt einen Simulationsschritt basierend auf der gewählten Aktion aus.
        
        Parameter:
        - action: Tuple (mode, count) für Scheduling-Strategie und Anzahl der Jobs
        
        Returns:
        - state: Neuer Umgebungszustand
        - reward: Erhaltene Belohnung
        - done: Flag ob Simulation beendet
        - info: Zusätzliche Informationen
        '''
        # Aktion in Komponenten zerlegen
        mode = action[0]  # Scheduling-Modus
        count = action[1] + 1  # Anzahl zu planender Jobs (1-21)
        started_jobs = []
        
        # Fall: Keine wartenden Jobs
        if len(self.simulation.waiting_jobs) == 0:
            self.env.run(until=self.env.now + 1)
            selected_jobs = []
        else:
            # Anzahl der tatsächlich zu planenden Jobs bestimmen
            num_to_schedule = min(count, len(self.simulation.waiting_jobs))
            
            # Scheduling-Modus 0: FIFO (First In First Out) mit Priorität
            if mode == 0:
                sorted_jobs = sorted(self.simulation.waiting_jobs,
                                   key=lambda job: (-job.get("Priorität", 0),
                                                  self.simulation.waiting_jobs.index(job)))
            
            # Scheduling-Modus 1: LIFO (Last In First Out) mit Priorität
            elif mode == 1:
                sorted_jobs = sorted(self.simulation.waiting_jobs,
                                   key=lambda job: (-job.get("Priorität", 0),
                                                  -self.simulation.waiting_jobs.index(job)))
            
            # Scheduling-Modus 2: SPT (Shortest Processing Time) mit Priorität als Tie-Breaker
            elif mode == 2:
                def job_time(job):
                    # Basiszeit: Summe aller Operationszeiten
                    base_time = sum(op["benötigteZeit"] for op in job["Operationen"])
                    # Zusatzzeiten: Umrüstzeiten und Mindestverweildauer im Zwischenlager
                    add_time = sum(op.get("umruestzeit", 0) for op in job["Operationen"])
                    add_time += sum(op["zwischenlager"]["minVerweildauer"] 
                                  if "zwischenlager" in op else 0
                                  for op in job["Operationen"])
                    return base_time + add_time
                
                sorted_jobs = sorted(self.simulation.waiting_jobs,
                                   key=lambda job: (job_time(job), -job.get("Priorität", 0)))
            
            # Ausgewählte Jobs aus der Warteliste entnehmen
            selected_jobs = sorted_jobs[:num_to_schedule]
            for job in selected_jobs:
                self.simulation.waiting_jobs.remove(job)
            
            # Liste der gestarteten Jobs mit Namen und Priorität erstellen
            started_jobs = [(job["Name"], job.get("Priorität", "n/a")) 
                           for job in selected_jobs]
            
            # Jobs in der Simulation starten
            for job in selected_jobs:
                self.env.process(self.simulation.schedule_job(job))
            
            # Maximale Verarbeitungszeit der ausgewählten Jobs berechnen
            max_processing_time = max(
                sum(op["benötigteZeit"] + op.get("umruestzeit", 0) +
                    (op["zwischenlager"]["minVerweildauer"] 
                     if "zwischenlager" in op else 0)
                    for op in job["Operationen"])
                for job in selected_jobs
            )
            
            # Simulation bis zum Ende der Verarbeitungszeit ausführen
            self.env.run(until=self.env.now + max_processing_time)
        
        # Zustand aktualisieren und neue fertige Jobs erfassen
        self.update_state()
        finished_job_details = {job["Name"]: job.get("Priorität", "n/a") 
                              for job in self.jobs_data}
        new_finished = {job: (ft, finished_job_details.get(job, "n/a"))
                       for job, ft in self.simulation.job_finish_times.items()
                       if job not in self.reported_finished_jobs}
        self.reported_finished_jobs.update(new_finished.keys())
        
        # Makespan-orientierte Belohnungsberechnung
        current_makespan = self.simulation.current_makespan
        prev_makespan = self.history[-1][0] if self.history else 0
        makespan_delta = current_makespan - prev_makespan
        
        # Hauptkomponente: Makespan-Minimierung
        makespan_reward = -makespan_delta
        
        # Zusätzliche Komponenten für effiziente Scheduling-Entscheidungen
        utilization_factor = len([m for m in self.simulation.machines.values() if m.count > 0]) / len(self.simulation.machines)
        completion_bonus = len(new_finished) * 50 if new_finished else 0
        
        # Bestrafung für Leerlauf
        idle_penalty = 0
        if not selected_jobs and len(self.simulation.waiting_jobs) > 0:
            idle_penalty = -100
        
        # Gesamtbelohnung mit Gewichtung auf Makespan-Minimierung
        reward = makespan_reward * 2.0 + \
                 utilization_factor * 100 + \
                 completion_bonus + \
                 idle_penalty
        
        # Prüfen ob alle Jobs abgearbeitet sind
        done = (len(self.simulation.waiting_jobs) == 0)
        
        # Zusatzinformationen sammeln und Verlauf aktualisieren
        info = {"reorder_events": self.simulation.reorder_log.copy()}
        self.simulation.reorder_log.clear()
        self.history.append((self.env.now, (mode, count), started_jobs, new_finished))
        
        return self.state, reward, done, info

    def get_history(self):
        '''
        Gibt den vollständigen Simulationsverlauf zurück.
        
        Returns:
        - Liste von Tupeln (Zeit, Aktion, gestartete Jobs, beendete Jobs)
        '''
        return self.history