"""
Modul für Graph-bezogene Hilfsfunktionen und Transformationen in der Produktionsplanung.

Dieses Modul bietet spezialisierte Funktionen für die Umwandlung von Produktionsplänen in
Graphstrukturen. Es ermöglicht die Modellierung von:
- Operationen als Knoten mit Attributen (Zeit, Maschine, Hilfsmittel, Umrüstzeit)
- Vorgängerbeziehungen als gerichtete Kanten
- Maschinenkonflikten als ungerichtete Kanten
- Visualisierung der Produktionsstruktur

Die Graphstruktur wird sowohl für die Visualisierung als auch für das Training von
Graph Neural Networks verwendet.
"""

# Importieren der benötigten Bibliotheken

# networkx (nx): Umfassende Bibliothek für die Erstellung und Analyse von Graphen
# Wird verwendet für: Grundlegende Graphstruktur, Layout-Algorithmen, Graphmanipulation
import networkx as nx

# torch: Framework für maschinelles Lernen und Tensoroperationen
# Wird verwendet für: Tensorgenerierung, Datentyp-Konvertierung
import torch

# PyTorch Geometric: Erweiterung von PyTorch für Graph Neural Networks
# Wird verwendet für: Spezielle Graphdatenstruktur für das Training
from torch_geometric.data import Data

# matplotlib: Standardbibliothek für wissenschaftliche Visualisierungen
# Wird verwendet für: Graphvisualisierung, Farbschemata, Layout
import matplotlib.pyplot as plt

# numpy: Bibliothek für numerische Berechnungen
# Wird verwendet für: Generierung von Farbpaletten, Array-Operationen
import numpy as np

def build_graph(data_json):
    """
    Erstellt einen gerichteten Graphen aus JSON-Daten der Produktionsplanung.
    
    Die Funktion verarbeitet eine JSON-Struktur mit Produktionsjobs und deren Operationen.
    Jede Operation wird zu einem Knoten im Graphen, wobei folgende Attribute gespeichert werden:
    - benötigteZeit: Ausführungszeit der Operation
    - Maschine: Zugehörige Produktionsmaschine
    - hilfsmittel: Flag für benötigte Hilfsmittel (1=ja, 0=nein)
    - umruestzeit: Zeit für das Umrüsten der Maschine
    
    Beispiel JSON-Struktur:
    {
        "jobs": [
            {
                "Name": "Job1",
                "Operationen": [
                    {
                        "Name": "Op1",
                        "benötigteZeit": 10,
                        "Maschine": "M1",
                        "benötigteHilfsmittel": true,
                        "umruestzeit": 5,
                        "Vorgänger": ["Op0"]
                    }
                ]
            }
        ]
    }
    
    Args:
        data_json (dict): JSON-Objekt mit der Produktionsplan-Struktur
    
    Returns:
        tuple:
            - G (nx.DiGraph): Gerichteter Graph mit Operationen als Knoten
            - op_to_job (dict): Zuordnung von Operations-IDs zu Job-Namen
    
    Hinweise:
        - Vorgängerbeziehungen werden als gerichtete Kanten modelliert
        - Fehlende Attribute (z.B. umruestzeit) werden mit Standardwerten belegt
        - Die Funktion validiert nicht die Vollständigkeit der Eingabedaten
    """
    # Initialisierung des gerichteten Graphen
    G = nx.DiGraph()
    # Dictionary zur Speicherung der Zuordnung von Operationen zu Jobs
    op_to_job = {}
    
    # Iteration über alle Jobs in den JSON-Daten
    for job in data_json["jobs"]:
        job_name = job["Name"]
        # Verarbeitung aller Operationen eines Jobs
        for operation in job["Operationen"]:
            op_name = operation["Name"]
            benötigteZeit = operation["benötigteZeit"]
            maschine = operation["Maschine"]
            # Überprüfung, ob Hilfsmittel benötigt werden (1 wenn ja, 0 wenn nein)
            hilfsmittel_flag = 1 if "benötigteHilfsmittel" in operation else 0
            # Erfassung der Umrüstzeit, Standard ist 0
            umruestzeit = operation.get("umruestzeit", 0)
            
            # Hinzufügen eines Knotens mit allen relevanten Attributen
            G.add_node(op_name, benötigteZeit=benötigteZeit, Maschine=maschine,
                      hilfsmittel=hilfsmittel_flag, umruestzeit=umruestzeit)
            # Zuordnung der Operation zum entsprechenden Job
            op_to_job[op_name] = job_name
            
            # Verarbeitung der Vorgängerbeziehungen
            vorgaenger = operation.get("Vorgänger")
            if vorgaenger:
                # Behandlung von einzelnen und mehreren Vorgängern
                if isinstance(vorgaenger, list):
                    for pred in vorgaenger:
                        G.add_edge(pred, op_name)
                else:
                    G.add_edge(vorgaenger, op_name)
    return G, op_to_job

def add_machine_conflicts(G, op_to_job):
    """
    Identifiziert und erstellt Konfliktkanten zwischen Operationen, die die gleiche Maschine nutzen.
    
    Die Funktion analysiert die Maschinennutzung aller Operationen und erstellt Konfliktkanten
    zwischen Operationen verschiedener Jobs, die die gleiche Maschine benötigen. Diese
    Konfliktkanten repräsentieren potenzielle Ressourcenkonflikte im Produktionsablauf.
    
    Beispiel:
    - Operation A (Job1) und Operation B (Job2) nutzen beide Maschine M1
    - Resultierende Konfliktkanten: (A -> B) und (B -> A)
    
    Funktionsweise:
    1. Gruppierung aller Operationen nach genutzter Maschine
    2. Für jede Maschine mit mehreren Operationen:
       - Vergleich aller Operationspaare
       - Wenn Operationen zu verschiedenen Jobs gehören: Konfliktkanten erstellen
    
    Args:
        G (nx.DiGraph): Graph mit Operationen als Knoten und deren Attributen
        op_to_job (dict): Zuordnung von Operations-IDs zu ihren Job-Namen
    
    Returns:
        list: Liste von Konfliktkanten als Paare [op1, op2], wobei jedes Paar
              bidirektional (also zweimal) enthalten ist
    
    Hinweise:
        - Konflikte werden nur zwischen verschiedenen Jobs berücksichtigt
        - Die Kanten sind bidirektional, da der Konflikt in beide Richtungen besteht
        - Die Reihenfolge der Operationen auf einer Maschine ist noch nicht festgelegt
    """
    # Dictionary zur Gruppierung von Operationen nach Maschinen
    machine_to_ops = {}
    
    # Sammeln aller Operationen pro Maschine
    for node, attr in G.nodes(data=True):
        machine = attr["Maschine"]
        if machine not in machine_to_ops:
            machine_to_ops[machine] = []
        machine_to_ops[machine].append(node)
    
    # Liste für die Konfliktkanten
    conflict_edges = []
    
    # Iteration über alle Maschinen und deren Operationen
    for machine, ops in machine_to_ops.items():
        # Wenn mehr als eine Operation auf einer Maschine ausgeführt wird
        if len(ops) > 1:
            # Erstellen von Konfliktkanten zwischen allen Operationspaaren
            # unterschiedlicher Jobs auf der gleichen Maschine
            for i in range(len(ops)):
                for j in range(i+1, len(ops)):
                    op1, op2 = ops[i], ops[j]
                    # Nur Konflikte zwischen verschiedenen Jobs berücksichtigen
                    if op_to_job[op1] != op_to_job[op2]:
                        # Bidirektionale Konfliktkanten hinzufügen
                        conflict_edges.append([op1, op2])
                        conflict_edges.append([op2, op1])
    return conflict_edges

def build_pyg_data(G, conflict_edges):
    """
    Konvertiert einen NetworkX-Graphen in ein PyTorch Geometric Data-Objekt für maschinelles Lernen.
    
    Diese Funktion transformiert die Graphstruktur in ein Format, das für Graph Neural Networks
    geeignet ist. Dabei werden folgende Umwandlungen vorgenommen:
    
    1. Knotenfeatures (x):
       - Benötigte Zeit (float)
       - Hilfsmittel-Flag (float: 0.0 oder 1.0)
       - Umrüstzeit (float)
       Beispiel: [[10.0, 1.0, 5.0], [8.0, 0.0, 3.0], ...]
    
    2. Kantenindex (edge_index):
       - Vorgängerkanten und Konfliktkanten als Indexpaar-Liste
       - Format: 2 x E Matrix (E = Anzahl Kanten)
       Beispiel: [[0, 1, 2], [1, 2, 0]] für Kanten 0->1, 1->2, 2->0
    
    3. Kantenattribute (edge_attr):
       - 0: Vorgängerkante
       - 1: Konfliktkante
       Beispiel: [0, 0, 1, 1] für zwei Vorgänger- und zwei Konfliktkanten
    
    Args:
        G (nx.DiGraph): NetworkX Graph mit Operationsknoten und Vorgängerkanten
        conflict_edges (list): Liste von Konfliktkanten als [op1, op2] Paare
    
    Returns:
        Data: PyTorch Geometric Data-Objekt mit:
            - x: Knotenfeature-Matrix (N x 3)
            - edge_index: Kantenindex-Matrix (2 x E)
            - edge_attr: Kantentyp-Vektor (E)
            wobei N = Anzahl Knoten, E = Anzahl Kanten
    
    Hinweise:
        - Alle Eingabewerte werden zu float-Tensoren konvertiert
        - Die Reihenfolge der Knoten wird durch node_to_index festgelegt
        - Die Graphstruktur bleibt durch die Indizierung erhalten
    """
    # Erstellen einer Liste aller Knoten und Zuordnung zu numerischen Indizes
    node_list = list(G.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    
    # Umwandlung der Vorgänger-Kanten in numerische Indizes
    precedence_edges = []
    for u, v in G.edges():
        precedence_edges.append([node_to_index[u], node_to_index[v]])
    # Markierung der Vorgänger-Kanten mit 0
    precedence_edge_attr = [0] * len(precedence_edges)
    
    # Umwandlung der Konfliktkanten in numerische Indizes
    conflict_edges_idx = [[node_to_index[u], node_to_index[v]] for u, v in conflict_edges]
    # Markierung der Konfliktkanten mit 1
    conflict_edge_attr = [1] * len(conflict_edges_idx)
    
    # Zusammenführen aller Kanten und ihrer Attribute
    all_edges = precedence_edges + conflict_edges_idx
    all_edge_attr = precedence_edge_attr + conflict_edge_attr
    
    # Konvertierung in PyTorch Tensoren
    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(all_edge_attr, dtype=torch.long)
    
    # Erstellen der Knotenfeatures (benötigteZeit, hilfsmittel, umruestzeit)
    x = []
    for node in node_list:
        attr = G.nodes[node]
        feature = [float(attr["benötigteZeit"]), float(attr["hilfsmittel"]), float(attr["umruestzeit"])]
        x.append(feature)
    x = torch.tensor(x, dtype=torch.float)
    
    # Erstellen des finalen PyTorch Geometric Data-Objekts
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def visualize_graph(G, op_to_job):
    """
    Erstellt eine visuelle Darstellung des Produktionsplans als farbigen, interaktiven Graphen.
    
    Die Visualisierung bietet folgende Eigenschaften:
    1. Knoten (Operationen):
       - Größe: 1000 Einheiten
       - Transparenz: 0.7 (70% sichtbar)
       - Farbe: Eindeutige Farbe pro Job aus der 'Set3' Farbpalette
       - Beschriftung: Name der Operation
    
    2. Kanten (Beziehungen):
       - Vorgängerkanten: Schwarz, mit Pfeilspitzen
       - Pfeilgröße: 20 Einheiten
    
    3. Layout:
       - Spring-Layout-Algorithmus für optimale Knotenpositionierung
       - Parameter: k=1 (Federkonstante), 50 Iterationen
       - Automatische Vermeidung von Überlappungen
    
    4. Legende:
       - Position: Oben links, außerhalb des Graphen
       - Zeigt Job-Namen mit zugehörigen Farben
       - Kreismarker für jeden Job
    
    Args:
        G (nx.DiGraph): NetworkX Graph mit den Operationen und Vorgängerbeziehungen
        op_to_job (dict): Zuordnung von Operations-IDs zu ihren Job-Namen
    
    Returns:
        plt.Figure: Matplotlib-Figure-Objekt mit der Graphvisualisierung
    
    Hinweise:
        - Die Visualisierung ist interaktiv (Zoom, Pan)
        - Die Knotenpositionen werden bei jedem Aufruf neu berechnet
        - Die Farbpalette unterstützt bis zu 12 verschiedene Jobs
        - Bei vielen Knoten kann das Layout unübersichtlich werden
    """
    # Erstellen einer neuen Figur mit angepasster Größe
    plt.figure(figsize=(12, 8))
    
    # Position der Knoten mit dem spring_layout berechnen
    # Dieses Layout eignet sich gut für die Visualisierung von Graphen mit vielen Kanten
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Sammeln aller einzigartigen Jobs für die Farbzuordnung
    unique_jobs = list(set(op_to_job.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_jobs)))
    job_to_color = dict(zip(unique_jobs, colors))
    
    # Erstellen einer Liste von Farben für jeden Knoten
    node_colors = [job_to_color[op_to_job[node]] for node in G.nodes()]
    
    # Zeichnen der Knoten
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                         node_size=1000, alpha=0.7)
    
    # Zeichnen der Kanten (Vorgängerbeziehungen)
    nx.draw_networkx_edges(G, pos, edge_color='black', 
                          arrows=True, arrowsize=20)
    
    # Beschriftung der Knoten mit den Operationsnamen
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Hinzufügen einer Legende für die Jobs
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, label=job, markersize=10)
                      for job, color in job_to_color.items()]
    plt.legend(handles=legend_elements, loc='upper left', 
               title='Jobs', bbox_to_anchor=(1, 1))
    
    # Anpassen des Layouts
    plt.title('Produktionsplan Graph Visualisierung')
    plt.axis('off')
    plt.tight_layout()
    
    return plt
