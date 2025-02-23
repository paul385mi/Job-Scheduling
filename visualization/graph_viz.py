# Dieses Modul enthält Funktionen zur Visualisierung von Produktionsgraphen
# und zur Ausgabe von Simulationsberichten

# Importieren der benötigten Bibliotheken für die Graphendarstellung
import matplotlib.pyplot as plt  # Für die Erstellung von Plots
import networkx as nx           # Für die Graphenmanipulation und -darstellung

def visualize_production_graph(G, conflict_edges):
    '''
    Visualisiert einen Produktionsgraphen mit Präzedenz- und Konfliktbeziehungen.
    
    Parameter:
    - G: NetworkX-Graph, der die Produktionsschritte (Knoten) und Präzedenzbeziehungen (Kanten) enthält
    - conflict_edges: Liste von Tupeln, die Maschinenkonflikte zwischen Produktionsschritten darstellt
    
    Returns:
    - fig, ax: Figure und Axes-Objekte des erstellten Plots
    '''
    # Erstellen einer Liste aller Knoten für spätere Indizierung
    node_list = list(G.nodes())
    
    # Initialisierung der Plot-Umgebung mit definierter Größe
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Berechnung der Knotenpositionierung mit festgelegtem Seed für Reproduzierbarkeit
    pos = nx.spring_layout(G, seed=42)
    
    # Zeichnen der Knoten als hellblaue Kreise mit Labels
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=1000)
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    # Zeichnen der Präzedenzkanten als schwarze Pfeile
    # Diese zeigen die Reihenfolgebeziehungen zwischen den Produktionsschritten
    for u, v in G.edges():
        ax.annotate("", xy=pos[v], xycoords='data', xytext=pos[u], textcoords='data',
                   arrowprops=dict(arrowstyle="->", color="black", shrinkA=5, 
                                 shrinkB=5, connectionstyle="arc3,rad=0.0"))
    
    # Zeichnen der Konfliktkanten als rote gestrichelte Linien
    # Diese zeigen an, welche Produktionsschritte nicht gleichzeitig ausgeführt werden können
    conflict_edges_idx = [[node_list.index(u), node_list.index(v)] 
                         for u, v in conflict_edges]
    for edge in conflict_edges_idx:
        u, v = edge
        ax.plot([pos[node_list[u]][0], pos[node_list[v]][0]],
                [pos[node_list[u]][1], pos[node_list[v]][1]], 
                color="red", linestyle="dashed")
    
    # Hinzufügen eines erklärenden Titels und Ausblenden der Achsen
    ax.set_title("Produktionsprozess: Präzedenz (schwarz) & Maschinenkonflikte (rot, gestrichelt)")
    ax.axis('off')
    return fig, ax

def print_step_report(env, action, reward, info, last_history, new_finished):
    print("\n================ Simulation Step Summary ================\n")
    print(f"Time: {env.env.now:.2f}")
    print(f"Action (Mode, Count): {action}")
    print(f"Reward: {reward:.3f}")
    print(f"Total Finished Jobs: {len(env.simulation.job_finish_times)}")
    print(f"Total Waiting Jobs: {len(env.simulation.waiting_jobs)}")
    print(f"Machine Utilization: {sum(1 for m in env.simulation.machines.values() if m.count > 0)}/{len(env.simulation.machines)}")
    
    print("\n--- Newly Finished Jobs ---")
    for job, (ft, priority) in new_finished.items():
        print(f"  {job}: Finished at {ft:.2f} (Priority: {priority})")

    print("\n--- Machine Status ---")
    for machine, resource in env.simulation.machines.items():
        print(f"  {machine}: {resource.count} jobs running")

    print("========================================================\n")
    
    # Ausgabe der Nachbestellungsereignisse, falls vorhanden
    print("Reorder Events:")
    if info['reorder_events']:
        print(f"{'Time':>8} | {'Aid':>10} | {'Action':>10} | {'Order Cost':>12}")
        print("-" * 50)
        for event in info['reorder_events']:
            print(f"{event['time']:8.2f} | {event['aid']:>10} | "
                  f"{event['action']:>10} | {event['order_cost']:12}")
    else:
        print("  None")
    print("-" * 50)
    
    # Ausgabe der neu gestarteten Jobs mit ihren Prioritäten
    print("Started Jobs (Name, Priority):")
    print(f"{'Job Name':>10} | {'Priority':>8}")
    print("-" * 30)
    for job_name, priority in last_history:
        print(f"{job_name:>10} | {priority:8}")
    print("-" * 50)
    
    # Ausgabe der fertiggestellten Jobs mit Zeitstempel und Priorität
    print("Newly Finished Jobs (Name, Finish Time, Priority):")
    print(f"{'Job Name':>10} | {'Finish Time':>12} | {'Priority':>8}")
    print("-" * 40)
    for job, (finish_time, priority) in new_finished.items():
        print(f"{job:>10} | {finish_time:12.3f} | {priority:8}")
    print("=" * 50)
    print("\n")
