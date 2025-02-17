# Dieses Modul implementiert ein Graph-Transformer-Modell für die Verarbeitung von Graphstrukturen

# Importieren der erforderlichen PyTorch-Komponenten
import torch.nn as nn  # Neuronale Netzwerk-Module
import torch.nn.functional as F  # Funktionale Operationen wie Aktivierungsfunktionen
# Import der spezialisierten Transformer-Implementierung für Graphen
from torch_geometric.nn import TransformerConv

class GraphTransformerModel(nn.Module):
    '''
    Ein Transformer-Modell zur Verarbeitung von Graphen und Erzeugung von Einbettungen (Embeddings).
    
    Architektur:
    1. Lineare Transformation der Eingabe
    2. Graph-Transformer-Layer mit Multi-Head-Attention
    3. ReLU-Aktivierung
    4. Lineare Ausgabeschicht
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        '''
        Initialisiert das Transformer-Modell.
        
        Parameter:
        - in_channels: Dimension der Eingabemerkmale
        - hidden_channels: Dimension der versteckten Schicht
        - out_channels: Dimension der Ausgabe
        - heads: Anzahl der Attention-Heads (Standard: 2)
        '''
        super(GraphTransformerModel, self).__init__()
        # Erste lineare Transformation zur Anpassung der Eingabedimension
        self.lin = nn.Linear(in_channels, hidden_channels)
        # Graph-Transformer-Layer mit Multi-Head-Attention
        # concat=True bedeutet, dass die Ausgaben der Heads konkateniert werden
        self.conv = TransformerConv(hidden_channels, hidden_channels, heads=heads, concat=True)
        # Ausgabe-Layer, beachtet die Konkatenierung der Attention-Heads
        self.lin_out = nn.Linear(hidden_channels * heads, out_channels)
        
    def forward(self, x, edge_index):
        '''
        Forward-Pass durch das Modell.
        
        Parameter:
        - x: Knotenmerkmale (Node Features) [N x in_channels]
        - edge_index: Adjazenzmatrix als Edge-List [2 x E]
        
        Returns:
        - Transformierte Knotenmerkmale [N x out_channels]
        '''
        # Erste Transformation der Eingabemerkmale
        x = self.lin(x)
        # Anwendung des Graph-Transformer-Layers
        x = self.conv(x, edge_index)
        # Nicht-lineare Aktivierung mit ReLU
        x = F.relu(x)
        # Finale lineare Transformation zur gewünschten Ausgabedimension
        x = self.lin_out(x)
        return x
