# Dieses Modul implementiert ein Graph-Transformer-Modell für die Verarbeitung von Graphstrukturen

# Importieren der erforderlichen PyTorch-Komponenten
import torch.nn as nn  # Neuronale Netzwerk-Module
import torch.nn.functional as F  # Funktionale Operationen wie Aktivierungsfunktionen
# Import der spezialisierten Transformer-Implementierung für Graphen
from torch_geometric.nn import TransformerConv

class GraphTransformerModel(nn.Module):
    '''
    Ein erweitertes Transformer-Modell zur Verarbeitung von Graphen.
    
    Verbesserte Architektur:
    1. Mehrschichtige Eingabetransformation
    2. Duale Graph-Transformer-Layer
    3. Residual Connections
    4. Dropout für Regularisierung
    5. Layer Normalization
    6. Erweiterte Ausgabeschicht
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        '''
        Initialisiert das erweiterte Transformer-Modell.
        
        Parameter:
        - in_channels: Dimension der Eingabemerkmale
        - hidden_channels: Dimension der versteckten Schicht
        - out_channels: Dimension der Ausgabe
        - heads: Anzahl der Attention-Heads (Standard: 4)
        - dropout: Dropout-Rate für Regularisierung (Standard: 0.1)
        '''
        super(GraphTransformerModel, self).__init__()
        
        # Eingabe-Verarbeitung
        self.input_net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Erste Transformer-Schicht
        self.conv1 = TransformerConv(
            hidden_channels, 
            hidden_channels, 
            heads=heads, 
            concat=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        
        # Zweite Transformer-Schicht
        self.conv2 = TransformerConv(
            hidden_channels * heads, 
            hidden_channels, 
            heads=heads, 
            concat=True,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(hidden_channels * heads)
        
        # Ausgabe-Verarbeitung
        self.output_net = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        # Residual Projektionen
        self.res1 = nn.Linear(hidden_channels, hidden_channels * heads)
        self.res2 = nn.Linear(hidden_channels * heads, hidden_channels * heads)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        '''
        Erweiterter Forward-Pass durch das Modell.
        
        Parameter:
        - x: Knotenmerkmale [N x in_channels]
        - edge_index: Adjazenzmatrix [2 x E]
        
        Returns:
        - Transformierte Knotenmerkmale [N x out_channels]
        '''
        # Eingabe-Verarbeitung
        x = self.input_net(x)
        
        # Erste Transformer-Schicht mit Residual
        identity = self.res1(x)
        x = self.conv1(x, edge_index)
        x = self.norm1(x + identity)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Zweite Transformer-Schicht mit Residual
        identity = self.res2(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x + identity)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Ausgabe-Verarbeitung
        x = self.output_net(x)
        
        return x
