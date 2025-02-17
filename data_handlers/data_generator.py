"""
Modul zur Generierung synthetischer Produktionsdaten

Dieses Modul ermöglicht die Erstellung realistischer Produktionsdaten für Simulationszwecke.
Es generiert eine hierarchische Struktur von Aufträgen (Jobs) und deren Operationen,
inklusive verschiedener Produktionsparameter wie:
- Bearbeitungszeiten
- Maschinenauswahl
- Produktionshilfsmittel
- Lager- und Rüstanforderungen

Die Daten werden in einem strukturierten JSON-Format gespeichert.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import random
import json
import os
from pathlib import Path

@dataclass
class ProductionConfig:
    """Konfigurationsklasse für die Generierung von Produktionsdaten.
    
    Diese Klasse definiert alle Parameter und Bereiche für die Datengenerierung:
    - Anzahl und Umfang der Aufträge
    - Verfügbare Ressourcen (Maschinen, Hilfsmittel, Materialien)
    - Zeit- und Kostenbereiche für verschiedene Produktionsaspekte
    - Prioritätsstufen für Aufträge
    """
    n_jobs: int = 50
    min_ops: int = 2
    max_ops: int = 5
    machines: List[str] = field(default_factory=lambda: ["M1", "M2", "M3", "M4"])
    production_aids: List[str] = field(
        default_factory=lambda: ["Schablone", "Öl", "Kühlmittel", "Werkzeug"]
    )
    materials: List[str] = field(
        default_factory=lambda: ["Material_A", "Material_B", "Material_C"]
    )
    time_ranges: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "processing": {"min": 20, "max": 60},
        "setup": {"min": 1, "max": 10},
        "storage": {"min": 5, "max": 20}
    })
    cost_ranges: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "setup": {"min": 10, "max": 50},
        "storage": {"min": 1, "max": 5}
    })
    priority_range: Dict[str, int] = field(
        default_factory=lambda: {"min": 1, "max": 10}
    )

class Operation:
    """Repräsentiert eine einzelne Produktionsoperation.
    
    Jede Operation enthält:
    - Eindeutigen Namen (basierend auf Job und Operationsnummer)
    - Bearbeitungszeit
    - Zugewiesene Maschine
    - Vorgängerbeziehungen
    - Produziertes Material
    - Optional: Produktionshilfsmittel, Rüst- und Lagerdetails
    """
    def __init__(
        self,
        job_name: str,
        op_number: int,
        config: ProductionConfig
    ):
        self.name = f"{job_name}_Op{op_number}"
        self.processing_time = random.randint(
            config.time_ranges["processing"]["min"],
            config.time_ranges["processing"]["max"]
        )
        self.machine = random.choice(config.machines)
        self.predecessors = (
            [f"{job_name}_Op{op_number-1}"] if op_number > 1 else None
        )
        self.produced_material = random.choice(config.materials)
        
        # Optional attributes
        self.production_aids = self._generate_production_aids(config)
        self.setup_details = self._generate_setup_details(config)
        self.storage_details = self._generate_storage_details(config)

    def _generate_production_aids(
        self,
        config: ProductionConfig
    ) -> Optional[List[str]]:
        """Generiert bei Bedarf Produktionshilfsmittel (30% Wahrscheinlichkeit).
        
        Returns:
            Liste von 1-2 zufällig ausgewählten Hilfsmitteln oder None
        """
        if random.random() < 0.3:
            return random.sample(
                config.production_aids,
                random.randint(1, 2)
            )
        return None

    def _generate_setup_details(
        self,
        config: ProductionConfig
    ) -> Dict[str, int]:
        """Generiert Rüstzeiten und -kosten, falls Hilfsmittel benötigt werden.
        
        Wird nur aufgerufen, wenn Produktionshilfsmittel vorhanden sind.
        Erzeugt realistische Rüstparameter innerhalb der konfigurierten Bereiche.
        
        Returns:
            Dictionary mit Rüstzeit und -kosten oder leeres Dictionary
        """
        if self.production_aids:
            return {
                "time": random.randint(
                    config.time_ranges["setup"]["min"],
                    config.time_ranges["setup"]["max"]
                ),
                "cost": random.randint(
                    config.cost_ranges["setup"]["min"],
                    config.cost_ranges["setup"]["max"]
                )
            }
        return {}

    def _generate_storage_details(
        self,
        config: ProductionConfig
    ) -> Optional[Dict[str, int]]:
        """Generiert Zwischenlageranforderungen mit 30% Wahrscheinlichkeit.
        
        Bestimmt:
        - Minimale Verweildauer im Zwischenlager
        - Anfallende Lagerkosten pro Zeiteinheit
        
        Returns:
            Dictionary mit Lagerparametern oder None
        """
        if random.random() < 0.3:
            return {
                "minVerweildauer": random.randint(
                    config.time_ranges["storage"]["min"],
                    config.time_ranges["storage"]["max"]
                ),
                "lagerkosten": random.randint(
                    config.cost_ranges["storage"]["min"],
                    config.cost_ranges["storage"]["max"]
                )
            }
        return None

    def to_dict(self) -> Dict[str, Union[str, int, List[str], Dict[str, int]]]:
        """Convert operation to dictionary format."""
        operation_dict = {
            "Name": self.name,
            "benötigteZeit": self.processing_time,
            "Maschine": self.machine,
            "Vorgänger": self.predecessors,
            "produziertesMaterial": self.produced_material
        }

        if self.production_aids:
            operation_dict["benötigteHilfsmittel"] = self.production_aids
            operation_dict["umruestzeit"] = self.setup_details["time"]
            operation_dict["umruestkosten"] = self.setup_details["cost"]

        if self.storage_details:
            operation_dict["zwischenlager"] = self.storage_details

        return operation_dict

class Job:
    """Repräsentiert einen Produktionsauftrag mit mehreren Operationen.
    
    Ein Job besteht aus:
    - Eindeutigem Namen
    - Prioritätsstufe (1-10)
    - Liste von Operationen in definierter Reihenfolge
    
    Die Anzahl der Operationen wird zufällig zwischen min_ops und max_ops gewählt.
    """
    def __init__(
        self,
        job_number: int,
        config: ProductionConfig
    ):
        self.name = f"Job_{job_number}"
        self.priority = random.randint(
            config.priority_range["min"],
            config.priority_range["max"]
        )
        self.operations = self._generate_operations(config)

    def _generate_operations(
        self,
        config: ProductionConfig
    ) -> List[Operation]:
        """Generate a list of operations for the job."""
        n_ops = random.randint(config.min_ops, config.max_ops)
        return [
            Operation(self.name, i, config)
            for i in range(1, n_ops + 1)
        ]

    def to_dict(self) -> Dict[str, Union[str, int, List[Dict]]]:
        """Convert job to dictionary format."""
        return {
            "Name": self.name,
            "Priorität": self.priority,
            "Operationen": [op.to_dict() for op in self.operations]
        }

class ProductionDataGenerator:
    """Hauptklasse zur Generierung synthetischer Produktionsdaten.
    
    Diese Klasse koordiniert den gesamten Generierungsprozess:
    - Erstellt die konfigurierte Anzahl von Jobs
    - Generiert für jeden Job die entsprechenden Operationen
    - Speichert die Daten in einem strukturierten JSON-Format
    
    Die Generierung erfolgt nach den in ProductionConfig definierten Parametern.
    """
    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or ProductionConfig()

    def generate_data(self) -> Dict[str, List[Dict]]:
        """Generate complete production dataset."""
        jobs = [
            Job(j, self.config).to_dict()
            for j in range(1, self.config.n_jobs + 1)
        ]
        return {"jobs": jobs}

    def save_to_json(
        self,
        output_dir: Union[str, Path],
        filename: str = "synthetic_data.json"
    ) -> Path:
        """Generate and save data to JSON file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        data = self.generate_data()
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        return output_path

def generate_test_data(
    output_dir: Union[str, Path],
    config: Optional[ProductionConfig] = None
) -> Path:
    """
    Hilfsfunktion zur vereinfachten Generierung von Testdaten.
    
    Diese Funktion kapselt den gesamten Generierungsprozess:
    1. Erstellt einen ProductionDataGenerator mit der gegebenen Konfiguration
    2. Generiert die Testdaten
    3. Speichert sie im JSON-Format
    
    Args:
        output_dir: Zielverzeichnis für die JSON-Datei
        config: Optionale Konfiguration für die Datengenerierung
        
    Returns:
        Pfad zur generierten JSON-Datei
    """
    generator = ProductionDataGenerator(config)
    return generator.save_to_json(output_dir)

if __name__ == "__main__":
    # Example usage with custom configuration
    custom_config = ProductionConfig(
        n_jobs=50,
        min_ops=2,
        max_ops=5,
        time_ranges={
            "processing": {"min": 20, "max": 60},
            "setup": {"min": 1, "max": 10},
            "storage": {"min": 5, "max": 20}
        }
    )
    
    output_dir = Path("/Users/paulmill/Desktop/2025_PYTHON/PROJEKTSTUDIUM/17-02-2025")
    output_path = generate_test_data(output_dir, custom_config)
    print(f"Generated test data saved to: {output_path}")
