#!/usr/bin/env python3
"""
Script zur Generierung von Produktionsdaten.
Verwendet den ProductionDataGenerator mit konfigurierbaren Parametern.
"""
import logging
from pathlib import Path
import sys
import argparse
from typing import Optional

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from data_handlers.data_generator import ProductionConfig, generate_test_data

# Logging-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / 'data_generation.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generiere synthetische Produktionsdaten'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(project_root),
        help='Ausgabeverzeichnis für die generierten Daten'
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=50,
        help='Anzahl der zu generierenden Jobs'
    )
    parser.add_argument(
        '--min-ops',
        type=int,
        default=2,
        help='Minimale Anzahl von Operationen pro Job'
    )
    parser.add_argument(
        '--max-ops',
        type=int,
        default=5,
        help='Maximale Anzahl von Operationen pro Job'
    )
    return parser.parse_args()

def create_custom_config(args: argparse.Namespace) -> ProductionConfig:
    """
    Erstelle eine benutzerdefinierte Konfiguration basierend auf den 
    Kommandozeilenargumenten.
    """
    return ProductionConfig(
        n_jobs=args.n_jobs,
        min_ops=args.min_ops,
        max_ops=args.max_ops,
        # Erweiterte Konfiguration für realistischere Produktionsdaten
        time_ranges={
            "processing": {"min": 20, "max": 60},  # Bearbeitungszeit in Minuten
            "setup": {"min": 1, "max": 10},        # Rüstzeit in Minuten
            "storage": {"min": 5, "max": 20}       # Lagerzeit in Minuten
        },
        cost_ranges={
            "setup": {"min": 10, "max": 50},       # Rüstkosten in EUR
            "storage": {"min": 1, "max": 5}        # Lagerkosten in EUR/Minute
        },
        # Erweiterte Maschinenliste für komplexere Szenarien
        machines=[f"M{i}" for i in range(1, 6)],
        # Realistische Produktionshilfsmittel
        production_aids=[
            "Schablone", "Öl", "Kühlmittel", "Werkzeug",
            "Messgerät", "Schutzausrüstung"
        ],
        # Verschiedene Materialtypen
        materials=[f"Material_{c}" for c in "ABCDEFG"]
    )

def main() -> Optional[Path]:
    """
    Hauptfunktion zur Datengenerierung.
    
    Returns:
        Optional[Path]: Pfad zur generierten Datei oder None bei Fehler
    """
    try:
        args = parse_arguments()
        output_dir = Path(args.output_dir)
        
        logger.info(f"Starte Datengenerierung mit {args.n_jobs} Jobs")
        logger.info(f"Operationen pro Job: {args.min_ops}-{args.max_ops}")
        
        # Erstelle benutzerdefinierte Konfiguration
        config = create_custom_config(args)
        
        # Generiere und speichere Daten
        output_path = generate_test_data(output_dir, config)
        
        logger.info(f"Daten erfolgreich generiert: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Fehler bei der Datengenerierung: {str(e)}")
        return None

if __name__ == "__main__":
    main()
