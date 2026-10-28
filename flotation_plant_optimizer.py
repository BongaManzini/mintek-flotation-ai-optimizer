#!/usr/bin/env python3
"""
Mintek Flotation Plant AI Optimizer
Industrial AI model for real-time silica impurity prediction and reagent dosing
optimization in iron ore froth flotation processes.

Author: Bonga Manzini (UNISA BSc Informatics)
Mintek SCI Grad Hackathon — Team UDS DIGGERS
"""

import os
import sys
import argparse
import csv
import json
import http.server
import socketserver
from typing import Dict, Any, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "MiningProcess_Flotation_Plant_Database.csv")
WEB_DIR = os.path.join(BASE_DIR, "web")


def parse_float_safe(val: str, default: float = 0.0) -> float:
    """Safely converts comma-formatted floating point strings (e.g. '55,2') to float."""
    if not val:
        return default
    try:
        clean = val.replace('"', '').replace(',', '.').strip()
        return float(clean)
    except ValueError:
        return default


def load_dataset_sample(file_path: str, max_rows: int = 5000) -> List[Dict[str, float]]:
    """Loads and parses process parameters from the mining plant database."""
    if not os.path.exists(file_path):
        print(f"️ Dataset file not found at: {file_path}")
        return []

    records = []
    with open(file_path, mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return []

        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            if len(row) < 24:
                continue

            records.append({
                "iron_feed": parse_float_safe(row[1]),
                "silica_feed": parse_float_safe(row[2]),
                "starch_flow": parse_float_safe(row[3]),
                "amina_flow": parse_float_safe(row[4]),
                "ore_pulp_flow": parse_float_safe(row[5]),
                "ore_pulp_ph": parse_float_safe(row[6]),
                "ore_pulp_density": parse_float_safe(row[7]),
                "col1_air": parse_float_safe(row[8]),
                "col2_air": parse_float_safe(row[9]),
                "col3_air": parse_float_safe(row[10]),
                "col1_level": parse_float_safe(row[15]),
                "col2_level": parse_float_safe(row[16]),
                "iron_concentrate": parse_float_safe(row[22]),
                "silica_concentrate": parse_float_safe(row[23])  # Target: Silica Impurity
            })

    return records


def predict_silica_impurity(features: Dict[str, float]) -> float:
    """
    Industrial heuristic regression model based on chemical flotation kinetics:
    Silica concentrate is driven by silica feed ratio, amina collector flow, and column aeration.
    """
    silica_feed = features.get("silica_feed", 15.0)
    amina = features.get("amina_flow", 500.0)
    ph = features.get("ore_pulp_ph", 10.0)

    # Amina collects silica; higher amina flow suppresses silica in the concentrate
    # Optimum pH is near 10.0 to 10.4 for reverse cationic flotation
    reagent_efficiency = (amina / 550.0) * (1.0 - abs(ph - 10.15) * 0.16)
    silica_pred = max(0.5, (silica_feed * 0.12) / max(0.4, reagent_efficiency))
    return round(silica_pred, 2)


def optimize_reagents(silica_feed: float, target_silica_max: float = 2.0) -> Dict[str, Any]:
    """Calculates optimal Amina collector and Starch depressant flow rates."""
    # Base dosage calculation for reverse cationic flotation
    recommended_amina = round(450.0 + (silica_feed * 8.5), 1)
    recommended_starch = round(2800.0 + (silica_feed * 15.0), 1)

    predicted_silica = (silica_feed * 0.12) / ((recommended_amina / 550.0))
    status = "OPTIMAL" if predicted_silica <= target_silica_max else "ACTION REQUIRED"

    # Metallurgical Recovery mass balance estimation
    projected_iron = round(68.2 - (predicted_silica * 0.88), 2)
    tailings_fe = 12.5
    feed_fe = 56.3
    num = projected_iron * (feed_fe - tailings_fe)
    den = feed_fe * (projected_iron - tailings_fe)
    recovery = round((num / max(1e-3, den)) * 100.0, 1)

    return {
        "status": status,
        "predicted_silica_impurity_percent": round(predicted_silica, 2),
        "target_max_impurity": target_silica_max,
        "recommended_amina_flow_m3h": recommended_amina,
        "recommended_starch_flow_m3h": recommended_starch,
        "projected_iron_grade_percent": projected_iron,
        "estimated_iron_recovery_percent": recovery
    }


def serve_web_portal(port: int = 8080):
    """Hosts the interactive research portal and digital twin locally."""
    if not os.path.exists(WEB_DIR):
        print(f" Web directory not found at: {WEB_DIR}")
        return

    os.chdir(WEB_DIR)
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║    Mintek AI Flotation Twin & Research Portal is Live!           ║")
        print(f"║   URL: http://localhost:{port}                                      ║")
        print("║   Press Ctrl+C to terminate the local server.                      ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n Server stopped.")


def main():
    parser = argparse.ArgumentParser(description="Mintek Industrial AI Mineral Processing Optimizer")
    parser.add_argument("--feed-silica", type=float, default=14.5, help="Current % Silica in raw ore feed")
    parser.add_argument("--target-impurity", type=float, default=2.0, help="Maximum allowed % Silica in concentrate")
    parser.add_argument("--eval-dataset", action="store_true", help="Evaluate model against plant database sample")
    parser.add_argument("--serve", type=int, nargs="?", const=8080, help="Start interactive research web portal on specified port (default: 8080)")
    args = parser.parse_args()

    if args.serve:
        serve_web_portal(args.serve)
        return

    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║        ️ Mintek Flotation Plant AI Optimizer                     ║")
    print("║        Industrial Mineral Processing & Impurity Reduction          ║")
    print("║        Author: Bonga Manzini | Team: UDS DIGGERS                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"Feed Silica Grade:      {args.feed_silica}%")
    print(f"Target Max Impurity:    {args.target_impurity}%")
    print("──────────────────────────────────────────────────────────────────────")

    rec = optimize_reagents(args.feed_silica, args.target_impurity)

    print("\n OPTIMIZATION RECOMMENDATIONS:")
    print(f"  • Recommended Amina Collector Flow:  {rec['recommended_amina_flow_m3h']} m³/h")
    print(f"  • Recommended Starch Depressant:    {rec['recommended_starch_flow_m3h']} m³/h")
    print(f"  • Forecasted Silica Impurity:       {rec['predicted_silica_impurity_percent']}%")
    print(f"  • Projected Iron (Fe) Concentrate:  {rec['projected_iron_grade_percent']}% Fe")
    print(f"  • Estimated Iron Recovery Rate:     {rec['estimated_iron_recovery_percent']}%")
    print(f"  • Plant Compliance Status:          [{rec['status']}]")

    if args.eval_dataset and os.path.exists(DATASET_PATH):
        print("\n Sampling 1,000 real plant telemetry records from database...")
        samples = load_dataset_sample(DATASET_PATH, max_rows=1000)
        if samples:
            errors = []
            for s in samples:
                pred = predict_silica_impurity(s)
                actual = s["silica_concentrate"]
                if actual > 0:
                    errors.append(abs(pred - actual))

            mae = sum(errors) / len(errors) if errors else 0.0
            print(f"   Sampled {len(samples)} plant telemetry timestamps.")
            print(f"   Mean Absolute Error (MAE): ±{mae:.3f}% silica concentration.")
            print(f"   Sub-second soft sensor eliminates 120-minute laboratory assay delay!")


if __name__ == "__main__":
    main()
