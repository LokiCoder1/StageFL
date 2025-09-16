"""
Experiment Metrics Module - Advanced metrics tracking for Federated Learning experiments
Optimized for configuration analysis and performance comparison
"""

import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import time


class ExperimentMetricsTracker:
    """
    Advanced metrics tracker for Federated Learning experiments.
    
    This tracker provides comprehensive monitoring and analysis of FL experiments,
    focusing on:
    - Convergence metrics: tracking loss/accuracy evolution and stability
    - Performance comparison: systematic comparison across configurations
    - Optimal configuration discovery: identifying best performing setups
    
    Key features:
    - Automatic experiment tracking with unique IDs
    - Real-time round-by-round metrics collection
    - Stability analysis of convergence patterns
    - CSV/JSON export for further analysis
    - Comparative reports across multiple experiments
    
    Usage:
        tracker = get_experiment_tracker()
        tracker.start_experiment(id, name, nodes, rounds, epochs)
        # During training:
        tracker.track_round(round_num, loss, accuracy)
        # At the end:
        tracker.finalize_experiment()
    """
    
    def __init__(self):
        self.metrics_dir = "pytorchtest/experiment_metrics"
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Storage per esperimenti
        self.experiments = {}
        self.comparison_table = []
        
        # File per risultati comparativi
        self.comparison_file = os.path.join(self.metrics_dir, "configuration_comparison.json")
        self.load_existing_results()
        
    def load_existing_results(self):
        """Carica risultati esistenti se presenti"""
        if os.path.exists(self.comparison_file):
            try:
                with open(self.comparison_file, 'r') as f:
                    data = json.load(f)
                    self.experiments = data.get("experiments", {})
                    self.comparison_table = data.get("comparison_table", [])
            except:
                print("⚠️  Impossibile caricare risultati esistenti, inizio nuovo")
    
    def start_experiment(self, experiment_id: int, group_name: str, 
                        nodes: str, rounds: str, epochs: str) -> None:
        """
        Inizia il tracking di un esperimento
        
        Args:
            experiment_id: ID numerico dell'esperimento
            group_name: Nome nel formato EXP_001_N5_R50_E10
            nodes: Numero di nodi (come stringa per compatibilità)
            rounds: Numero di round (come stringa per compatibilità)
            epochs: Numero di epoche locali (come stringa per compatibilità)
        """
        self.current_experiment = {
            "experiment_id": experiment_id,
            "group_name": group_name,
            "nodes": int(nodes) if nodes != "unknown" else 0,
            "rounds": int(rounds) if rounds != "unknown" else 0,
            "epochs": int(epochs) if epochs != "unknown" else 0,
            "start_time": time.time(),
            "round_metrics": {},
            "convergence_history": []
        }
        
        print(f"📊 Experiment Metrics Tracker initialized for: {group_name}")
    
    def track_round(self, round_num: int, aggregated_loss: float, 
                   aggregated_accuracy: float, client_losses: List[float] = None) -> None:
        """
        Traccia le metriche di un singolo round
        
        Args:
            round_num: Numero del round corrente
            aggregated_loss: Loss aggregata del round
            aggregated_accuracy: Accuracy aggregata del round
            client_losses: Lista delle loss dei singoli client (opzionale)
        """
        if not hasattr(self, 'current_experiment'):
            print("⚠️  No active experiment!")
            return
            
        round_data = {
            "round": round_num,
            "loss": aggregated_loss,
            "accuracy": aggregated_accuracy,
            "timestamp": time.time()
        }
        
        # Se abbiamo le loss dei client, calcola la varianza
        if client_losses:
            round_data["client_loss_variance"] = float(np.var(client_losses))
            round_data["client_loss_std"] = float(np.std(client_losses))
        
        self.current_experiment["round_metrics"][round_num] = round_data
        self.current_experiment["convergence_history"].append({
            "round": round_num,
            "loss": aggregated_loss,
            "accuracy": aggregated_accuracy
        })
        
        # Check se abbiamo raggiunto 90% accuracy per la prima volta
        if aggregated_accuracy >= 0.9 and "first_90_accuracy_round" not in self.current_experiment:
            self.current_experiment["first_90_accuracy_round"] = round_num
            print(f"🎯 Reached 90% accuracy at round {round_num}!")
    
    def calculate_stability_score(self, last_n_rounds: int = 10) -> float:
        """
        Calcola un punteggio di stabilità basato sulla varianza degli ultimi N round
        
        Returns:
            Score tra 0 e 1, dove 1 = massima stabilità
        """
        if not hasattr(self, 'current_experiment'):
            return 0.0
            
        history = self.current_experiment["convergence_history"]
        if len(history) < last_n_rounds:
            last_n_rounds = len(history)
            
        if last_n_rounds < 2:
            return 1.0  # Non abbastanza dati, assumiamo stabile
            
        recent_losses = [h["loss"] for h in history[-last_n_rounds:]]
        recent_accuracies = [h["accuracy"] for h in history[-last_n_rounds:]]
        
        # Normalizza la varianza per ottenere un score tra 0 e 1
        loss_variance = np.var(recent_losses)
        acc_variance = np.var(recent_accuracies)
        
        # Combina le due varianze (pesate)
        combined_variance = 0.7 * loss_variance + 0.3 * acc_variance
        
        # Converti in score di stabilità (1 = stabile, 0 = instabile)
        stability_score = 1 / (1 + combined_variance * 100)  # Fattore di scala empirico
        
        return float(stability_score)
    
    def finalize_experiment(self) -> Dict:
        """
        Conclude l'esperimento e calcola le metriche finali
        
        Returns:
            Dizionario con tutte le metriche finali
        """
        if not hasattr(self, 'current_experiment'):
            print("⚠️  No active experiment to finalize!")
            return {}
        
        exp = self.current_experiment
        exp["end_time"] = time.time()
        exp["total_time_seconds"] = exp["end_time"] - exp["start_time"]
        exp["total_time_minutes"] = exp["total_time_seconds"] / 60
        
        # Metriche finali
        final_round = max(exp["round_metrics"].keys())
        final_metrics = exp["round_metrics"][final_round]
        
        exp["final_accuracy"] = final_metrics["accuracy"]
        exp["final_loss"] = final_metrics["loss"]
        exp["stability_score"] = self.calculate_stability_score()
        
        # Round per raggiungere 90% (se raggiunto)
        exp["rounds_to_90_percent"] = exp.get("first_90_accuracy_round", None)
        
        # Calcola tasso di convergenza medio
        if len(exp["convergence_history"]) > 1:
            losses = [h["loss"] for h in exp["convergence_history"]]
            rounds = list(range(1, len(losses) + 1))
            # Pendenza della retta di regressione
            slope = np.polyfit(rounds, losses, 1)[0]
            exp["convergence_rate"] = float(-slope)  # Negativo perché vogliamo positivo quando decresce
        else:
            exp["convergence_rate"] = 0.0
        
        # Salva l'esperimento
        self.experiments[exp["group_name"]] = exp
        
        # Aggiungi alla tabella comparativa
        self.add_to_comparison_table(exp)
        
        # Salva tutto su file
        self.save_results()
        
        print(f"\n✅ Experiment {exp['group_name']} completed!")
        print(f"   📈 Final accuracy: {exp['final_accuracy']:.4f}")
        print(f"   📉 Final loss: {exp['final_loss']:.4f}")
        print(f"   ⏱️  Total time: {exp['total_time_minutes']:.1f} minutes")
        
        return exp
    
    def add_to_comparison_table(self, exp: Dict) -> None:
        """Aggiunge l'esperimento alla tabella comparativa"""
        entry = {
            "group_name": exp["group_name"],
            "experiment_id": exp["experiment_id"],
            "nodes": exp["nodes"],
            "rounds": exp["rounds"],
            "epochs": exp["epochs"],
            "final_accuracy": round(exp["final_accuracy"], 4),
            "final_loss": round(exp["final_loss"], 4),
            "rounds_to_90": exp["rounds_to_90_percent"],
            "stability_score": round(exp["stability_score"], 3),
            "convergence_rate": round(exp["convergence_rate"], 6),
            "time_minutes": round(exp["total_time_minutes"], 1)
        }
        
        # Rimuovi entry esistente se presente (per re-run)
        self.comparison_table = [e for e in self.comparison_table 
                                if e["group_name"] != exp["group_name"]]
        
        self.comparison_table.append(entry)
        
        # Ordina per accuracy decrescente
        self.comparison_table.sort(key=lambda x: x["final_accuracy"], reverse=True)
    
    def save_results(self) -> None:
        """Salva tutti i risultati su file"""
        data = {
            "experiments": self.experiments,
            "comparison_table": self.comparison_table,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.comparison_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Salva anche una versione CSV per Excel
        self.save_comparison_csv()
    
    def save_comparison_csv(self) -> None:
        """Salva la tabella comparativa in formato CSV"""
        import csv
        
        csv_file = os.path.join(self.metrics_dir, "configuration_comparison.csv")
        
        if not self.comparison_table:
            return
            
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.comparison_table[0].keys())
            writer.writeheader()
            writer.writerows(self.comparison_table)
    
    def get_best_configuration(self) -> Optional[Dict]:
        """Restituisce la configurazione con la migliore accuracy"""
        if not self.comparison_table:
            return None
        return self.comparison_table[0]  # Già ordinato per accuracy
    
    def get_most_efficient_configuration(self) -> Optional[Dict]:
        """Restituisce la configurazione che raggiunge 90% nel minor numero di round"""
        valid_configs = [c for c in self.comparison_table if c["rounds_to_90"] is not None]
        if not valid_configs:
            return None
        return min(valid_configs, key=lambda x: x["rounds_to_90"])
    
    def print_comparison_report(self) -> str:
        """Genera un report testuale di confronto"""
        if not self.comparison_table:
            return "Nessun esperimento completato ancora."
        
        report = """
╔══════════════════════════════════════════════════════════════════════╗
║              FEDERATED LEARNING CONFIGURATION ANALYSIS               ║
╚══════════════════════════════════════════════════════════════════════╝

📊 COMPARISON TABLE:
"""
        
        # Header
        report += f"\n{'Experiment':<20} {'Nodes':<6} {'Rounds':<7} {'Epochs':<8} {'Accuracy':<10} {'Loss':<8} {'90% at':<8} {'Stability':<11} {'Time(m)':<10}"
        report += f"\n{'-'*20} {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*11} {'-'*10}"
        
        # Righe
        for exp in self.comparison_table[:10]:  # Top 10
            rounds_90 = str(exp['rounds_to_90']) if exp['rounds_to_90'] else "N/A"
            report += f"\n{exp['group_name']:<20} {exp['nodes']:<6} {exp['rounds']:<7} {exp['epochs']:<8} "
            report += f"{exp['final_accuracy']:<10.4f} {exp['final_loss']:<8.4f} {rounds_90:<8} "
            report += f"{exp['stability_score']:<11.3f} {exp['time_minutes']:<10.1f}"
        
        # Best configurations
        best_acc = self.get_best_configuration()
        most_eff = self.get_most_efficient_configuration()
        
        report += "\n\n🏆 OPTIMAL CONFIGURATIONS:"
        
        if best_acc:
            report += f"\n\n📈 Best Accuracy: {best_acc['group_name']}"
            report += f"\n   → Accuracy: {best_acc['final_accuracy']:.4f}"
            report += f"\n   → Configuration: {best_acc['nodes']} nodes, {best_acc['rounds']} rounds, {best_acc['epochs']} epochs"
        
        if most_eff:
            report += f"\n\n⚡ Most Efficient (fastest to 90%): {most_eff['group_name']}"
            report += f"\n   → Reaches 90% at: round {most_eff['rounds_to_90']}"
            report += f"\n   → Configuration: {most_eff['nodes']} nodes, {most_eff['rounds']} rounds, {most_eff['epochs']} epochs"
        
        # Pattern analysis
        report += "\n\n📊 PATTERN ANALYSIS:"
        
        # Effetto del numero di nodi
        nodes_analysis = {}
        for exp in self.comparison_table:
            n = exp['nodes']
            if n not in nodes_analysis:
                nodes_analysis[n] = []
            nodes_analysis[n].append(exp['final_accuracy'])
        
        report += "\n\n• Effect of node count on average accuracy:"
        for nodes, accs in sorted(nodes_analysis.items()):
            avg_acc = np.mean(accs)
            report += f"\n  {nodes} nodes: {avg_acc:.4f} (from {len(accs)} experiments)"
        
        report += "\n\n" + "="*70 + "\n"
        
        return report


# Global singleton for easy access
experiment_tracker = None

def get_experiment_tracker() -> ExperimentMetricsTracker:
    """Returns the singleton instance of the experiment tracker"""
    global experiment_tracker
    if experiment_tracker is None:
        experiment_tracker = ExperimentMetricsTracker()
    return experiment_tracker
