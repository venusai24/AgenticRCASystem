import os
import subprocess
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any

from agentic_rca.handoff.adapters import AlertPayload, TelemetryAdapter

logger = logging.getLogger(__name__)

class RunOrchestrator:
    """
    Coordinates the fetching of telemetry data, formatting it into the expected CSV schemas,
    and triggering the AgenticRCA run.
    """
    
    def __init__(self, telemetry_adapter: TelemetryAdapter, base_data_dir: str = "incident_data"):
        self.telemetry_adapter = telemetry_adapter
        self.base_data_dir = Path(base_data_dir)
        
    async def process_alert(self, payload: AlertPayload):
        """
        The main workflow triggered after the Wait Queue delay.
        """
        incident_id = f"incident_{payload.incident_ts}"
        incident_dir = self.base_data_dir / incident_id
        
        # 1. Fetch Data via Adapter
        logger.info(f"Fetching telemetry for incident {incident_id}...")
        metrics_data = self.telemetry_adapter.fetch_metrics(payload.incident_ts)
        logs_data = self.telemetry_adapter.fetch_logs(payload.incident_ts)
        traces_data = self.telemetry_adapter.fetch_traces(payload.incident_ts)
        
        # 2. Format and write CSVs
        self._write_csvs(incident_dir, metrics_data, logs_data, traces_data)
        
        # 3. Trigger AgenticRCA subprocess
        self._trigger_agentic_rca(incident_dir, payload)

    def _write_csvs(self, incident_dir: Path, metrics: Dict[str, Any], logs: Dict[str, Any], traces: Dict[str, Any]):
        """
        Writes the normalized dictionary data into the 6 specific CSV files required by AgenticRCA.
        (Mocked behavior for the wrapper architecture)
        """
        incident_dir.mkdir(parents=True, exist_ok=True)
        
        # Example of writing a mock dataframe to match SchemaOfCSVs.MD
        # In reality, this would map the adapter's dictionary output to pandas DataFrames
        logger.info(f"Writing strictly formatted CSVs to {incident_dir}")
        
        # Write dummy files just to satisfy the orchestrator structure
        # baseline_app_metrics, cluster_app_metrics, etc.
        pd.DataFrame(columns=["timestamp", "tc", "rr", "sr", "cnt", "mrt"]).to_csv(
            incident_dir / "baseline_app_metrics.csv", index=False
        )
        pd.DataFrame(columns=["timestamp", "tc", "rr", "sr", "cnt", "mrt"]).to_csv(
            incident_dir / "cluster_app_metrics.csv", index=False
        )
        pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"]).to_csv(
            incident_dir / "baseline_container_metrics.csv", index=False
        )
        pd.DataFrame(columns=["timestamp", "cmdb_id", "kpi_name", "value"]).to_csv(
            incident_dir / "container_metrics.csv", index=False
        )
        pd.DataFrame(columns=["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"]).to_csv(
            incident_dir / "incident_traces.csv", index=False
        )
        pd.DataFrame(columns=["log_id", "timestamp", "cmdb_id", "log_name", "value"]).to_csv(
            incident_dir / "cluster_incident_logs.csv", index=False
        )

    def _trigger_agentic_rca(self, incident_dir: Path, payload: AlertPayload):
        """
        Forks the AgenticRCA process.
        """
        cmd = [
            "python", "-m", "agentic_rca", "run",
            "--data-dir", str(incident_dir),
            "--incident-ts", str(payload.incident_ts)
        ]
        
        if payload.human_hypothesis:
            cmd.extend(["--human-hypothesis", payload.human_hypothesis])
            
        logger.info(f"Triggering AgenticRCA: {' '.join(cmd)}")
        
        # Run subprocess (in a real service, might want to capture stdout/stderr properly)
        try:
            # We use Popen so it runs in the background, or run if we want to block and await results.
            # We'll block here to log the completion.
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.info(f"AgenticRCA run completed successfully for {incident_dir.name}")
            else:
                logger.error(f"AgenticRCA run failed for {incident_dir.name}: {result.stderr}")
        except Exception as e:
            logger.error(f"Failed to launch AgenticRCA process: {e}")
