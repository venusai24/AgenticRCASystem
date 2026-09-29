"""
Normalized plugin interfaces (adapters) for Alerting Sources and Telemetry Sources.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

class AlertPayload(BaseModel):
    """Normalized alert payload extracted from an incoming webhook."""
    incident_ts: int
    source_name: str
    affected_services: List[str]
    human_hypothesis: Optional[str] = None
    raw_payload: Dict[str, Any]

class AlertingAdapter(ABC):
    """Plugin interface for incoming webhooks (e.g., PagerDuty, Alertmanager)."""
    
    @abstractmethod
    def can_handle(self, source_name: str) -> bool:
        """Return True if this adapter can handle the given source."""
        pass
        
    @abstractmethod
    def parse_payload(self, raw_data: Dict[str, Any]) -> AlertPayload:
        """Extract standard fields from the source-specific webhook payload."""
        pass

class TelemetryAdapter(ABC):
    """Plugin interface for fetching data from observability stacks."""
    
    @abstractmethod
    def fetch_metrics(self, incident_ts: int, window_minutes: int = 30) -> Dict[str, Any]:
        """Fetch metrics and return data formatted for AgenticRCA CSVs."""
        pass
        
    @abstractmethod
    def fetch_logs(self, incident_ts: int, window_minutes: int = 30) -> Dict[str, Any]:
        """Fetch logs and return data formatted for AgenticRCA CSVs."""
        pass
        
    @abstractmethod
    def fetch_traces(self, incident_ts: int, window_minutes: int = 30) -> Dict[str, Any]:
        """Fetch traces (with ms timestamps) and return data formatted for AgenticRCA CSVs."""
        pass

# --- Mock Implementations for Demonstration ---

class PagerDutyAdapter(AlertingAdapter):
    def can_handle(self, source_name: str) -> bool:
        return source_name.lower() == "pagerduty"
        
    def parse_payload(self, raw_data: Dict[str, Any]) -> AlertPayload:
        # Example extraction from a PagerDuty-like payload
        incident_ts = raw_data.get("created_at_epoch", 0)
        description = raw_data.get("description", "")
        return AlertPayload(
            incident_ts=incident_ts,
            source_name="pagerduty",
            affected_services=raw_data.get("services", []),
            human_hypothesis=f"Initial alert description: {description}",
            raw_payload=raw_data
        )

class DatadogAdapter(TelemetryAdapter):
    def fetch_metrics(self, incident_ts: int, window_minutes: int = 30) -> Dict[str, Any]:
        # Simulate fetching and normalizing 60s/120s cadence metrics
        return {"status": "mock_success", "source": "datadog_metrics"}
        
    def fetch_logs(self, incident_ts: int, window_minutes: int = 30) -> Dict[str, Any]:
        return {"status": "mock_success", "source": "datadog_logs"}
        
    def fetch_traces(self, incident_ts: int, window_minutes: int = 30) -> Dict[str, Any]:
        return {"status": "mock_success", "source": "datadog_traces"}
