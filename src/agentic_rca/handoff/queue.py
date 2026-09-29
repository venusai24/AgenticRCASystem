import asyncio
import logging
from typing import Callable, Coroutine, Any

logger = logging.getLogger(__name__)

class WaitQueue:
    """
    A lightweight queue to delay processing of incoming webhooks.
    This ensures that telemetry for the incident window has fully propagated 
    to observability systems (Datadog, Jaeger, etc.) before we attempt to fetch it.
    """
    
    def __init__(self, wait_time_seconds: int = 300):
        self.wait_time_seconds = wait_time_seconds
        
    async def schedule_processing(self, payload: Any, processor_func: Callable[[Any], Coroutine[Any, Any, None]]):
        """
        Schedules the processing of an alert payload after the configured wait time.
        In a production environment, this would ideally push to a distributed queue like Celery or SQS.
        For this lightweight service, we use asyncio to manage the delay.
        """
        logger.info(f"Alert received. Waiting {self.wait_time_seconds} seconds before fetching telemetry...")
        
        # We spawn a background task to wait and then process
        asyncio.create_task(self._wait_and_process(payload, processor_func))
        
    async def _wait_and_process(self, payload: Any, processor_func: Callable[[Any], Coroutine[Any, Any, None]]):
        try:
            await asyncio.sleep(self.wait_time_seconds)
            logger.info("Wait time complete. Triggering orchestrator for payload.")
            await processor_func(payload)
        except Exception as e:
            logger.error(f"Failed to process alert after wait: {e}")
