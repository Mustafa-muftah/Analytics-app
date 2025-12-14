from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class NotificationService:
    """
    Sends notifications and alerts
    Future: Integrate with email, SMS, Slack, webhooks
    """
    
    def __init__(self):
        self.enabled = False
        self.channels: List[str] = []
    
    async def send_alert(self, alert_type: str, data: Dict[str, Any]) -> bool:
        """Send an alert notification"""
        if not self.enabled:
            logger.debug(f"Notifications disabled, skipping: {alert_type}")
            return False
        
        logger.info(f"Alert triggered: {alert_type} - {data}")
        return True
    
    async def send_traffic_spike_alert(self, camera_id: str, count: int, threshold: int):
        """Send alert when traffic exceeds threshold"""
        return await self.send_alert("traffic_spike", {
            "camera_id": camera_id,
            "current_count": count,
            "threshold": threshold
        })


# Singleton instance
notification_service = NotificationService()