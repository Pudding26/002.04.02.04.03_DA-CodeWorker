from datetime import datetime, timedelta, timezone

class RetryInfo:
    @staticmethod
    def compute_next_retry(
        attempts: int,
        baseline: datetime,
        penalty_step: int = 1,
        backoff: float = 1.5
    ) -> datetime:
        """
        Calculate the next retry time using exponential backoff.
        
        Formula:
        delay = penalty_step × (backoff ^ attempts)
        next_retry = baseline + delay

        All datetimes are assumed to be timezone-aware (UTC).
        """
        penalty = penalty_step * attempts
        delay_seconds = int(penalty * (backoff ** attempts))
        return baseline + timedelta(seconds=delay_seconds)
