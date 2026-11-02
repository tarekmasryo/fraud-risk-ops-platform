# Worker crash recovery limitation

The current worker atomically claims a `queued` job as `running`. If the process dies after the claim, restarting it does not reclaim that row: both claim paths select only `queued` jobs. Redis wake-up signaling does not recover an already-running job.

This limitation is outside dependency maintenance. For a local recovery, stop competing workers, back up the SQLite store, identify the affected job and inspect prediction/audit records before deciding whether to mark it failed or submit a replacement. Preserve the original job and evidence. Do not blindly reset all running jobs to queued: scoring may have persisted predictions before the crash, so replay can duplicate records.

A future recovery change needs explicit ownership/leases, crash-point tests, and a transaction or idempotency design covering prediction persistence and terminal job updates. It must distinguish stale ownership from a slow live worker. No automatic retry or schema change is introduced by this maintenance effort.
