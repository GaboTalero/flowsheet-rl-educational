# Security policy

This repository is an educational research prototype, not production process-control software.

Please report a suspected security vulnerability privately through the repository owner’s GitHub profile rather than opening a public issue. Include the affected version, reproduction steps, and potential impact. Do not include secrets or sensitive process data in reports, examples, checkpoints, or logs.

Only the latest tagged release is intended to receive security fixes. PyTorch checkpoints use serialization mechanisms that can execute code when loaded; load checkpoints only from sources you trust.
