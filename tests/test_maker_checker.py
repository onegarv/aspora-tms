"""Tests for MakerCheckerWorkflow."""
import pytest


# TODO: test valid proposal enters pending_approval
# TODO: test rejected on unknown nostro
# TODO: test rejected on duplicate idempotency_key
# TODO: test single-checker approval triggers execution
# TODO: test dual-checker: first approval does not execute, second does
# TODO: test self-approval raises PermissionError
# TODO: test same checker cannot provide both dual-checker approvals
# TODO: test auto-escalation fires after timeout
# TODO: test checker without authority raises PermissionError
