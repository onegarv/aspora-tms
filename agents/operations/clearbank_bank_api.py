"""
ClearBankBankAPI — implements the abstract BankAPI interface using ClearBank FPS/CHAPS.

Rail routing:
  "fps"   → POST /v3/payments/fps         (GBP, up to £1,000,000)
  "chaps" → POST /payments/chaps/v5/...   (GBP, no upper limit; UK business hours)

Only handles GBP. All other currencies must use separate BankAPI adapters.

Idempotency fence:
  An in-process dict maps instruction_id → bank_ref so that retries within a
  single process don't resubmit. In production, pair this with a DB-backed fence
  that survives process restarts.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from agents.operations.clearbank_client import ClearBankClient, ClearBankError
from agents.operations.fund_mover import BankAPI

logger = logging.getLogger("tms.clearbank.bank_api")

# FPS scheme limit per ClearBank docs
_FPS_LIMIT = Decimal("1000000")


class ClearBankBankAPI(BankAPI):
    """
    Plugs into FundMover as the bank_api for GBP transfers.

    nostro_map maps destination_nostro account IDs (as used inside the TMS)
    to ClearBank-side IBAN and name:
      { "NOSTRO-GBP-001": { "iban": "GB00CLRB...", "name": "Aspora GBP Nostro" } }
    """

    def __init__(
        self,
        client: ClearBankClient,
        nostro_map: dict[str, dict] | None = None,
    ) -> None:
        self._client     = client
        self._nostro_map = nostro_map or {}
        # In-memory idempotency fence: instruction_id → bank_ref
        self._submitted: dict[str, str] = {}

    async def submit_transfer(
        self,
        *,
        instruction_id:     str,
        amount:             Decimal,
        currency:           str,
        rail:               str,
        source_account:     str,
        destination_nostro: str,
    ) -> str:
        """
        Submit a GBP transfer via FPS or CHAPS.

        Returns the ClearBank payment ID (bank_ref) on success.
        Raises ValueError for unsupported currency/rail.
        Raises ClearBankError on HTTP failure.
        """
        if currency.upper() != "GBP":
            raise ValueError(
                f"ClearBankBankAPI only handles GBP, got {currency}. "
                "Use a different BankAPI adapter for non-GBP rails."
            )

        nostro_info = self._nostro_map.get(destination_nostro)
        if not nostro_info:
            raise ValueError(
                f"No IBAN mapping found for nostro '{destination_nostro}'. "
                "Add an entry to ClearBankBankAPI nostro_map."
            )

        creditor_iban = nostro_info["iban"]
        creditor_name = nostro_info.get("name", "Aspora GBP Nostro")
        end_to_end_id = f"E2E{instruction_id}"[:35]
        remittance    = f"Prefund {instruction_id}"[:140]

        try:
            if rail == "fps":
                if amount > _FPS_LIMIT:
                    raise ValueError(
                        f"Amount {amount} GBP exceeds FPS limit of {_FPS_LIMIT}. "
                        "Use CHAPS rail for amounts above £1,000,000."
                    )
                resp = await self._client.fps_payment(
                    instruction_id=instruction_id[:35],
                    end_to_end_id=end_to_end_id,
                    amount=amount,
                    creditor_name=creditor_name,
                    creditor_account_iban=creditor_iban,
                    remittance_info=remittance,
                )
                # FPS response shape: { transactions: [{ paymentId: "..." }], halLinks: [...] }
                try:
                    bank_ref = resp["transactions"][0]["paymentId"]
                except (KeyError, IndexError):
                    bank_ref = instruction_id  # fallback if shape unexpected

            elif rail == "chaps":
                resp = await self._client.chaps_payment(
                    instruction_id=instruction_id[:35],
                    end_to_end_id=end_to_end_id,
                    amount=amount,
                    creditor_name=creditor_name,
                    creditor_account_iban=creditor_iban,
                    purpose="INTC",
                    category_purpose="CASH",
                    remittance_info=remittance,
                )
                # CHAPS response shape: { paymentId: "...", instructionIdentification: "..." }
                bank_ref = resp.get("paymentId", instruction_id)

            else:
                raise ValueError(
                    f"ClearBankBankAPI does not support rail '{rail}'. "
                    "Supported rails: 'fps', 'chaps'."
                )

        except ClearBankError:
            # Re-raise so FundMover state machine records SUBMIT_UNKNOWN
            raise

        self._submitted[instruction_id] = bank_ref
        logger.info(
            "clearbank transfer submitted: instruction=%s rail=%s amount=%s bank_ref=%s",
            instruction_id, rail, amount, bank_ref,
        )
        return bank_ref

    async def get_transfer_status(self, bank_ref: str) -> dict[str, Any]:
        """
        Poll for settlement status.

        ClearBank sandbox confirms synchronously, so we report settled=True
        immediately. In production, wire this to ClearBank webhooks or the
        Accounts balance endpoint to confirm actual settlement.
        """
        return {
            "bank_ref":       bank_ref,
            "settled":        True,   # sandbox: ClearBank confirms on submission
            "settled_amount": None,   # FundMover fills from proposal.amount
        }

    async def find_by_instruction_id(self, instruction_id: str) -> str | None:
        """
        Idempotency fence: return bank_ref if already submitted, else None.

        In production, query ClearBank (or a DB) to survive process restarts.
        """
        return self._submitted.get(instruction_id)
