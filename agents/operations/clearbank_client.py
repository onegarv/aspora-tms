"""
ClearBank API client — Python port of the Java backoffice implementation.

Reference: /Desktop/clearbank-fund-transfer-api/reference.md
Source parity: src/main/java/com/aspora/ops/backoffice/lib/clearbank/

Three components mirror the Java layout:
  ClearBankConfig      — holds credentials and account IDs
  ClearBankCryptoUtil  — SHA256withRSA signing for DigitalSignature header
  ClearBankApiCommand  — builds headers, executes async HTTP via httpx
  ClearBankClient      — thin wrapper exposing fps_payment / chaps_payment / get_accounts
"""

from __future__ import annotations

import base64
import json
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal

import httpx

logger = logging.getLogger("tms.clearbank")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ClearBankConfig:
    """Mirrors Java ClearBankConfig. All fields required for live mode."""
    token: str
    base_url: str                  # No trailing slash — e.g. https://institution-api-sim.clearbank.co.uk
    private_key: str               # PKCS#8 Base64-encoded RSA private key (PEM headers stripped or kept)
    clearbank_account_id: str      # ClearBank-assigned account IBAN (sourceAccount in CHAPS)
    source_account_id: str         # Debtor account IBAN (debtorAccount)
    legal_name: str                # Debtor legal entity name
    legal_address: str             # Debtor street address (for FPS debtor block)


# ── Crypto ────────────────────────────────────────────────────────────────────

class ClearBankCryptoUtil:
    """
    SHA256withRSA digital signature as required by ClearBank.

    Workflow (mirrors Java ClearBankCryptoUtil):
      1. Serialize body to JSON (same bytes as sent over the wire).
      2. Sign with RSA private key (SHA256withRSA / PKCS1v15 + SHA-256).
      3. Base64-encode the signature.
      4. Place in 'DigitalSignature' header.

    Private key format: PKCS#8, Base64-encoded DER (PEM headers optional).
    """

    def __init__(self, config: ClearBankConfig) -> None:
        self._private_key = self._load_key(config.private_key)

    @staticmethod
    def _load_key(pem_or_b64: str):
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.backends import default_backend

        stripped = (
            pem_or_b64
            .replace("-----BEGIN PRIVATE KEY-----", "")
            .replace("-----END PRIVATE KEY-----", "")
            .replace("-----BEGIN RSA PRIVATE KEY-----", "")
            .replace("-----END RSA PRIVATE KEY-----", "")
            .replace("\n", "")
            .replace("\r", "")
            .strip()
        )
        der_bytes = base64.b64decode(stripped)
        return serialization.load_der_private_key(
            der_bytes, password=None, backend=default_backend()
        )

    def sign(self, body: str) -> str:
        """Return Base64-encoded SHA256withRSA signature of the UTF-8 body."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        sig_bytes = self._private_key.sign(
            body.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        return base64.b64encode(sig_bytes).decode("ascii")


# ── HTTP command ───────────────────────────────────────────────────────────────

class ClearBankError(Exception):
    def __init__(self, status_code: int, path: str, detail: str) -> None:
        self.status_code = status_code
        self.path        = path
        self.detail      = detail
        super().__init__(f"ClearBank HTTP {status_code} on {path}: {detail[:300]}")


class ClearBankApiCommand:
    """
    Builds ClearBank-required headers and executes a single HTTP call.

    Auth headers per reference.md:
      Authorization: Bearer {token}           — all requests
      X-Request-Id:  {UUID no hyphens}        — all requests
      Content-Type:  application/json         — POST/PUT only
      DigitalSignature: {Base64 signature}    — POST/PUT only
    """

    def __init__(self, config: ClearBankConfig, crypto: ClearBankCryptoUtil) -> None:
        self._config = config
        self._crypto = crypto

    def _base_url(self) -> str:
        return self._config.base_url.rstrip("/")

    def _request_id(self) -> str:
        return uuid.uuid4().hex  # UUID without hyphens

    async def post(self, path: str, body: dict) -> dict:
        body_json  = json.dumps(body, separators=(",", ":"))
        request_id = self._request_id()
        headers = {
            "Authorization":    f"Bearer {self._config.token}",
            "X-Request-Id":     request_id,
            "Content-Type":     "application/json",
            "DigitalSignature": self._crypto.sign(body_json),
        }
        url = self._base_url() + path
        logger.debug("clearbank POST %s request_id=%s", path, request_id)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, content=body_json, headers=headers)

        logger.info(
            "clearbank response status=%s path=%s request_id=%s",
            response.status_code, path, request_id,
        )
        if response.status_code not in (200, 202):
            raise ClearBankError(
                status_code=response.status_code,
                path=path,
                detail=response.text,
            )
        return response.json() if response.text.strip() else {}

    async def get(self, path: str) -> dict:
        request_id = self._request_id()
        headers = {
            "Authorization": f"Bearer {self._config.token}",
            "X-Request-Id":  request_id,
        }
        url = self._base_url() + path

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)

        if response.status_code != 200:
            raise ClearBankError(
                status_code=response.status_code,
                path=path,
                detail=response.text,
            )
        return response.json()


# ── Client ────────────────────────────────────────────────────────────────────

class ClearBankClient:
    """
    Thin wrapper around ClearBankApiCommand.

    Methods mirror Java ClearBankClientImpl:
      fps_payment(...)   → POST /v3/payments/fps
      chaps_payment(...) → POST /payments/chaps/v5/customer-payments
      get_accounts()     → GET  /v3/Accounts
    """

    FPS_PATH      = "/v3/payments/fps"
    CHAPS_PATH    = "/payments/chaps/v5/customer-payments"
    ACCOUNTS_PATH = "/v3/Accounts"

    def __init__(self, config: ClearBankConfig) -> None:
        self._config = config
        self._crypto = ClearBankCryptoUtil(config)
        self._cmd    = ClearBankApiCommand(config, self._crypto)

    async def fps_payment(
        self,
        *,
        instruction_id: str,       # max 35, alphanumeric + hyphens (idempotency key)
        end_to_end_id: str,        # max 35
        amount: Decimal,
        creditor_name: str,
        creditor_account_iban: str,
        remittance_info: str = "",
    ) -> dict:
        """
        POST /v3/payments/fps — Faster Payments (up to £1,000,000).
        Returns ClearbankFpsPaymentRes: { transactions: [...], halLinks: [...] }
        """
        body: dict = {
            "paymentInstructions": [
                {
                    "paymentInstructionIdentification": instruction_id,
                    "paymentTypeCode": "SIP",
                    "debtor": {
                        "name":    self._config.legal_name,
                        "address": self._config.legal_address,
                    },
                    "debtorAccount": {
                        "identification": {"iban": self._config.source_account_id}
                    },
                    "creditTransfers": [
                        {
                            "paymentIdentification": {
                                "instructionIdentification": instruction_id,
                                "endToEndIdentification":   end_to_end_id,
                            },
                            "amount": {
                                "value":    str(amount),
                                "currency": "GBP",
                            },
                            "creditor": {"name": creditor_name},
                            "creditorAccount": {
                                "identification": {"iban": creditor_account_iban}
                            },
                            **(
                                {"remittanceInformation": {"unstructured": remittance_info}}
                                if remittance_info else {}
                            ),
                        }
                    ],
                }
            ]
        }
        return await self._cmd.post(self.FPS_PATH, body)

    async def chaps_payment(
        self,
        *,
        instruction_id: str,       # max 35, alphanumeric + hyphens
        end_to_end_id: str,        # max 35
        amount: Decimal,
        creditor_name: str,
        creditor_account_iban: str,
        purpose: str = "INTC",          # exactly 4 uppercase letters
        category_purpose: str = "CASH", # 2-4 uppercase letters
        remittance_info: str = "",
    ) -> dict:
        """
        POST /payments/chaps/v5/customer-payments — CHAPS high-value (no limit).
        Returns ClearbankChapPaymentRes: { instructionIdentification, paymentId, reason }
        """
        body: dict = {
            "instructionIdentification": instruction_id,
            "endToEndIdentification":    end_to_end_id,
            "instructedAmount": {
                "amount":   str(amount),
                "currency": "GBP",
            },
            "sourceAccount":  {"iban": self._config.clearbank_account_id},
            "debtorAccount":  {"iban": self._config.source_account_id},
            "debtor":         {"name": self._config.legal_name},
            "creditorAccount": {"iban": creditor_account_iban},
            "creditor":        {"name": creditor_name},
            "purpose":         purpose,
            "categoryPurpose": category_purpose,
            **(
                {"remittanceInformation": {"unstructured": remittance_info}}
                if remittance_info else {}
            ),
        }
        return await self._cmd.post(self.CHAPS_PATH, body)

    async def get_accounts(self) -> dict:
        """GET /v3/Accounts — returns AccountsResponse with list of ClearbankAccount."""
        return await self._cmd.get(self.ACCOUNTS_PATH)
