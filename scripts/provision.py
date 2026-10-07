"""
Tenant and API key provisioning CLI script.

This script is the ONLY way to create tenants and issue API keys in Milestone 1.
The Admin HTTP API is introduced in a later control-plane phase.

Usage:
    # Create a tenant and get an API key
    python scripts/provision.py --name "my-app"

    # Create a tenant with a custom label for the key
    python scripts/provision.py --name "my-app" --key-label "dev-laptop"

    # List all tenants
    python scripts/provision.py --list

The raw API key is printed ONCE and never stored.
Store it securely — it cannot be retrieved again.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import secrets
import sys
import uuid

# Ensure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sqlalchemy import select, text

from coalai.db.models import ApiKey, Tenant
from coalai.db.session import close_db, get_session, init_db


def _generate_raw_key() -> str:
    """Generate a new API key in the coal_sk_<32-bytes-base64url> format."""
    random_bytes = secrets.token_urlsafe(32)
    return f"coal_sk_{random_bytes}"


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


async def cmd_create(name: str, key_label: str | None, rpm: int, rpd: int) -> None:
    """Create a tenant and issue one API key."""
    raw_key = _generate_raw_key()
    key_hash = _hash_key(raw_key)
    tenant_id = uuid.uuid4()
    key_id = uuid.uuid4()

    async with get_session() as session:
        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            status="active",
        )
        api_key = ApiKey(
            key_id=key_id,
            tenant_id=tenant_id,
            key_hash=key_hash,
            label=key_label or "default",
            requests_per_minute=rpm,
            requests_per_day=rpd,
        )
        session.add(tenant)
        session.add(api_key)

    print("\n✅  Tenant created successfully")
    print(f"   Tenant ID : {tenant_id}")
    print(f"   Name      : {name}")
    print(f"   Key ID    : {key_id}")
    print(f"   Key Label : {key_label or 'default'}")
    print(f"   RPM limit : {rpm}")
    print(f"   RPD limit : {rpd}")
    print()
    print("🔑  API Key (shown ONCE — store this securely):")
    print(f"\n   {raw_key}\n")
    print("   Authorization header:  Bearer " + raw_key)
    print()


async def cmd_list() -> None:
    """List all tenants."""
    async with get_session() as session:
        result = await session.execute(
            select(Tenant).order_by(Tenant.created_at)
        )
        tenants = result.scalars().all()

    if not tenants:
        print("No tenants found.")
        return

    print(f"\n{'Tenant ID':<38} {'Name':<30} {'Status'}")
    print("-" * 80)
    for t in tenants:
        print(f"{str(t.tenant_id):<38} {t.name:<30} {t.status}")
    print()


async def cmd_issue_key(tenant_id_str: str, label: str | None, rpm: int, rpd: int) -> None:
    """Issue a new API key for an existing tenant."""
    try:
        tenant_id = uuid.UUID(tenant_id_str)
    except ValueError:
        print(f"Error: '{tenant_id_str}' is not a valid UUID.", file=sys.stderr)
        sys.exit(1)

    raw_key = _generate_raw_key()
    key_hash = _hash_key(raw_key)
    key_id = uuid.uuid4()

    async with get_session() as session:
        # Verify tenant exists
        result = await session.execute(select(Tenant).where(Tenant.tenant_id == tenant_id))
        tenant = result.scalar_one_or_none()
        if tenant is None:
            print(f"Error: Tenant {tenant_id} not found.", file=sys.stderr)
            sys.exit(1)

        api_key = ApiKey(
            key_id=key_id,
            tenant_id=tenant_id,
            key_hash=key_hash,
            label=label or "issued",
            requests_per_minute=rpm,
            requests_per_day=rpd,
        )
        session.add(api_key)

    print(f"\n✅  API key issued for tenant {tenant_id}")
    print(f"   Key ID : {key_id}")
    print()
    print("🔑  API Key (shown ONCE — store this securely):")
    print(f"\n   {raw_key}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="COALAI provisioning CLI — create tenants and issue API keys."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # coalai-provision create
    create_p = sub.add_parser("create", help="Create a tenant and issue an API key")
    create_p.add_argument("--name", required=True, help="Tenant name")
    create_p.add_argument("--key-label", default=None, help="Label for the API key")
    create_p.add_argument("--rpm", type=int, default=60, help="Requests per minute limit")
    create_p.add_argument("--rpd", type=int, default=10000, help="Requests per day limit")

    # coalai-provision list
    sub.add_parser("list", help="List all tenants")

    # coalai-provision issue-key
    issue_p = sub.add_parser("issue-key", help="Issue a new API key for an existing tenant")
    issue_p.add_argument("--tenant-id", required=True, help="Tenant UUID")
    issue_p.add_argument("--label", default=None, help="Key label")
    issue_p.add_argument("--rpm", type=int, default=60)
    issue_p.add_argument("--rpd", type=int, default=10000)

    args = parser.parse_args()

    # Load .env if present
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    init_db()

    if args.command == "create":
        asyncio.run(cmd_create(args.name, args.key_label, args.rpm, args.rpd))
    elif args.command == "list":
        asyncio.run(cmd_list())
    elif args.command == "issue-key":
        asyncio.run(cmd_issue_key(args.tenant_id, args.label, args.rpm, args.rpd))

    asyncio.run(close_db())


if __name__ == "__main__":
    main()
