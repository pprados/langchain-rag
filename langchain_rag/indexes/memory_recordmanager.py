from datetime import datetime
from typing import List, Optional, Sequence

from langchain.indexes.base import RecordManager


class MemoryRecordManager(RecordManager):
    def __init__(self, namespace: str):
        super().__init__(namespace=namespace)

    def create_schema(self) -> None:
        """Create the database schema for the record manager."""
        pass

    async def acreate_schema(self) -> None:
        """Create the database schema for the record manager."""
        pass

    def get_time(self) -> float:
        """Get the current server time as a high resolution timestamp!

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time as a float timestamp.
        """
        return datetime.now().timestamp()

    async def aget_time(self) -> float:
        """Get the current server time as a high resolution timestamp!

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time as a float timestamp.
        """
        return datetime.now().timestamp()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: if provided, updates should only happen if the
              updated_at field is at least this time.

        Raises:
            ValueError: If the length of keys doesn't match the length of group_ids.
        """
        pass

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Optional[Sequence[Optional[str]]] = None,
        time_at_least: Optional[float] = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: if provided, updates should only happen if the
              updated_at field is at least this time.

        Raises:
            ValueError: If the length of keys doesn't match the length of group_ids.
        """
        pass

    def exists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """
        return [False] * len(keys)

    async def aexists(self, keys: Sequence[str]) -> List[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        """
        return [False] * len(keys)

    def list_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.
            limit: optional limit on the number of records to return.

        Returns:
            A list of keys for the matching records.
        """
        return []

    async def alist_keys(
        self,
        *,
        before: Optional[float] = None,
        after: Optional[float] = None,
        group_ids: Optional[Sequence[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.
            limit: optional limit on the number of records to return.

        Returns:
            A list of keys for the matching records.
        """
        return []

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """
        pass

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        """
        pass
