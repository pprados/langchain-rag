import asyncio
import contextlib
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from langchain_core.stores import BaseStore
from sqlalchemy import Column, Engine, PickleType, and_, create_engine, delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    declarative_base,
    mapped_column,
    scoped_session,
    sessionmaker,
)

Base = declarative_base()


def items_equal(x: Any, y: Any) -> bool:
    return x == y


class Value(Base):  # type: ignore[valid-type,misc]
    """Table used to save values."""

    # ATTENTION:
    # Prior to modifying this table, please determine whether
    # we should create migrations for this table to make sure
    # users do not experience data loss.
    __tablename__ = "docstore"

    namespace: Mapped[str] = mapped_column(primary_key=True, index=True, nullable=False)
    key: Mapped[str] = mapped_column(primary_key=True, index=True, nullable=False)
    # value: Mapped[Any] = Column(type_=PickleType, index=False, nullable=False)
    value: Any = Column("earthquake", PickleType(comparator=items_equal))


# This is a fix of original SQLStore.
# This can will be removed when a PR will be merged.
class SQLStore(BaseStore[str, bytes]):
    """BaseStore interface that works on an SQL database.

    Examples:
        Create a SQLStore instance and perform operations on it:

        .. code-block:: python

            from langchain_rag.storage import SQLStore

            # Instantiate the SQLStore with the root path
            sql_store = SQLStore(namespace="test", db_url="sqllite://:memory:")

            # Set values for keys
            sql_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = sql_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            sql_store.mdelete(["key1"])

            # Iterate over keys
            for key in sql_store.yield_keys():
                print(key)

    """

    def __init__(
        self,
        *,
        namespace: str,
        db_url: Optional[Union[str, Path]] = None,
        engine: Optional[Union[Engine, AsyncEngine]] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        async_mode: bool = False,
    ):
        if db_url is None and engine is None:
            raise ValueError("Must specify either db_url or engine")

        if db_url is not None and engine is not None:
            raise ValueError("Must specify either db_url or engine, not both")

        _engine: Union[Engine, AsyncEngine]
        if db_url:
            if async_mode:
                _engine = create_async_engine(url=str(db_url), **(engine_kwargs or {}))
            else:
                _engine = create_engine(url=str(db_url), **(engine_kwargs or {}))
        elif engine:
            _engine = engine

        else:
            raise AssertionError("Something went wrong with configuration of engine.")

        _session_factory: Union[sessionmaker[Session], async_sessionmaker[AsyncSession]]
        if isinstance(_engine, AsyncEngine):
            _session_factory = async_sessionmaker(bind=_engine)
        else:
            _session_factory = sessionmaker(bind=_engine)

        self.engine = _engine
        self.dialect = _engine.dialect.name
        self.session_factory = _session_factory
        self.namespace = namespace

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    async def acreate_schema(self) -> None:
        """Create the database schema."""

        if not isinstance(self.engine, AsyncEngine):
            raise AssertionError("This method is not supported for sync engines.")

        async with self.engine.begin() as session:
            # print("*********************************")
            # asyncio.set_event_loop(asyncio.new_event_loop())
            # print(f"{id(asyncio.get_event_loop())=}")
            # print(f"{asyncio.get_event_loop().is_running()=}")

            await session.run_sync(Base.metadata.create_all)

    def drop(self) -> None:
        Base.metadata.drop_all(bind=self.engine.connect())

    async def amget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        if not isinstance(self.engine, AsyncEngine):
            raise AssertionError("This method is not supported for sync engines.")
        result = {}
        async with self._amake_session() as session:
            async with session.begin():
                q = select(Value).filter(
                    and_(
                        Value.key.in_(keys),
                        Value.namespace == self.namespace,
                    )
                )
                for v in (await session.execute(q)).scalars():
                    result[v.key] = v.value
        return [result.get(key) for key in keys]

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        result = {}

        with self._make_session() as session:
            for v in session.query(Value).filter(  # type: ignore
                and_(
                    Value.key.in_(keys),
                    Value.namespace == self.namespace,
                )
            ):
                result[v.key] = v.value
        return [result.get(key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        # try:
        values: Dict[str, bytes] = dict(key_value_pairs)
        with self._make_session() as session:
            self._mdelete(list(values.keys()), session)
            session.add_all(
                [
                    Value(namespace=self.namespace, key=k, value=v)
                    for k, v in values.items()
                ]
            )

    async def amset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        if not isinstance(self.engine, AsyncEngine):
            raise AssertionError("This method is not supported for sync engines.")
        async with self._amake_session() as session:
            async with session.begin():
                # await self._amdetete([key for key, _ in key_value_pairs], session)
                session.add_all(
                    [
                        Value(namespace=self.namespace, key=k, value=v)
                        for k, v in key_value_pairs
                    ]
                )
                await session.commit()

    def _mdelete(self, keys: Sequence[str], session: Session) -> None:
        with session.begin():
            session.query(Value).filter(  # type: ignore
                and_(
                    Value.key.in_(keys),
                    Value.namespace == self.namespace,
                )
            ).delete()

    async def _amdelete(self, keys: Sequence[str], session: Session) -> None:
        async with self._amake_session() as session:
            async with session.begin():
                delete(Value).filter(
                    and_(
                        Value.key.in_(keys),
                        Value.namespace == self.namespace,
                    )
                )

    def mdelete(self, keys: Sequence[str]) -> None:
        with self._make_session() as session:
            self._mdelete(keys, session)
            session.commit()

    async def amdelete(self, keys: Sequence[str]) -> None:
        if not isinstance(self.engine, AsyncEngine):
            raise AssertionError("This method is not supported for sync engines.")
        async with self._amake_session() as session:
            await self._amdelete(keys, session)
            await session.commit()

    def yield_keys(self, *, prefix: Optional[str] = None) -> Iterator[str]:
        with self._make_session() as session:
            if prefix:
                clause = and_(
                    Value.key.like(f"{prefix}%"),
                    Value.namespace == self.namespace,
                )
            else:
                clause = Value.namespace == self.namespace
            for v in session.query(Value).filter(clause):
                yield str(v.key)
            session.close()

    async def ayield_keys(self, *, prefix: Optional[str] = None) -> AsyncIterator[str]:
        if not isinstance(self.engine, AsyncEngine):
            raise AssertionError("This method is not supported for sync engines.")
        async with self._amake_session() as session:
            async with session.begin():
                try:
                    if prefix:
                        clause = and_(
                            Value.key.like(f"{prefix}%"),
                            Value.namespace == self.namespace,
                        )
                    else:
                        clause = Value.namespace == self.namespace
                    q = select(Value).filter(clause)
                    for v in (await session.execute(q)).scalars():
                        yield str(v.key)
                    await session.close()
                except:
                    await session.rollback()
                    raise

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """Create a session and close it after use."""

        if isinstance(self.session_factory, async_sessionmaker):
            raise AssertionError("This method is not supported for async engines.")

        session = scoped_session(self.session_factory)()
        try:
            yield session
        finally:
            session.commit()

    @contextlib.asynccontextmanager
    async def _amake_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Create a session and close it after use."""

        if not isinstance(self.session_factory, async_sessionmaker):
            raise AssertionError("This method is not supported for sync engines.")

        async with self.session_factory() as session:
            yield session
