import os
import os.path
import pickle
from pathlib import Path
from typing import Any, Union, Iterator, Optional, Sequence, Tuple, List, Generic, \
    TypeVar, Dict
from urllib.parse import urlparse

from langchain.schema import BaseStore
from langchain.storage import EncoderBackedStore, LocalFileStore
from sqlalchemy import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

K = TypeVar("K")
V = TypeVar("V")
class SQLStore(BaseStore[K,V]):
    """ Simulation of SQL Store"""
    def __init__(self,
                 *,
                 engine: Optional[Union[Engine, AsyncEngine]] = None,
                 db_url: Union[str, Path],
                 engine_kwargs: Optional[Dict[str, Any]] = None,
                 async_mode: bool = False,
                 ):
        parsed = urlparse(db_url)
        if parsed.scheme != 'sqlite':
            raise ValueError("db_url must be in form sqllite://...")
        root_path,_ = os.path.splitext(parsed.path)
        root_path+=".ds"
        self._delegate = EncoderBackedStore[str, Any](
            store=LocalFileStore(root_path=root_path),
            key_encoder=lambda x: x,
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads
        )

    def mget(self, keys: Sequence[K]) -> List[Optional[V]]:
        return self._delegate.mget(keys)
    def mset(self, key_value_pairs: Sequence[Tuple[K, V]]) -> None:
        self._delegate.mset(key_value_pairs)
    def mdelete(self, keys: Sequence[K]) -> None:
        self._delegate.mdelete(keys)
    def yield_keys(
            self, *, prefix: Optional[str] = None
    ) -> Union[Iterator[K], Iterator[str]]:
        return self._delegate.yield_keys(prefix=prefix)

    def create_schema(self) -> None:
        pass
