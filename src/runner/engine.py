from contextlib import asynccontextmanager, contextmanager
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.postgres import PostgresSaver

from flow.builder import a_build_graph, build_graph
from constants import DB_URI, DB_POOL_MAX_SIZE, DB_CONNECTION_KWARGS


@asynccontextmanager
async def get_async_app():
    async with AsyncConnectionPool(
        conninfo=DB_URI,
        max_size=DB_POOL_MAX_SIZE,
        kwargs=DB_CONNECTION_KWARGS,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        workflow = a_build_graph()
        app = workflow.compile(checkpointer=checkpointer)
        yield app

@contextmanager
def get_sync_app():
    with ConnectionPool(
        conninfo=DB_URI,
        max_size=DB_POOL_MAX_SIZE,
        kwargs=DB_CONNECTION_KWARGS,
    ) as pool:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        workflow = build_graph()
        app = workflow.compile(checkpointer=checkpointer)
        yield app
