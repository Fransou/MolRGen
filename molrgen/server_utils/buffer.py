import asyncio
import logging
from collections import deque
from typing import Any, List, Literal, Tuple

import ray

from molrgen.reward.molecular_verifier_pydantic_model import (
    MolecularVerifierOutputMetadataModel,
)
from molrgen.server_utils.utils import (
    BatchMolecularVerifierServerResponse,
    MolecularVerifierServerMetadata,
    MolecularVerifierServerQuery,
    MolecularVerifierServerResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("molecular_verifier_buffer")
logger.setLevel(logging.INFO)


class RewardBuffer:
    def __init__(
        self,
        app: Any,
        buffer_time: float = 1.0,
        max_batch_size: int = 1024,
        server_mode: Literal["singleton", "batch"] = "singleton",
    ) -> None:
        """Asynchronous buffer for batching and processing reward scoring requests.

        This class implements a request buffering system that collects incoming molecular
        verifier queries and processes them in optimized batches to maximize throughput
        and reduce latency. It uses asyncio for non-blocking I/O and Ray for distributed
        computation of reward scores.

        The buffering strategy works as follows:

        1. Requests are added to an asyncio queue without blocking the caller
        2. A background task periodically flushes the queue or when it reaches max size
        3. Multiple queries are merged into a single batch with aligned completions/metadata
        4. Results are computed in parallel using Ray actors
        5. Results are grouped back to individual queries and returned asynchronously

        This approach provides several benefits:

        - Amortizes overhead across multiple requests
        - Allows Ray to optimize kernel launches on GPU
        - Reduces context switching and memory fragmentation

        Attributes:
            app (Any): FastAPI/Starlette application instance containing Ray actors in
                app.state.reward_model for distributed reward computation.

            buffer_time (float): Maximum time in seconds to buffer requests before
                processing. Trades off latency for batch efficiency. Setting this too high
                increases latency; too low reduces batch efficiency.
                Default: 1.0

            max_batch_size (int): Maximum number of queries to process in a single batch.
                Larger batches improve GPU utilization but increase per-request latency.
                Should be tuned based on GPU memory and target latency.
                Default: 1024

            server_mode (Literal["singleton", "batch"]): Server operation mode.

                - "singleton": Each query contains a single completion to score.
                - "batch": Each query can contain multiple completions to score.
                Default: "singleton"


        Note:
            This class is thread-safe for asyncio but not for multi-threaded access.
            All methods must be called from the same asyncio event loop.
        """
        self.app = app
        self.buffer_time = buffer_time
        self.server_mode = server_mode
        self.max_batch_size = max_batch_size
        self.queue: deque[Tuple[MolecularVerifierServerQuery, asyncio.Future]] = deque()
        self.lock = asyncio.Lock()
        self.processing_task = asyncio.create_task(self._batch_loop())

    async def add_query(
        self, query: MolecularVerifierServerQuery
    ) -> MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse:
        """Add a query to the buffer and wait for its result asynchronously.

        This method is non-blocking: it immediately queues the request and returns
        an awaitable future. The actual computation happens asynchronously in the
        background batch processing task. Multiple callers can call this method
        concurrently without blocking each other.

        The method uses an asyncio.Lock to ensure thread-safe queue access. If an
        error occurs during queueing, the exception is logged and re-raised to the
        caller.

        Args:
            query (MolecularVerifierServerQuery): A molecular verifier query containing:

                - metadata: List of metadata dictionaries (one per completion)
                - query: List of completion strings to score
                - prompts: Optional list of original prompts for tracking

        Returns:
            MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse:
                Response type depends on server_mode:

                - In "singleton" mode: MolecularVerifierServerResponse containing:
                    - reward: Single reward score
                    - meta: Single metadata object with detailed scoring information
                    - error: Error message if scoring failed
                    - next_turn_feedback: Optional feedback for multi-turn conversations

                - In "batch" mode: BatchMolecularVerifierServerResponse containing:
                    - rewards: List of individual reward scores (one per completion)
                    - metas: List of metadata objects (one per completion)
                    - error: Error message if scoring failed
                    - next_turn_feedback: Optional feedback for multi-turn conversations

        Raises:
            Exception: Any exception raised during queueing or during result
                computation in the background task. The exception is logged
                before being raised to the caller.
        """
        try:
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            async with self.lock:
                self.queue.append((query, future))
            return await future  # type:ignore
        except Exception as e:
            logger.error(f"Error in add_query: {e}")
            raise e

    async def _batch_loop(self) -> None:
        """Background task that continuously processes buffered queries in batches.

        This coroutine runs indefinitely in the background (started in __init__),
        periodically waking up to process accumulated requests. It implements a
        simple time-based flushing strategy: every buffer_time seconds, any pending
        queries are processed as a batch.

        The loop continues even if individual batch processing raises exceptions
        (which are caught in _process_pending_queries). This ensures the buffer
        remains responsive and recovers from transient errors.

        Processing frequency:

        - Minimum: Every buffer_time seconds
        - Maximum: Immediate if batch reaches max_batch_size (handled in
          _process_pending_queries)

        Note:
            This task should only be created once during __init__ and continues
            until the application shuts down or the event loop exits.
        """
        while True:
            await asyncio.sleep(self.buffer_time)
            await self._process_pending_queries()

    async def _process_pending_queries(self) -> None:
        """Process all pending queries in the buffer as one or more batches.

        This method is called periodically by _batch_loop. It safely removes queries
        from the queue (up to max_batch_size at a time) and calls _process_batch
        to compute rewards in parallel. Results are then delivered to each query's
        future, unblocking the corresponding add_query caller.

        If the queue is empty, this method returns immediately without doing work.
        If the queue has items, they are popped in chunks of max_batch_size and
        processed separately.

        Error handling:

        - Exceptions during _process_batch are caught and set on all futures in
          that batch, preventing one failure from affecting other batches
        - The loop continues processing other batches even if one fails
        - All errors are logged for debugging

        Concurrency:

        - Uses asyncio.Lock to safely access the queue without race conditions
        - Futures are unblocked sequentially after batch completion

        Returns:
            None. Results are delivered via asyncio.Future.set_result() and
            asyncio.Future.set_exception().

        Raises:
            Exceptions are caught and logged but not re-raised. They are instead
            passed to futures via set_exception().
        """
        try:
            async with self.lock:
                if not self.queue:
                    return

                batch = [
                    self.queue.popleft()
                    for _ in range(min(len(self.queue), self.max_batch_size))
                ]

            queries, futures = zip(*batch)
            logger.info(f"Processing batch of size {len(queries)}")
            try:
                responses = await self._process_batch(list(queries))
                for fut, res in zip(futures, responses):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
        except Exception as e:
            logger.error(f"Error in _process_pending_queries: {e}")
            raise e

    async def _process_batch(
        self, queries: List[MolecularVerifierServerQuery]
    ) -> List[MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse]:
        """Process a batch of queries by merging, scoring, and grouping results.

        This is the core batch processing pipeline. It combines multiple queries
        into a single large batch, sends it to the Ray actor for parallel scoring,
        and then groups results back to individual queries.

        Pipeline steps:

        1. **Merge inputs**: Concatenate all completions and metadata from all
           queries. Track which query each item came from using query_indices.

        2. **Handle empty batch**: Return error responses if no valid completions.

        3. **Compute scores**: Send merged batch to Ray actor via remote call.
           The actor returns BatchMolecularVerifierOutputModel with rewards and
           metadata for each completion.

        4. **Group by query**: Redistribute results back to original queries using
           query_indices. Each query gets back only its own results.

        5. **Aggregate metrics**: For each query, compute aggregate reward as the
           average of its individual rewards. Pack metadata into structured objects.

        Args:
            queries (List[MolecularVerifierServerQuery]): A list of queries to
                process together. Each query can contain multiple completions.
                Typically 1-16 queries per batch.

        Returns:
            List[MolecularVerifierServerResponse]: One response per input query,

                in the same order. Each response contains:
                - reward: Average reward across all completions in that query
                - reward_list: Individual rewards for each completion
                - meta: Detailed metadata from the verifier for each completion
                - error: Error message if batch processing failed

        Raises:
            Any exception raised by the Ray actor's get_score method is
            propagated to the caller and will be set on all futures in
            _process_pending_queries.
        """
        app = self.app

        # --- Step 1. Merge all batched inputs ---
        all_completions: List[str] = []
        all_metadata: List[dict[str, Any]] = []
        query_indices: List[int] = []

        for i, q in enumerate(queries):
            all_completions.extend(q.query)
            all_metadata.extend(q.metadata)
            query_indices.extend([i] * len(q.query))

        if len(all_completions) == 0:
            # All failed early
            return [
                q
                if isinstance(q, MolecularVerifierServerResponse)
                else MolecularVerifierServerResponse(
                    reward=0.0, reward_list=[], error="Empty batch"
                )
                for q in queries
            ]

        # --- Step 2. Compute batched reward ---
        reward_actor = app.state.reward_model

        # Run in parallel
        rewards_job = reward_actor.get_score.remote(
            completions=all_completions, metadata=all_metadata
        )

        out = ray.get(rewards_job)
        rewards = out.rewards
        parsed_answers = out.parsed_answers
        metadatas = out.verifier_metadatas
        # --- Step 3. Group results by original query ---
        grouped_results: List[List[float]] = [[] for _ in range(len(queries))]
        grouped_meta: List[List[MolecularVerifierOutputMetadataModel]] = [
            [] for _ in range(len(queries))
        ]
        grouped_parsed_answers: List[List[str]] = [[] for _ in range(len(queries))]

        for r, m, p, idx in zip(rewards, metadatas, parsed_answers, query_indices):
            grouped_results[idx].append(r)
            grouped_meta[idx].append(m)
            grouped_parsed_answers[idx].append(p)

        # --- Step 4. Compute per-query metrics ---
        responses: List[
            MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse
        ] = []
        for i, q in enumerate(queries):
            if isinstance(q, MolecularVerifierServerResponse):
                # prefilled error
                responses.append(q)
                continue
            rewards_i = grouped_results[i]
            metadata_i = grouped_meta[i]
            parsed_answers_i = grouped_parsed_answers[i]
            # Transform metadata to pydantic models
            server_metadata_i: List[MolecularVerifierServerMetadata] = [
                MolecularVerifierServerMetadata.model_validate(m.model_dump())
                for m in metadata_i
            ]
            for serv_m, p_answer in zip(server_metadata_i, parsed_answers_i):
                serv_m.parsed_answer = p_answer

            response: (
                MolecularVerifierServerResponse | BatchMolecularVerifierServerResponse
            )
            if self.server_mode == "singleton":
                assert len(rewards_i) == 1, (
                    "Expected singleton mode to have one reward per query"
                )
                response = MolecularVerifierServerResponse(
                    reward=rewards_i[0],
                    meta=server_metadata_i[0],
                    error=None,
                )
            elif self.server_mode == "batch":
                response = BatchMolecularVerifierServerResponse(
                    rewards=rewards_i,
                    metas=server_metadata_i,
                    error=None,
                )
            responses.append(response)

        return responses
