import uuid
import logging
from datetime import datetime, timezone

from typing import Dict, Optional
from connectors.cosmosdb import CosmosDBClient
from strategies.agent_strategy_factory import AgentStrategyFactory
from strategies.base_agent_strategy import BaseAgentStrategy
from dependencies import get_config
from opentelemetry.trace import SpanKind
from telemetry import Telemetry

tracer = Telemetry.get_tracer(__name__)

class Orchestrator:
    agentic_strategy = BaseAgentStrategy

    def __init__(self, conversation_id: str, principal_id: str = None):
        self.conversation_id = conversation_id
        self.principal_id = principal_id
        cfg = get_config()
        self.database_client = CosmosDBClient()
        self.database_container = cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
        
    @classmethod
    async def create(cls, conversation_id: str = None, principal_id: str = None, user_context: Dict = {}):
        instance = cls(conversation_id=conversation_id, principal_id=principal_id)
        cfg = get_config()

        agentic_strategy_name = cfg.get("AGENT_STRATEGY", "single_agent_rag")
        instance.agentic_strategy = await AgentStrategyFactory.get_strategy(agentic_strategy_name)
        if not instance.agentic_strategy:
            raise EnvironmentError("AGENT_STRATEGY must be set")

        instance.agentic_strategy.user_context = user_context

        return instance

    async def stream_response(self, ask: str, question_id: Optional[str] = None):
        logging.debug(f"Starting conversation {self.conversation_id}")

        with tracer.start_as_current_span('stream_response', kind=SpanKind.SERVER) as span:

            span.set_attribute('conversation_id', self.conversation_id)

            # 1) Load or create our conversation document in Cosmos
            if not self.conversation_id:
                self.conversation_id = str(uuid.uuid4())
                
                # For anonymous users, use anonymous-{conversation_id} as partition key to avoid hot partitions
                # For authenticated users, use their principal_id
                partition_key = f"anonymous-{self.conversation_id}" if self.principal_id == "anonymous" else self.principal_id
                
                # Auto-generate conversation name from first 50 characters of the ask
                default_name = ask[:50] if ask else "Untitled Conversation"
                conversation = {
                    "id": self.conversation_id,
                    "name": default_name,
                    "principal_id": partition_key,  # Store the partition key in the document
                    "lastUpdated": datetime.now(timezone.utc).isoformat()
                }
                await self.database_client.create_document(
                    self.database_container, 
                    self.conversation_id, 
                    conversation,
                    partition_key=partition_key
                )
            else:
                # For existing conversations, determine the partition key
                partition_key = f"anonymous-{self.conversation_id}" if self.principal_id == "anonymous" else self.principal_id
                conversation = await self.database_client.get_document(
                    self.database_container, 
                    self.conversation_id,
                    partition_key=partition_key
                )
                if conversation is None:
                    raise ValueError(f"Conversation {self.conversation_id} not found")

            # Optionally record the incoming question (id + text) for traceability
            if question_id:
                questions = conversation.get("questions") or []
                questions.append({
                    "question_id": question_id,
                    "text": ask
                })
                conversation["questions"] = questions

            # 2) Hand off the conversation dict to the strategy
            self.agentic_strategy.conversation = conversation

            # 3) Stream all chunks from the strategy
            try:
                yield f"{self.conversation_id} "
                async for chunk in self.agentic_strategy.initiate_agent_flow(ask):
                    yield chunk
            finally:
                # 4) Persist whatever the strategy has updated (e.g. thread_id)
                await self.database_client.update_document(
                    self.database_container, 
                    self.agentic_strategy.conversation
                )

            logging.debug(f"Finished conversation {self.conversation_id}")

    async def save_feedback(self, feedback: Dict):
        """
        Save user feedback into the same Cosmos DB container as the conversation.
        """
        if not self.conversation_id:
            raise ValueError("Conversation ID is required to save feedback")

        # For anonymous users, use anonymous-{conversation_id} as partition key; for authenticated, use principal_id
        partition_key = f"anonymous-{self.conversation_id}" if self.principal_id == "anonymous" else self.principal_id

        # Retrieve existing conversation document
        conversation = await self.database_client.get_document(
            self.database_container,
            self.conversation_id,
            partition_key=partition_key
        )
        if conversation is None:
            raise ValueError(f"Conversation {self.conversation_id} not found in database")

        # Try to resolve question_id if the client didn't provide it
        try:
            provided_question_id = feedback.get("question_id")
            if not provided_question_id:
                questions = conversation.get("questions") or []
                resolved_question_id = None

                # 1) Prefer matching by question text (most recent first)
                fb_text = (feedback.get("text") or "").strip()
                if fb_text:
                    for q in reversed(questions):
                        if (q.get("text") or "").strip() == fb_text:
                            resolved_question_id = q.get("question_id")
                            break

                # 2) Fallback to the last question's question_id
                if not resolved_question_id and questions:
                    resolved_question_id = questions[-1].get("question_id")

                if resolved_question_id:
                    feedback["question_id"] = resolved_question_id
                else:
                    logging.warning(
                        f"Could not resolve question_id for feedback in conversation {self.conversation_id}; saving with question_id=null"
                    )
        except Exception as e:
            # Do not fail feedback saving if resolution logic errors; just log
            logging.exception("Error attempting to resolve question_id from conversation questions: %s", e)

        if "feedback" not in conversation:
            conversation["feedback"] = []
        conversation["feedback"].append(feedback)

        await self.database_client.update_document(
            self.database_container, 
            conversation
        )
        logging.info(f"Feedback saved for conversation {self.conversation_id}")
