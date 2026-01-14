import logging
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from orchestration.orchestrator import Orchestrator
from connectors.appconfig import AppConfigClient
from connectors.cosmosdb import (
    query_user_conversations,
    read_user_conversation,
    update_conversation_name,
    soft_delete_conversation,
)
from dependencies import get_config, validate_auth, validate_access_token, get_user_groups_from_graph
from telemetry import Telemetry
from schemas import OrchestratorRequest, ConversationListResponse, ConversationMetadata, ConversationDetail, ORCHESTRATOR_RESPONSES
from constants import APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME
from util.tools import is_azure_environment

# ----------------------------------------
# Initialization and logging
# - Minimal early logging so config/auth warnings are visible during startup
# - Azure SDK and HTTP pipeline logs are verbose only when LOG_LEVEL=DEBUG;
#   otherwise they’re kept at WARNING to reduce noise
# ----------------------------------------

## Early minimal logging (INFO) until config is loaded; refined by Telemetry.configure_basic
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Suppress Azure SDK HTTP logging immediately (before any Azure imports)
for _azure_logger in [
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.identity",
    "azure.core",
    "azure"
]:
    logger = logging.getLogger(_azure_logger)
    logger.setLevel(logging.CRITICAL)
    logger.propagate = False
    logger.disabled = True
    logger.handlers.clear()
    
# Load version from VERSION file 
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
try:
    APP_VERSION = VERSION_FILE.read_text().strip()
except FileNotFoundError:
    APP_VERSION = "0.0.0"

# 2) Create configuration client (sets cfg.auth_failed=True if auth is unavailable)
cfg: AppConfigClient = get_config()

# 3) Configure logging level/format from LOG_LEVEL
Telemetry.configure_basic(cfg)
Telemetry.log_log_level_diagnostics(cfg)

# 4) If authentication failed, exit immediately
if getattr(cfg, "auth_failed", False):
    logging.warning("The orchestrator is not authenticated (run 'az login' or configure Managed Identity). Exiting...")
    logging.shutdown()
    os._exit(1)

# ----------------------------------------
# Create FastAPI app with lifespan
# ----------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    Telemetry.configure_monitoring(cfg, APPLICATION_INSIGHTS_CONNECTION_STRING, APP_NAME)
    yield  # <-- application runs here
    # cleanup logic after shutdown

app = FastAPI(
    title="GPT-RAG Orchestrator",
    description="GPT-RAG Orchestrator FastAPI",
    version=APP_VERSION,
    lifespan=lifespan
)


# ----------------------------------------
# Helper function for auth validation
# ----------------------------------------
async def validate_user_access(authorization: Optional[str], endpoint_name: str, require_auth: bool = False) -> str:
    """
    Validates user authorization and returns principal_id.
    
    Args:
        authorization: Authorization header value (e.g., "Bearer <token>")
        endpoint_name: Name of endpoint for logging (e.g., "[Orchestrator]")
        require_auth: If True, endpoint requires authentication to be enabled and valid token
    
    Returns:
        principal_id (OID) of the authenticated user, or "anonymous" if auth disabled and not required
    
    Raises:
        HTTPException: If authorization fails, is invalid, or auth is required but disabled
    """
    enable_authentication = cfg.get("ENABLE_AUTHENTICATION", default=False, type=bool)
    
    if not enable_authentication:
        if require_auth:
            logging.warning(f"{endpoint_name} Authentication required but ENABLE_AUTHENTICATION is False")
            raise HTTPException(status_code=401, detail="Authentication is required for this endpoint")
        logging.debug(f"{endpoint_name} Authorization disabled, treating as anonymous")
        return "anonymous"
    
    # Validate Authorization header
    if not authorization or not authorization.startswith("Bearer "):
        logging.warning(f"{endpoint_name} Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    
    access_token = authorization[7:]  # Remove "Bearer " prefix
    logging.debug(f"{endpoint_name} Access token received, length: %d chars", len(access_token))
    
    try:
        # Validate token and extract user info
        user_info = await validate_access_token(access_token)
        principal_id = user_info.get("oid")
        
        logging.debug(f"{endpoint_name} User info extracted: OID=%s, Username=%s, Name=%s", 
                     principal_id, user_info.get("preferred_username"), user_info.get("name"))
        
        # Fetch user groups from Graph API
        logging.debug(f"{endpoint_name} Fetching user groups from Graph API...")
        groups = await get_user_groups_from_graph(principal_id)
        
        logging.debug(f"{endpoint_name} User groups: %s", groups)
        
        # Check authorization based on groups/principals
        allowed_names = [n.strip() for n in cfg.get("ALLOWED_USER_NAMES", default="").split(",") if n.strip()]
        allowed_ids = [id.strip() for id in cfg.get("ALLOWED_USER_PRINCIPALS", default="").split(",") if id.strip()]
        allowed_groups = [g.strip() for g in cfg.get("ALLOWED_GROUP_NAMES", default="").split(",") if g.strip()]
        
        logging.debug(f"{endpoint_name} Authorization checks - Allowed names: %s, IDs: %s, Groups: %s", 
                     allowed_names, allowed_ids, allowed_groups)
        
        is_authorized = (
            not (allowed_names or allowed_ids or allowed_groups) or
            user_info.get("preferred_username") in allowed_names or
            principal_id in allowed_ids or
            any(g in allowed_groups for g in groups)
        )
        
        if not is_authorized:
            logging.warning(f"{endpoint_name} ❌ Access denied for user %s (%s)", 
                           principal_id, user_info.get("preferred_username"))
            raise HTTPException(status_code=403, detail="You are not authorized to perform this action")
        
        logging.info(f"{endpoint_name} ✅ Authorization successful for user %s", user_info.get("preferred_username"))
        return principal_id
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"{endpoint_name} Error validating user token: %s", e)
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.post(
    "/orchestrator",
    dependencies=[Depends(validate_auth)], 
    summary="Ask orchestrator a question",
    response_description="Returns the orchestrator’s response in real time, streamed via SSE.",
    responses=ORCHESTRATOR_RESPONSES
)
async def orchestrator_endpoint(
    body: OrchestratorRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    dapr_api_token: Optional[str] = Header(None, alias="dapr-api-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Accepts JSON payload with ask/question, optional conversation_id and context,
    then streams back an answer via SSE.
    """

    # Determine operation type first (defensive: body may not include type)
    op_type = getattr(body, "type", None)

    # Extract and validate user info from access token if authentication is enabled
    user_context = body.user_context or {}
    
    principal_id = await validate_user_access(authorization, "[Orchestrator]")
    user_context["principal_id"] = principal_id
    
    if principal_id != "anonymous":
        # Extract additional user info for non-anonymous users
        access_token = authorization[7:]
        user_info = await validate_access_token(access_token)
        user_context["principal_name"] = user_info.get("preferred_username")
        user_context["user_name"] = user_info.get("name")
        user_context["groups"] = await get_user_groups_from_graph(principal_id)

    # Feedback submissions: allow missing ask/question; validate only what's required
    if op_type == "feedback":
        # Handle feedback submission
        conversation_id = body.conversation_id
        if not conversation_id:
            logging.error(f"No 'conversation_id' provided in feedback body, and payload is {body}")
            raise HTTPException(status_code=400, detail="No 'conversation_id' field in request body")

        # Create orchestrator instance and save feedback
        principal_id = user_context.get("principal_id", "anonymous")
        orchestrator = await Orchestrator.create(
            conversation_id=conversation_id, 
            principal_id=principal_id,
            user_context=user_context
        )
        # Build feedback dict defensively; optional fields may be absent
        _qid = getattr(body, "question_id", None)
        feedback = {
            "conversation_id": conversation_id,
            "question_id": _qid,
            "is_positive": getattr(body, "is_positive", None),
            "stars_rating": getattr(body, "stars_rating", None),
            # Normalize empty strings to None
            "feedback_text": (getattr(body, "feedback_text", None) or "").strip() or None,
        }
        await orchestrator.save_feedback(feedback)
        return {"status": "success", "message": "Feedback saved successfully"}    

    # For non-feedback operations, require an ask/question
    ask = (getattr(body, "ask", None) or getattr(body, "question", None))
    if not ask:
        raise HTTPException(status_code=400, detail="No 'ask' or 'question' field in request body")

    principal_id = user_context.get("principal_id", "anonymous")
    orchestrator = await Orchestrator.create(
        conversation_id=body.conversation_id,
        principal_id=principal_id,
        user_context=user_context
    )

    async def sse_event_generator():
        try:
            _qid = getattr(body, "question_id", None) 
            async for chunk in orchestrator.stream_response(ask, _qid):
                yield f"{chunk}"
        except Exception as e:
            logging.exception("Error in SSE generator")
            yield "event: error\ndata: An internal server error occurred.\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream"
    )


@app.get(
    "/conversations",
    dependencies=[Depends(validate_auth)],
    summary="List user conversations",
    response_model=ConversationListResponse,
    responses={
        200: {"description": "OK — list of conversations"},
        401: {"description": "Unauthorized — missing or invalid credentials"},
        500: {"description": "Internal Server Error"}
    }
)
async def list_conversations(
    skip: int = 0,
    limit: int = 10,
    name: Optional[str] = None,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    dapr_api_token: Optional[str] = Header(None, alias="dapr-api-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Retrieve paginated list of conversations for the authenticated user.
    
    Query parameters:
    - skip: Number of conversations to skip (default: 0)
    - limit: Maximum number of conversations to return (default: 10)
    - name: Optional filter by conversation name (exact match)
    
    Returns metadata only (id, name, created_at, lastUpdated) without message content.
    """
    
    principal_id = await validate_user_access(authorization, "[ListConversations]", require_auth=True)
    
    try:
        conversation_docs = await query_user_conversations(
            principal_id=principal_id,
            skip=skip,
            limit=limit,
            name=name,
        )
        conversations = [
            ConversationMetadata(
                id=doc.get("id"),
                name=doc.get("name"),
                created_at=doc.get("_ts"),
                last_updated=doc.get("lastUpdated"),
            )
            for doc in conversation_docs
        ]

        logging.debug(
            "[ListConversations] User %s: retrieved %d conversations",
            principal_id,
            len(conversations),
        )

        has_more = len(conversations) == limit
        return ConversationListResponse(
            conversations=conversations,
            has_more=has_more,
            skip=skip,
            limit=limit,
        )

    except Exception as e:
        logging.error("[ListConversations] Error retrieving conversations: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving conversations")


@app.get(
    "/conversations/{conversation_id}",
    dependencies=[Depends(validate_auth)],
    summary="Get specific conversation",
    response_model=ConversationDetail,
    responses={
        200: {"description": "OK — full conversation detail"},
        401: {"description": "Unauthorized — missing or invalid credentials"},
        403: {"description": "Forbidden — not the owner of this conversation"},
        404: {"description": "Not Found — conversation does not exist"},
        500: {"description": "Internal Server Error"}
    }
)
async def get_conversation(
    conversation_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    dapr_api_token: Optional[str] = Header(None, alias="dapr-api-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Retrieve full details of a specific conversation including all messages.
    
    Path parameters:
    - conversation_id: The ID of the conversation to retrieve
    
    Returns the complete conversation document with all messages.
    Verifies that the requester is the owner (principal_id matches).
    """
    
    principal_id = await validate_user_access(authorization, "[GetConversation]", require_auth=True)
    
    try:
        conversation_doc = await read_user_conversation(conversation_id, principal_id)
        if conversation_doc is None:
            logging.debug(
                "[GetConversation] Conversation %s not found for user %s",
                conversation_id,
                principal_id,
            )
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation_doc.get("principal_id") != principal_id:
            logging.warning(
                "[GetConversation] Access denied: user %s tried to access conversation %s",
                principal_id,
                conversation_id,
            )
            raise HTTPException(status_code=403, detail="You do not have permission to access this conversation")

        logging.debug(
            "[GetConversation] User %s retrieved conversation %s", principal_id, conversation_id
        )

        return ConversationDetail(
            id=conversation_doc.get("id"),
            name=conversation_doc.get("name"),
            principal_id=conversation_doc.get("principal_id"),
            created_at=conversation_doc.get("_ts"),
            last_updated=conversation_doc.get("lastUpdated"),
            messages=conversation_doc.get("messages", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error("[GetConversation] Error retrieving conversation: %s", e)
        raise HTTPException(status_code=500, detail="Error retrieving conversation")


@app.patch(
    "/conversations/{conversation_id}",
    dependencies=[Depends(validate_auth)],
    summary="Update conversation name",
    responses={
        200: {"description": "OK — conversation updated"},
        401: {"description": "Unauthorized — missing or invalid credentials"},
        403: {"description": "Forbidden — not the owner of this conversation"},
        404: {"description": "Not Found — conversation does not exist or is deleted"},
        500: {"description": "Internal Server Error"}
    }
)
async def update_conversation(
    conversation_id: str,
    body: dict,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    dapr_api_token: Optional[str] = Header(None, alias="dapr-api-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Update a conversation's name.
    
    Request body should contain:
    - name: New conversation name
    
    Returns the updated conversation metadata.
    Verifies that the requester is the owner (principal_id matches).
    """
    
    principal_id = await validate_user_access(authorization, "[UpdateConversation]", require_auth=True)
    
    try:
        # Validate that conversation exists and user owns it
        conversation_doc = await read_user_conversation(conversation_id, principal_id)
        if conversation_doc is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation_doc.get("principal_id") != principal_id:
            logging.warning(
                "[UpdateConversation] Access denied: user %s tried to update conversation %s",
                principal_id,
                conversation_id,
            )
            raise HTTPException(status_code=403, detail="You do not have permission to access this conversation")

        # Extract new name from body
        new_name = body.get("name", "").strip()
        if not new_name:
            raise HTTPException(status_code=400, detail="Conversation name cannot be empty")

        # Update the conversation
        updated_doc = await update_conversation_name(conversation_id, principal_id, new_name)
        if updated_doc is None:
            raise HTTPException(status_code=500, detail="Failed to update conversation")

        logging.debug(
            "[UpdateConversation] User %s updated conversation %s name", principal_id, conversation_id
        )

        return {
            "id": updated_doc.get("id"),
            "name": updated_doc.get("name"),
            "lastUpdated": updated_doc.get("lastUpdated"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error("[UpdateConversation] Error updating conversation: %s", e)
        raise HTTPException(status_code=500, detail="Error updating conversation")


@app.delete(
    "/conversations/{conversation_id}",
    dependencies=[Depends(validate_auth)],
    summary="Delete conversation",
    responses={
        200: {"description": "OK — conversation deleted"},
        401: {"description": "Unauthorized — missing or invalid credentials"},
        403: {"description": "Forbidden — not the owner of this conversation"},
        404: {"description": "Not Found — conversation does not exist"},
        500: {"description": "Internal Server Error"}
    }
)
async def delete_conversation(
    conversation_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    dapr_api_token: Optional[str] = Header(None, alias="dapr-api-token"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
):
    """
    Soft delete a conversation (marks as deleted without removing data).
    
    The conversation will no longer appear in list endpoints and cannot be accessed or modified.
    The data is retained for audit purposes.
    
    Verifies that the requester is the owner (principal_id matches).
    """
    
    principal_id = await validate_user_access(authorization, "[DeleteConversation]", require_auth=True)
    
    try:
        # Validate that conversation exists and user owns it
        conversation_doc = await read_user_conversation(conversation_id, principal_id)
        if conversation_doc is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if conversation_doc.get("principal_id") != principal_id:
            logging.warning(
                "[DeleteConversation] Access denied: user %s tried to delete conversation %s",
                principal_id,
                conversation_id,
            )
            raise HTTPException(status_code=403, detail="You do not have permission to access this conversation")

        # Soft delete the conversation
        deleted_doc = await soft_delete_conversation(conversation_id, principal_id)
        if deleted_doc is None:
            raise HTTPException(status_code=500, detail="Failed to delete conversation")

        logging.info(
            "[DeleteConversation] User %s deleted conversation %s", principal_id, conversation_id
        )

        return {"status": "success", "message": "Conversation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logging.error("[DeleteConversation] Error deleting conversation: %s", e)
        raise HTTPException(status_code=500, detail="Error deleting conversation")

# Instrumentation
HTTPXClientInstrumentor().instrument()
FastAPIInstrumentor.instrument_app(app)

# Run the app locally (avoid nested event loop when started by uvicorn CLI)
if __name__ == "__main__" and not is_azure_environment():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info", timeout_keep_alive=60)

