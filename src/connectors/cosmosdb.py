import logging
from datetime import datetime, timezone
from typing import List, Optional

from azure.cosmos.aio import CosmosClient
from dependencies import get_config


class CosmosDBClient:
    """
    CosmosDBClient uses the Cosmos SDK's.
    """

    def __init__(self):
        """
        Initializes the Cosmos DB client with credentials and endpoint.
        """
        # ==== Load all config parameters in one place ====
        self.cfg = get_config()
        self.database_account_name = self.cfg.get("DATABASE_ACCOUNT_NAME")
        self.database_name = self.cfg.get("DATABASE_NAME")
        self.db_uri = f"https://{self.database_account_name}.documents.azure.com:443/"
        # ==== End config block ====

    async def list_documents(self, container_name) -> list:
        """
        Lists all documents from the given container.
        """
        async with CosmosClient(self.db_uri, credential=self.cfg.aiocredential) as db_client:
            db = db_client.get_database_client(database=self.database_name)
            container = db.get_container_client(container_name)

            # Correct usage without the outdated argument
            query = "SELECT * FROM c"
            items_iterable = container.query_items(query=query, partition_key=None)

            documents = []
            async for item in items_iterable:
                documents.append(item)

            return documents

    async def get_document(self, container, key, partition_key=None) -> dict: 
        """Retrieve a document by key with optional partition key.
        
        Args:
            container: Container name
            key: Document ID
            partition_key: Partition key value (required for containers with partition key)
                          If None, uses key as partition key for backward compatibility
        """
        # Use provided partition_key or fall back to key for backward compatibility
        pk_value = partition_key if partition_key is not None else key
        
        async with CosmosClient(self.db_uri, credential=self.cfg.aiocredential) as db_client:
            db = db_client.get_database_client(database=self.database_name)
            container = db.get_container_client(container)
            try:
                document = await container.read_item(item=key, partition_key=pk_value)
                logging.info(f"[cosmosdb] document {key} retrieved from partition {pk_value}.")
            except Exception as e:
                document = None
                logging.debug(f"[cosmosdb] document {key} does not exist: {e}")
            return document

    async def create_document(self, container, key, body=None, partition_key=None) -> dict: 
        """Create a new document with optional partition key.
        
        Args:
            container: Container name
            key: Document ID
            body: Document body (dict). If not provided, creates minimal document with id only.
                  If partition_key is provided, it's added to body as principal_id.
            partition_key: Partition key value. If provided, ensures it's in the document body.
        """
        async with CosmosClient(self.db_uri, credential=self.cfg.aiocredential) as db_client:
            db = db_client.get_database_client(database=self.database_name)
            container = db.get_container_client(container)
            try:
                if body is None:
                    body = {"id": key}
                else:
                    body["id"] = key  # ensure the document id is set
                body["lastUpdated"] = datetime.now(timezone.utc).isoformat()
                
                # If partition_key provided, ensure it's in the body (for principal_id containers)
                if partition_key is not None:
                    body["principal_id"] = partition_key
                
                document = await container.create_item(body=body)                    
                logging.info(f"[cosmosdb] document {key} created in partition {partition_key or 'default'}.")
            except Exception as e:
                document = None
                logging.error(f"[cosmosdb] error creating document {key}. Error: {e}", exc_info=True)
            return document
        
    async def update_document(self, container, document) -> dict: 
        """Update an existing document.
        
        The partition key (principal_id) is inferred from the document body.
        """
        async with CosmosClient(self.db_uri, credential=self.cfg.aiocredential) as db_client:
            db = db_client.get_database_client(database=self.database_name)
            container = db.get_container_client(container)
            try:
                document["lastUpdated"] = datetime.now(timezone.utc).isoformat()
                document = await container.replace_item(item=document["id"], body=document)
                logging.info(f"[cosmosdb] document {document['id']} updated.")
            except Exception as e:
                document = None
                logging.error(f"[cosmosdb] could not update document: {e}", exc_info=True)
            return document


async def query_user_conversations(
    principal_id: str,
    skip: int,
    limit: int,
    name: Optional[str] = None,
) -> List[dict]:
    """Return the requested span of a user's conversations, excluding soft-deleted ones."""
    cosmos = CosmosDBClient()
    async with CosmosClient(cosmos.db_uri, credential=cosmos.cfg.aiocredential) as db_client:
        db = db_client.get_database_client(database=cosmos.database_name)
        container_name = cosmos.cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
        container = db.get_container_client(container_name)

        if name:
            query = """
                SELECT c.id, c.name, c._ts, c.lastUpdated
                FROM c
                WHERE c.principal_id = @principal_id 
                  AND CONTAINS(c.name, @name) 
                  AND (NOT IS_DEFINED(c.isDeleted) OR c.isDeleted = false)
                ORDER BY c._ts DESC
                OFFSET @skip LIMIT @limit
            """
            parameters = [
                {"name": "@principal_id", "value": principal_id},
                {"name": "@name", "value": name},
                {"name": "@skip", "value": skip},
                {"name": "@limit", "value": limit},
            ]
        else:
            query = """
                SELECT c.id, c.name, c._ts, c.lastUpdated
                FROM c
                WHERE c.principal_id = @principal_id 
                  AND (NOT IS_DEFINED(c.isDeleted) OR c.isDeleted = false)
                ORDER BY c._ts DESC
                OFFSET @skip LIMIT @limit
            """
            parameters = [
                {"name": "@principal_id", "value": principal_id},
                {"name": "@skip", "value": skip},
                {"name": "@limit", "value": limit},
            ]

        items_iterable = container.query_items(
            query=query,
            parameters=parameters,
            partition_key=principal_id,
        )

        conversations = []
        async for document in items_iterable:
            conversations.append(document)

        logging.debug(
            "[CosmosDB] User %s retrieved %d conversations", principal_id, len(conversations)
        )
        return conversations


async def read_user_conversation(conversation_id: str, principal_id: str) -> Optional[dict]:
    """Return the conversation document if the partition matches and not soft-deleted."""
    cosmos = CosmosDBClient()
    async with CosmosClient(cosmos.db_uri, credential=cosmos.cfg.aiocredential) as db_client:
        db = db_client.get_database_client(database=cosmos.database_name)
        container_name = cosmos.cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
        container = db.get_container_client(container_name)

        try:
            doc = await container.read_item(item=conversation_id, partition_key=principal_id)
            # Return None if the document is soft-deleted
            if doc.get("isDeleted") == True:
                logging.debug(
                    "[CosmosDB] Conversation %s is marked as deleted",
                    conversation_id,
                )
                return None
            return doc
        except Exception as exc:
            logging.debug(
                "[CosmosDB] Conversation %s not found or invalid partition for %s: %s",
                conversation_id,
                principal_id,
                exc,
            )
            return None


async def update_conversation_name(conversation_id: str, principal_id: str, new_name: str) -> Optional[dict]:
    """Update the name of a conversation (soft-deleted conversations cannot be updated)."""
    cosmos = CosmosDBClient()
    async with CosmosClient(cosmos.db_uri, credential=cosmos.cfg.aiocredential) as db_client:
        db = db_client.get_database_client(database=cosmos.database_name)
        container_name = cosmos.cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
        container = db.get_container_client(container_name)

        try:
            doc = await container.read_item(item=conversation_id, partition_key=principal_id)
            
            # Check if soft-deleted
            if doc.get("isDeleted") == True:
                logging.warning(
                    "[CosmosDB] Cannot update soft-deleted conversation %s", conversation_id
                )
                return None
            
            # Update the name and lastUpdated
            doc["name"] = new_name
            doc["lastUpdated"] = datetime.now(timezone.utc).isoformat()
            updated_doc = await container.replace_item(item=conversation_id, body=doc)
            
            logging.info(
                "[CosmosDB] Conversation %s name updated to '%s'", conversation_id, new_name
            )
            return updated_doc
            
        except Exception as exc:
            logging.error(
                "[CosmosDB] Error updating conversation %s: %s",
                conversation_id,
                exc,
            )
            return None


async def soft_delete_conversation(conversation_id: str, principal_id: str) -> Optional[dict]:
    """Soft delete a conversation by setting isDeleted=True."""
    cosmos = CosmosDBClient()
    async with CosmosClient(cosmos.db_uri, credential=cosmos.cfg.aiocredential) as db_client:
        db = db_client.get_database_client(database=cosmos.database_name)
        container_name = cosmos.cfg.get("CONVERSATIONS_DATABASE_CONTAINER", "conversations")
        container = db.get_container_client(container_name)

        try:
            doc = await container.read_item(item=conversation_id, partition_key=principal_id)
            
            # Set soft delete flag and timestamp
            doc["isDeleted"] = True
            doc["deletedAt"] = datetime.now(timezone.utc).isoformat()
            doc["lastUpdated"] = doc["deletedAt"]
            updated_doc = await container.replace_item(item=conversation_id, body=doc)
            
            logging.info(
                "[CosmosDB] Conversation %s soft deleted", conversation_id
            )
            return updated_doc
            
        except Exception as exc:
            logging.error(
                "[CosmosDB] Error soft deleting conversation %s: %s",
                conversation_id,
                exc,
            )
            return None