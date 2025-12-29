import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import setup_logger
import numpy as np
from lightrag.utils import wrap_embedding_func_with_attrs


# Setup logger
setup_logger("lightrag", level="INFO")


# Set your working directory
WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("GEMINI_API_KEY"),
        model_name="gemini-2.0-flash-exp",
        **kwargs
    )  # type: ignore 


# Configure the embedding model with proper attributes
@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=2048,
    model_name="models/text-embedding-004"
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts,
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/text-embedding-004"
    )


async def initialize_rag():
    """Initialize the LightRAG instance with rate limiting"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_name="gemini-2.0-flash",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        embedding_func_max_async=4,
        embedding_batch_num=8,
        llm_model_max_async=2, 
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        graph_storage="PGGraphStorage", # Use PG for graph storage
        vector_storage="PGVectorStorage", # Use PostgreSQL for vector storage
        doc_status_storage="PGDocStatusStorage", # Use PG for doc status storage
        kv_storage="PGKVStorage", # Use PG for key-value storage
    )
    
    # IMPORTANT: Initialize storage backends
    await rag.initialize_storages()
    return rag


async def main():
    rag = None
    try:
        print("Initializing LightRAG with Gemini models...")
        rag = await initialize_rag()
        
        # Read the book
        book_path = "Data/book-small.txt"
        print(f"\nReading book from: {book_path}")
        
        with open(book_path, "r", encoding="utf-8") as f:
            book_content = f.read()
        
        print(f"Book loaded: {len(book_content)} characters")
        
        # Insert the book content into RAG
        print("\nInserting book content into LightRAG...")
        print("This may take several minutes due to rate limiting...")
        await rag.ainsert(book_content)
        print("Book content inserted successfully!")
        
        # Test queries with different modes
        print("\n" + "="*60)
        print("Testing queries...")
        print("="*60)
        
        queries = [
            "What are the top themes in this story?",
            "Who are the main characters?",
            "What is the plot summary?"
        ]
        
        # Test different query modes
        modes = ["naive", "local", "global", "hybrid"]
        
        for query in queries[:1]:  # Test with first query
            print(f"\n\nQuery: {query}")
            print("-"*60)
            
            for mode in modes:
                print(f"\n[{mode.upper()} MODE]")
                try:
                    result = await rag.aquery(
                        query,
                        param=QueryParam(mode=mode)
                    )
                    print(result[:300] + "..." if len(result) > 300 else result)
                except Exception as e:
                    print(f"Error in {mode} mode: {e}")
        
        print("\n" + "="*60)
        print("RAG system is ready for queries!")
        print("="*60)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
