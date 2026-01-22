#!/usr/bin/env python3
# setup_chroma.py
import sys
import os
sys.path.append('src')

print("🔧 Setting up Chroma Vector Database...")
print("="*50)

try:
    from vector_store import initialize_vector_store
    
    # Check if Chroma DB already exists
    chroma_path = "/tmp/xeno_db"
    if os.path.exists(chroma_path):
        print(f"⚠️  Chroma DB already exists at {chroma_path}")
        response = input("Recreate it? (y/n): ").strip().lower()
        
        if response == 'y':
            import shutil
            shutil.rmtree(chroma_path)
            print(f"🗑️  Deleted existing Chroma DB")
            recreate = True
        else:
            recreate = False
            print("⏭️  Using existing Chroma DB")
    else:
        recreate = True
    
    if recreate:
        print("🔄 Initializing vector store...")
        
        # This will create the Chroma database
        collection, vector_store, retriever = initialize_vector_store()
        
        print(f"✅ Chroma DB created successfully!")
        print(f"   Location: {chroma_path}")
        print(f"   Collection: {collection.name}")
        print(f"   Documents: {collection.count()}")
        
        # Test the retriever
        print("\n🧪 Testing retrieval...")
        test_query = "What is XENO Bot?"
        results = retriever.get_relevant_documents(test_query)
        print(f"   Query: '{test_query}'")
        print(f"   Retrieved {len(results)} documents")
        
    else:
        # Just initialize to verify it works
        collection, vector_store, retriever = initialize_vector_store()
        print(f"✅ Existing Chroma DB loaded")
        print(f"   Collection: {collection.name}")
        print(f"   Documents: {collection.count()}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*50)