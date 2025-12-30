#!/usr/bin/env python3
# validate_system.py
import sys
import os
sys.path.append('src')

print("="*60)
print("XENO BOT - SYSTEM VALIDATION")
print("="*60)

def check_file(filepath, description):
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: NOT FOUND")
        return False

print("\n📁 FILE CHECK:")
print("-"*40)

# Check all required files
files_to_check = [
    ("Chroma DB", "/tmp/xeno_db", os.path.isdir),
    ("SQLite DB", "xeno_memory.db", os.path.exists),
    ("Knowledge Base", "XENO_Uganda_KnowledgeBase_Advisory.json", os.path.exists),
    (".env file", ".env", os.path.exists),
]

all_good = True
for name, path, check_func in files_to_check:
    if check_func(path):
        if name == "Chroma DB":
            # Count files in Chroma DB
            try:
                file_count = len(os.listdir(path))
                print(f"   Contains {file_count} files")
            except:
                pass
    else:
        all_good = False

print("\n🧪 MODULE IMPORT CHECK:")
print("-"*40)

modules = [
    ("config", "import config"),
    ("logger", "import logger"),
    ("utils", "import utils"),
    ("neo4j_client", "import neo4j_client"),
    ("vector_store", "import vector_store"),
    ("memory", "import memory"),
    ("knowledge_base", "import knowledge_base"),
    ("graph.schema", "from graph import schema"),
    ("graph.knowledge_ingest", "from graph import knowledge_ingest"),
    ("cognitive.cognitive_memory", "from cognitive import cognitive_memory"),
    ("cognitive.reasoning", "from cognitive import reasoning"),
    ("cognitive.retrieval", "from cognitive import retrieval"),
    ("intent_classifier", "import intent_classifier"),
    ("response_generator", "import response_generator"),
]

import_errors = []
for name, code in modules:
    try:
        exec(code)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {str(e)[:50]}")
        import_errors.append((name, e))

print("\n" + "="*60)
print("VALIDATION SUMMARY")
print("="*60)

if all_good and len(import_errors) == 0:
    print("\n🎉 ALL SYSTEMS READY!")
    print("\nNext steps:")
    print("1. Run tests: python run_tests.py")
    print("2. Start app: python app.py")
else:
    print("\n⚠️  ISSUES FOUND:")
    
    if not all_good:
        print("- Some database/files missing")
    
    if import_errors:
        print(f"- {len(import_errors)} module import errors")
        for name, error in import_errors[:3]:  # Show first 3 errors
            print(f"  • {name}: {error}")
    
    print("\nTroubleshooting:")
    if not os.path.exists("xeno_memory.db"):
        print("- Run: python setup_sqlite.py")
    if import_errors:
        print("- Check dependencies: pip install -r requirements.txt")

print("="*60)