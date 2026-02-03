import sys

print("Testing service imports...")

tests = [
    ("config", lambda: __import__('config')),
    ("NLPService", lambda: __import__('services.nlp_service', fromlist=['NLPService'])),
    ("RAGService", lambda: __import__('services.rag_service', fromlist=['RAGService'])),
    ("EnhancedOllamaService", lambda: __import__('services.ollama_service', fromlist=['EnhancedOllamaService'])),
    ("EnhancedRegretPredictor", lambda: __import__('models.ml_pipeline', fromlist=['EnhancedRegretPredictor'])),
    ("JournalService", lambda: __import__('services.journal_service', fromlist=['JournalService'])),
    ("EmotionDetectionService", lambda: __import__('services.emotion_detection_service', fromlist=['EmotionDetectionService'])),
]

results = []
for name, importer in tests:
    try:
        importer()
        print(f"✓ {name}")
        results.append(True)
    except Exception as e:
        print(f"✗ {name}: {str(e)[:100]}")
        results.append(False)

print(f"\nPassed: {sum(results)}/{len(results)}")

if all(results):
    print("\n--- Testing instantiation ---")
    try:
        from services.nlp_service import NLPService
        nlp = NLPService()
        print("✓ NLPService instance created")
    except Exception as e:
        print(f"✗ NLPService instantiation: {e}")
