from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

CAREER_KNOWLEDGE_BASE = [
    {"id": "job_change_1", "category": "job_change", "title": "Evaluating Job Change Decisions",
     "content": "When considering a job change, evaluate total compensation including equity and benefits. Consider the company's growth trajectory and your potential role in it. Research the team culture through employee reviews and informational interviews. Assess the learning opportunities and skill development potential."},
    {"id": "job_change_2", "category": "job_change", "title": "Red Flags in Job Offers",
     "content": "Watch for vague role descriptions or constantly changing requirements. High turnover in the team or department is a warning sign. Unwillingness to provide clear information about compensation or equity. Pressure tactics to accept quickly without time to evaluate."},
    {"id": "career_switch_1", "category": "career_switch", "title": "Career Transition Strategies",
     "content": "Build transferable skills before making the switch. Create a financial buffer for potential income gaps. Network extensively in your target industry. Consider transitional roles that bridge your current and target fields."},
    {"id": "career_switch_2", "category": "career_switch", "title": "When to Consider Career Change",
     "content": "Persistent lack of motivation despite role changes within the field. Industry decline or limited growth opportunities. Significant misalignment between your values and industry practices."},
    {"id": "startup_1", "category": "startup", "title": "Startup Risk Assessment",
     "content": "Evaluate the founding team's experience and track record. Understand the funding situation and runway. Research the market size and competitive landscape. Clarify your equity stake, vesting schedule, and cliff period."},
    {"id": "startup_2", "category": "startup", "title": "Startup Compensation Trade-offs",
     "content": "Lower base salary is often offset by equity potential. Calculate the expected value of equity under different scenarios. Ensure you understand the liquidation preferences and dilution risks. Factor in the learning acceleration and network building opportunities."},
    {"id": "work_life_1", "category": "work_life_balance", "title": "Sustainable Career Pace",
     "content": "Burnout recovery typically takes 6-12 months. Regular high-stress periods without recovery lead to cumulative damage. Role changes alone don't fix systemic overwork patterns."},
    {"id": "skills_1", "category": "skills", "title": "Skill Development ROI",
     "content": "Technical skills have shorter half-lives than soft skills. Meta-skills like learning ability compound over time. Industry timing matters - entering growing fields early provides outsized returns."},
    {"id": "financial_1", "category": "financial", "title": "Financial Security in Career Moves",
     "content": "Maintain 6-12 months of expenses as a career transition buffer. Consider the impact on long-term savings and retirement plans. Factor in healthcare and insurance considerations."},
    {"id": "networking_1", "category": "networking", "title": "Strategic Networking",
     "content": "Quality connections outperform quantity. Focus on people 1-2 steps ahead in your target path. Provide value before asking for favors."},
    {"id": "education_1", "category": "education", "title": "Further Education Decisions",
     "content": "ROI of graduate education varies significantly by field and program. Consider opportunity cost of time out of workforce. Online alternatives may provide similar learning at lower cost."},
    {"id": "freelance_1", "category": "freelance", "title": "Freelance Transition",
     "content": "Build a client pipeline before leaving full-time employment. Plan for irregular income and self-employment taxes. Invest in systems for client management and invoicing."},
    {"id": "leadership_1", "category": "leadership", "title": "Leadership Role Transitions",
     "content": "Individual contributor to manager is often the hardest transition. Success metrics change from personal output to team effectiveness. Technical skills become less important than people skills."},
    {"id": "remote_1", "category": "remote_work", "title": "Remote Work Considerations",
     "content": "Remote work success depends heavily on self-discipline and communication skills. Consider the impact on career visibility and promotion opportunities. Factor in home office setup costs and ongoing expenses."}
]

@dataclass
class CareerDocument:
    id: str
    category: str
    title: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict = field(default_factory=dict)

class RAGService:
    def __init__(self, persist_dir: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.collection = None
        self.embedding_model = None
        self.documents: Dict[str, CareerDocument] = {}
        self.initialized = False

    async def initialize(self):
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(name="career_knowledge", metadata={"hnsw:space": "cosine"})
            await self._populate_knowledge_base()
            self.initialized = True
            return {"status": "initialized", "documents": len(self.documents)}
        except ImportError:
            self._initialize_fallback()
            return {"status": "fallback", "reason": "dependencies_unavailable"}

    def _initialize_fallback(self):
        for doc in CAREER_KNOWLEDGE_BASE:
            self.documents[doc["id"]] = CareerDocument(id=doc["id"], category=doc["category"], title=doc["title"], content=doc["content"])
        self.initialized = True

    async def _populate_knowledge_base(self):
        if not self.collection or not self.embedding_model:
            return

        existing = self.collection.get()
        existing_ids = set(existing['ids']) if existing and existing['ids'] else set()

        for doc in CAREER_KNOWLEDGE_BASE:
            if doc["id"] in existing_ids:
                self.documents[doc["id"]] = CareerDocument(id=doc["id"], category=doc["category"], title=doc["title"], content=doc["content"])
                continue

            full_text = f"{doc['title']}\n\n{doc['content']}"
            embedding = self.embedding_model.encode(full_text).tolist()

            self.collection.add(ids=[doc["id"]], embeddings=[embedding], documents=[full_text],
                metadatas=[{"category": doc["category"], "title": doc["title"]}])

            self.documents[doc["id"]] = CareerDocument(id=doc["id"], category=doc["category"], title=doc["title"], content=doc["content"], embedding=embedding)

    def retrieve(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[Dict]:
        if not self.initialized:
            return []

        if self.collection and self.embedding_model:
            try:
                query_embedding = self.embedding_model.encode(query).tolist()
                where_filter = {"category": category} if category else None
                results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where_filter)

                retrieved = []
                if results and results['ids'] and results['ids'][0]:
                    for i, doc_id in enumerate(results['ids'][0]):
                        doc = self.documents.get(doc_id)
                        if doc:
                            distance = results['distances'][0][i] if results.get('distances') else 0
                            retrieved.append({"id": doc.id, "title": doc.title, "content": doc.content,
                                "category": doc.category, "relevance": 1 - distance})
                return retrieved
            except Exception:
                pass

        return self._keyword_search(query, top_k, category)

    def _keyword_search(self, query: str, top_k: int, category: Optional[str] = None) -> List[Dict]:
        query_words = set(query.lower().split())
        scored_docs = []

        for doc in self.documents.values():
            if category and doc.category != category:
                continue
            doc_text = f"{doc.title} {doc.content}".lower()
            score = sum(1 for word in query_words if word in doc_text)
            if score > 0:
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [{"id": doc.id, "title": doc.title, "content": doc.content, "category": doc.category, "relevance": score / max(1, len(query_words))} for doc, score in scored_docs[:top_k]]

    def add_document(self, doc_id: str, category: str, title: str, content: str) -> bool:
        if not self.initialized:
            return False

        doc = CareerDocument(id=doc_id, category=category, title=title, content=content)
        self.documents[doc_id] = doc

        if self.collection and self.embedding_model:
            try:
                full_text = f"{title}\n\n{content}"
                embedding = self.embedding_model.encode(full_text).tolist()
                doc.embedding = embedding
                self.collection.add(ids=[doc_id], embeddings=[embedding], documents=[full_text],
                    metadatas=[{"category": category, "title": title}])
            except Exception:
                pass
        return True

    def add_media_content(self, media_id: str, media_type: str, title: str, content: str, source_url: str = None) -> bool:
        """Add extracted content from media (video, URL) to knowledge base"""
        if not self.initialized or not content or len(content.strip()) < 10:
            return False

        try:
            category_map = {
                'youtube': 'media_video',
                'video_file': 'media_video',
                'url': 'media_article',
                'article': 'media_article'
            }
            category = category_map.get(media_type, 'media_content')

            doc_id = f"media_{media_type}_{media_id}"

            full_content = f"Title: {title}\n"
            if source_url:
                full_content += f"Source: {source_url}\n"
            full_content += f"Type: {media_type}\n\n{content}"

            if len(full_content) > 100000:
                full_content = full_content[:50000] + "\n\n[...content truncated...]\n\n" + full_content[-50000:]

            return self.add_document(doc_id, category, title, full_content)

        except Exception as e:
            print(f"Error adding media content to RAG: {str(e)}")
            return False

    def add_transcript_chunks(self, video_id: str, transcript: str, title: str) -> bool:
        """Add video transcript as multiple documents for better retrieval"""
        if not self.initialized or not transcript or len(transcript.strip()) < 50:
            return False

        try:
            chunks = self._split_into_chunks(transcript, max_length=2000, overlap=200)

            doc_ids = []
            for i, chunk in enumerate(chunks):
                doc_id = f"transcript_{video_id}_{i}"
                chunk_title = f"{title} (Part {i+1})"
                success = self.add_document(doc_id, 'media_transcript', chunk_title, chunk)
                if success:
                    doc_ids.append(doc_id)

            return len(doc_ids) > 0

        except Exception as e:
            print(f"Error adding transcript chunks: {str(e)}")
            return False

    def _split_into_chunks(self, text: str, max_length: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        
        sentences = text.replace('\n\n', '\n').split('\n')
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_length and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap//10:]) if overlap > 0 else ""
                current_chunk = overlap_text + "\n" + sentence
            else:
                current_chunk += "\n" + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def get_media_context(self, query: str, user_id: str = None, max_docs: int = 5) -> str:
        """Retrieve context from user's media content"""
        results = self.retrieve(query, top_k=max_docs, category=None)

        media_results = [r for r in results if any(x in r.get('id', '') for x in ['media_', 'transcript_'])]

        if not media_results:
            return ""

        context_parts = ["Relevant content from your uploaded media:"]
        for i, result in enumerate(media_results[:3], 1):
            context_parts.append(f"\n{i}. {result['title']}")
            context_parts.append(f"   {result['content'][:200]}...")

        return "\n".join(context_parts)

    def get_context_for_decision(self, decision_type: str, description: str, max_docs: int = 3) -> str:
        query = f"{decision_type}: {description}"
        relevant_docs = self.retrieve(query, top_k=max_docs, category=self._map_decision_to_category(decision_type))

        if not relevant_docs:
            relevant_docs = self.retrieve(query, top_k=max_docs)

        if not relevant_docs:
            return ""

        context_parts = ["Relevant career insights:"]
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"\n{i}. {doc['title']}")
            content_preview = doc['content'][:300]
            context_parts.append(f"   {content_preview}...")

        return "\n".join(context_parts)

    def _map_decision_to_category(self, decision_type: str) -> Optional[str]:
        mapping = {
            'job_change': 'job_change', 'career_switch': 'career_switch', 'startup': 'startup',
            'education': 'education', 'freelance': 'freelance', 'promotion': 'leadership',
            'relocation': 'job_change', 'work_life_balance': 'work_life_balance'
        }
        return mapping.get(decision_type)

    def get_statistics(self) -> Dict:
        return {
            "initialized": self.initialized,
            "document_count": len(self.documents),
            "categories": list(set(doc.category for doc in self.documents.values())),
            "embedding_model": self.embedding_model_name if self.embedding_model else "fallback"
        }

    def cleanup(self):

        try:
            if self.embedding_model is not None:
                if hasattr(self.embedding_model, 'tokenizer'):
                    tokenizer = self.embedding_model.tokenizer
                    if hasattr(tokenizer, 'clean_up_tokenization_spaces'):
                        pass

                del self.embedding_model
                self.embedding_model = None

            if hasattr(self, 'client') and self.client is not None:
                del self.client
                self.client = None

            self.collection = None
            self.documents.clear()
            self.initialized = False

            import gc
            gc.collect()

            try:
                from multiprocessing import resource_tracker
            except ImportError:
                pass

        except Exception:
            pass
