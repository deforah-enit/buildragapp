"""
Graph RAG Module v2 - Production Grade
=====================================
- Analyzes document type first
- Extracts ALL entities (no limits)
- Creates domain-aware relationships
- Builds hierarchical, structured graphs
- Works for ANY document type
"""

import logging
import json
from typing import List, Dict, Tuple, Optional
import os
from dotenv import load_dotenv
from openai import OpenAI

try:
    from neo4j import GraphDatabase, Session
except ImportError:
    GraphDatabase = None
    Session = None

load_dotenv()
logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """Production-grade knowledge graph construction with document understanding"""
    
    def __init__(self):
        """Initialize graph RAG pipeline"""
        self.driver = None
        self.openai_client = None
        self.chunk_map = {}
        self.doc_type = None
        self.doc_structure = None
        
        self._initialize_neo4j()
        self._initialize_openai()
    
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            if GraphDatabase is None:
                raise ImportError("neo4j driver not installed")
            
            uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
            username = os.getenv("NEO4J_USERNAME", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "password")
            
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("✓ Neo4j connection established")
        except Exception as e:
            logger.warning(f"Neo4j initialization warning: {e}")
            self.driver = None
    
    def _initialize_openai(self):
        """Initialize OpenAI client"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            
            self.openai_client = OpenAI(api_key=api_key)
            logger.info("✓ OpenAI client initialized")
        except Exception as e:
            logger.error(f"OpenAI initialization error: {e}")
            raise
    
    # ====== PHASE 1: Document Type Analysis ======
    
    def _analyze_document_type(self, text: str) -> Dict:
        """
        Analyze document to understand its type and structure
        
        Returns: {
            'type': 'meal_plan' | 'medical' | 'research' | 'news' | etc,
            'categories': ['PROTEIN', 'CARBS', ...],
            'relationship_types': ['BELONGS_TO', 'HAS_QUANTITY', ...],
            'hierarchy': True/False,
            'description': 'What this document is about'
        }
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze this document and return structured information as JSON.

Analyze for:
1. Document type (meal_plan, medical, research, news, technical, educational, etc.)
2. Main categories/groupings in the document
3. What relationships matter for this document type
4. Is there a clear hierarchy or structure?

Return EXACTLY this JSON format (no markdown, no extra text):
{
    "type": "meal_plan|medical|research|news|other",
    "categories": ["CATEGORY1", "CATEGORY2", ...],
    "relationship_types": ["REL_TYPE1", "REL_TYPE2", ...],
    "has_hierarchy": true/false,
    "description": "brief description of document"
}

Examples:
Meal plan: {"type": "meal_plan", "categories": ["PROTEIN", "CARBS", "FATS", "FLUIDS", "FRUITS", "VEGETABLES"], "relationship_types": ["BELONGS_TO", "HAS_QUANTITY", "RECOMMENDED_DAILY"], "has_hierarchy": true, "description": "..."}

Medical: {"type": "medical", "categories": ["SYMPTOM", "DISEASE", "TREATMENT", "MEDICATION"], "relationship_types": ["CAUSES", "TREATS", "SIDE_EFFECT", "PRESCRIBED_FOR"], "has_hierarchy": false, "description": "..."}"""
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this document:\n{text[:5000]}"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            doc_structure = json.loads(response_text)
            
            logger.info(f"📋 Document Type: {doc_structure.get('type')}")
            logger.info(f"📁 Categories: {doc_structure.get('categories')}")
            logger.info(f"🔗 Relationship Types: {doc_structure.get('relationship_types')}")
            
            return doc_structure
        
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            # Return generic structure as fallback
            return {
                "type": "generic",
                "categories": ["ENTITY"],
                "relationship_types": ["RELATED_TO"],
                "has_hierarchy": False,
                "description": "Generic document"
            }
    
    # ====== PHASE 2: Extract ALL Entities ======
    
    def _extract_quantities_from_text(self, text: str) -> Dict[str, str]:
        """
        Extract quantities using pattern matching (more reliable than GPT)
        Looks for patterns like "150-180 grams Item" or "1 medium Apple"
        
        Returns: {"Item": "quantity", ...}
        """
        import re
        quantities = {}
        
        # Pattern 1: "NUMBER-NUMBER unit Item" (e.g., "150-180 grams Pomegranates")
        pattern1 = r'(\d+[-–]\d+)\s*(grams?|g|ml|liters?|l|cups?|tbsp?|tsp?|oz|lbs?)\s+([A-Za-z\s/]+?)(?=\s+[A-Z]|$|\n)'
        matches1 = re.finditer(pattern1, text, re.IGNORECASE)
        for match in matches1:
            qty = match.group(1) + " " + match.group(2)
            item = match.group(3).strip()
            if item and len(item) > 2:
                quantities[item] = qty
        
        # Pattern 2: "NUMBER unit Item" (e.g., "1 medium Apple", "1 Kiwi")
        pattern2 = r'(\d+)\s*(medium|large|small|medium-sized)?\s+([A-Za-z\s/]+?)(?=\s+[A-Z]|$|\n)'
        matches2 = re.finditer(pattern2, text, re.IGNORECASE)
        for match in matches2:
            qty = match.group(1)
            size = match.group(2)
            item = match.group(3).strip()
            
            if item and len(item) > 2:
                # Add size modifier if present
                if size:
                    qty = f"{qty} {size}"
                quantities[item] = qty
        
        logger.info(f"Pattern-based extraction found {len(quantities)} quantities")
        return quantities
    
    def _extract_all_entities(self, text: str, doc_type: str) -> List[Dict]:
        """
        Extract ALL entities from document (no limits!)
        Enhanced with pattern-based quantity extraction
        
        Returns: [
            {"name": "Tofu", "type": "PROTEIN", "quantity": "fist size", "description": "..."},
            ...
        ]
        """
        try:
            # FIRST: Extract quantities using pattern matching (more reliable)
            pattern_quantities = self._extract_quantities_from_text(text)
            logger.info(f"Found {len(pattern_quantities)} quantities via pattern matching")
            
            if doc_type == "meal_plan":
                extraction_prompt = """Extract ALL food items, nutrients, and ingredients from this meal plan.

For each item, extract:
- name: The food/ingredient name
- category: PROTEIN | CARBS | FATS | FLUIDS | FRUITS | VEGETABLES | NUTRIENT
- quantity: Recommended serving (copy EXACTLY from the document if available, e.g., "fist size", "2-3 liters", "150-180 grams")
- description: Brief description if available

IMPORTANT: 
1. Extract EVERY item. No limits. No filtering.
2. Copy quantities EXACTLY from the document - don't guess or interpret
3. If a specific quantity is shown (like "150-180 grams"), use that exact number

Return as JSON array (no markdown):
[
  {"name": "Tofu", "category": "PROTEIN", "quantity": "fist size", "description": "plant-based protein"},
  {"name": "Water", "category": "FLUIDS", "quantity": "2-3 liters", "description": "hydration"},
  ...
]"""
            
            elif doc_type == "medical":
                extraction_prompt = """Extract ALL medical entities from this document.

For each entity, extract:
- name: Entity name
- category: SYMPTOM | DISEASE | TREATMENT | MEDICATION | TEST | CONDITION
- description: What it is

Return as JSON array (no markdown):
[
  {"name": "headache", "category": "SYMPTOM", "description": "pain in head"},
  ...
]"""
            
            elif doc_type == "research":
                extraction_prompt = """Extract ALL concepts, findings, and entities from this research document.

For each entity, extract:
- name: Concept/finding name
- category: CONCEPT | FINDING | METHOD | CONCLUSION | HYPOTHESIS
- description: What it means

Return as JSON array (no markdown):
[
  {"name": "machine learning", "category": "CONCEPT", "description": "..."},
  ...
]"""
            
            else:  # Generic
                extraction_prompt = """Extract ALL important entities from this document.

For each entity, extract:
- name: Entity name
- category: The type of entity
- description: What it is

Return as JSON array (no markdown):
[
  {"name": "entity1", "category": "TYPE1", "description": "..."},
  ...
]"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": f"Extract from:\n{text}"}
                ],
                temperature=0.3,
                max_tokens=4000  # Increased for more entities
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            entities = json.loads(response_text)
            
            # SECOND: Enhance entities with pattern-matched quantities
            if doc_type == "meal_plan" and pattern_quantities:
                for entity in entities:
                    entity_name = entity.get('name', '')
                    # Try to find matching quantity from pattern extraction
                    for pattern_name, pattern_qty in pattern_quantities.items():
                        # Fuzzy match: check if names are similar
                        if entity_name.lower() in pattern_name.lower() or pattern_name.lower() in entity_name.lower():
                            if not entity.get('quantity') or entity.get('quantity') == 'varies':
                                entity['quantity'] = pattern_qty
                                logger.info(f"Updated {entity_name} quantity from pattern: {pattern_qty}")
                                break
            
            logger.info(f"✅ Extracted {len(entities)} entities")
            logger.info(f"   Sample: {[e.get('name') for e in entities[:5]]}")
            
            return entities
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in entity extraction: {e}")
            logger.error(f"Response was: {response_text[:200]}")
            return []
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return []
    
    # ====== PHASE 3: Extract Domain-Aware Relationships ======
    
    def _extract_relationships(
        self, 
        text: str, 
        doc_type: str, 
        entities: List[Dict]
    ) -> List[Dict]:
        """
        Extract domain-aware relationships
        
        Returns: [
            {"subject": "Tofu", "relationship": "BELONGS_TO", "object": "PROTEIN"},
            {"subject": "Tofu", "relationship": "HAS_QUANTITY", "object": "fist size"},
            ...
        ]
        """
        try:
            entity_names = [e.get('name') for e in entities]
            
            if doc_type == "meal_plan":
                rel_prompt = """Extract relationships between food items in this meal plan.

Relationship types:
- BELONGS_TO: Food belongs to category (e.g., "Tofu BELONGS_TO PROTEIN")
- HAS_QUANTITY: Food has a serving size (e.g., "Tofu HAS_QUANTITY fist size")
- CONTAINS_NUTRIENT: Food contains a nutrient (e.g., "Eggs CONTAINS_NUTRIENT protein")
- PAIRS_WITH: Foods that work well together (e.g., "Chicken PAIRS_WITH Rice")
- IS_ALTERNATIVE: Foods that are alternatives (e.g., "Tofu IS_ALTERNATIVE Chicken")

IMPORTANT: Only use relationships that exist in the document.

Return as JSON array (no markdown):
[
  {"subject": "Tofu", "relationship": "BELONGS_TO", "object": "PROTEIN"},
  {"subject": "Tofu", "relationship": "HAS_QUANTITY", "object": "fist size"},
  ...
]"""
            
            elif doc_type == "medical":
                rel_prompt = """Extract medical relationships.

Relationship types:
- CAUSES: Condition causes symptom (e.g., "Diabetes CAUSES Hyperglycemia")
- TREATS: Treatment treats disease
- SIDE_EFFECT: Medication has side effect
- SYMPTOM_OF: Symptom is symptom of disease
- DIAGNOSED_BY: Disease diagnosed by test

Return as JSON array (no markdown):
[
  {"subject": "Diabetes", "relationship": "CAUSES", "object": "Hyperglycemia"},
  ...
]"""
            
            else:  # Generic
                rel_prompt = """Extract meaningful relationships between entities.

Return as JSON array (no markdown):
[
  {"subject": "entity1", "relationship": "RELATED_TO", "object": "entity2"},
  ...
]"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": rel_prompt},
                    {"role": "user", "content": f"Extract from:\n{text}\n\nEntities: {entity_names[:20]}"}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            response_text = response.choices[0].message.content.strip()
            relationships = json.loads(response_text)
            
            logger.info(f"✅ Extracted {len(relationships)} relationships")
            logger.info(f"   Types: {set(r.get('relationship') for r in relationships)}")
            
            return relationships
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in relationship extraction: {e}")
            return []
        except Exception as e:
            logger.error(f"Relationship extraction error: {e}")
            return []
    
    # ====== PHASE 4: Build Structured Graph ======
    
    def _build_structured_graph(
        self,
        doc_type: str,
        categories: List[str],
        entities: List[Dict],
        relationships: List[Dict]
    ) -> int:
        """
        Build structured graph in Neo4j
        
        Returns: Number of relationships created
        """
        if not self.driver:
            logger.warning("Neo4j not available")
            return 0
        
        relation_count = 0
        
        try:
            with self.driver.session() as session:
                # Step 1: Create category nodes
                logger.info(f"Creating {len(categories)} category nodes...")
                for category in categories:
                    session.run(
                        """
                        MERGE (c:Category {name: $name})
                        SET c.type = $doc_type
                        """,
                        name=category,
                        doc_type=doc_type
                    )
                
                # Step 2: Create ALL entity nodes (NO LIMITS!)
                logger.info(f"Creating {len(entities)} entity nodes...")
                for entity in entities:
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        SET e.category = $category,
                            e.quantity = $quantity,
                            e.description = $description,
                            e.doc_type = $doc_type
                        """,
                        name=entity.get('name'),
                        category=entity.get('category'),
                        quantity=entity.get('quantity'),
                        description=entity.get('description'),
                        doc_type=doc_type
                    )
                
                # Step 3: Create domain-aware relationships
                logger.info(f"Creating {len(relationships)} relationships...")
                for rel in relationships:
                    subject = rel.get('subject')
                    relationship = rel.get('relationship')
                    object_ = rel.get('object')
                    
                    if relationship == "BELONGS_TO":
                        # Food belongs to category
                        session.run(
                            """
                            MATCH (e:Entity {name: $subject})
                            MATCH (c:Category {name: $object})
                            MERGE (e)-[:BELONGS_TO]->(c)
                            """,
                            subject=subject,
                            object=object_
                        )
                    
                    elif relationship == "HAS_QUANTITY":
                        # Food has a quantity/serving size
                        session.run(
                            """
                            MATCH (e:Entity {name: $subject})
                            SET e.quantity = $quantity
                            """,
                            subject=subject,
                            quantity=object_
                        )
                    
                    elif relationship == "CONTAINS_NUTRIENT":
                        # Food contains nutrient
                        session.run(
                            """
                            MATCH (e:Entity {name: $subject})
                            MERGE (n:Entity {name: $object})
                            MERGE (e)-[:CONTAINS_NUTRIENT]->(n)
                            """,
                            subject=subject,
                            object=object_
                        )
                    
                    elif relationship == "PAIRS_WITH":
                        # Foods that pair well
                        session.run(
                            """
                            MATCH (e1:Entity {name: $subject})
                            MATCH (e2:Entity {name: $object})
                            MERGE (e1)-[:PAIRS_WITH]->(e2)
                            """,
                            subject=subject,
                            object=object_
                        )
                    
                    elif relationship == "CAUSES":
                        # Medical: condition causes symptom
                        session.run(
                            """
                            MATCH (e1:Entity {name: $subject})
                            MATCH (e2:Entity {name: $object})
                            MERGE (e1)-[:CAUSES]->(e2)
                            """,
                            subject=subject,
                            object=object_
                        )
                    
                    elif relationship == "TREATS":
                        # Medical: treatment treats disease
                        session.run(
                            """
                            MATCH (e1:Entity {name: $subject})
                            MATCH (e2:Entity {name: $object})
                            MERGE (e1)-[:TREATS]->(e2)
                            """,
                            subject=subject,
                            object=object_
                        )
                    
                    else:
                        # Generic relationship
                        rel_type = relationship.upper()
                        session.run(
                            f"""
                            MATCH (e1:Entity {{name: $subject}})
                            MATCH (e2:Entity {{name: $object}})
                            MERGE (e1)-[:{rel_type}]->(e2)
                            """,
                            subject=subject,
                            object=object_
                        )
                    
                    relation_count += 1
                
                logger.info(f"✅ Created {relation_count} relationships")
        
        except Exception as e:
            logger.error(f"Graph building error: {e}")
        
        return relation_count
    
    # ====== Public API ======
    
    def ingest_chunks(self, chunks: List[Dict]) -> int:
        """
        Ingest document chunks using new architecture
        """
        logger.info(f"Ingesting {len(chunks)} chunks...")
        
        # Combine all chunks into full text for analysis
        full_text = "\n".join([c['text'] for c in chunks])
        
        # PHASE 1: Analyze document type
        logger.info("📋 PHASE 1: Analyzing document structure...")
        doc_structure = self._analyze_document_type(full_text)
        self.doc_type = doc_structure.get('type')
        self.doc_structure = doc_structure
        
        # PHASE 2: Extract ALL entities
        logger.info("🔍 PHASE 2: Extracting all entities...")
        entities = self._extract_all_entities(full_text, self.doc_type)
        
        if not entities:
            logger.warning("No entities extracted!")
            return 0
        
        # PHASE 3: Extract relationships
        logger.info("🔗 PHASE 3: Extracting relationships...")
        relationships = self._extract_relationships(
            full_text, 
            self.doc_type, 
            entities
        )
        
        # PHASE 4: Build graph
        logger.info("🏗️  PHASE 4: Building structured graph...")
        categories = doc_structure.get('categories', [])
        relation_count = self._build_structured_graph(
            self.doc_type,
            categories,
            entities,
            relationships
        )
        
        logger.info(f"""
=====================================
✅ INGESTION COMPLETE
=====================================
Document Type: {self.doc_type}
Entities: {len(entities)}
Relationships: {relation_count}
Categories: {categories}
=====================================
        """)
        
        return relation_count
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search using document-aware approach
        Returns chunks with entity metadata (quantity, description, etc.)
        """
        if not self.driver:
            logger.warning("Neo4j not available")
            return []
        
        results = []
        seen_chunks = set()
        
        try:
            # Extract entities from query using same doc type understanding
            entities, _ = self._extract_all_entities(query, self.doc_type or "generic")
            
            if not entities:
                return []
            
            with self.driver.session() as session:
                # Search for matching entities and their chunks
                for entity in entities[:5]:
                    entity_name = entity.get('name')
                    
                    # Get entity properties
                    entity_result = session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        RETURN e.name as name, 
                               e.description as description,
                               e.quantity as quantity,
                               e.category as category
                        LIMIT 1
                        """,
                        name=entity_name
                    )
                    
                    entity_record = entity_result.single()
                    
                    if not entity_record:
                        continue
                    
                    entity_data = {
                        'name': entity_record.get('name'),
                        'description': entity_record.get('description'),
                        'quantity': entity_record.get('quantity'),
                        'category': entity_record.get('category')
                    }
                    
                    # Get chunks containing this entity
                    chunk_result = session.run(
                        """
                        MATCH (e:Entity {name: $name})-[:APPEARS_IN]->(c:Chunk)
                        RETURN c.id as chunk_id, c.text as text, c.source as source
                        """,
                        name=entity_name
                    )
                    
                    for chunk_record in chunk_result:
                        chunk_id = chunk_record.get('chunk_id')
                        
                        # Avoid duplicates
                        if chunk_id in seen_chunks:
                            continue
                        
                        seen_chunks.add(chunk_id)
                        
                        # Build context string with entity metadata
                        entity_context = f"\n[ENTITY: {entity_data.get('name')}"
                        if entity_data.get('quantity'):
                            entity_context += f" | QUANTITY: {entity_data.get('quantity')}"
                        if entity_data.get('description'):
                            entity_context += f" | DESCRIPTION: {entity_data.get('description')}"
                        entity_context += "]\n"
                        
                        text_with_context = entity_context + chunk_record.get('text', '')
                        
                        results.append({
                            'text': text_with_context,
                            'chunk_id': chunk_id,
                            'source': chunk_record.get('source'),
                            'entity_name': entity_data.get('name'),
                            'entity_quantity': entity_data.get('quantity'),
                            'entity_description': entity_data.get('description'),
                            'score': 0.8,
                            'search_type': 'graph'
                        })
                        
                        # Log for debugging
                        logger.info(f"Graph search found: {entity_data.get('name')} | Qty: {entity_data.get('quantity')}")
        
        except Exception as e:
            logger.error(f"Graph search error: {e}")
        
        return results[:top_k]
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            if self.driver is None:
                return False
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def reset(self):
        """Reset knowledge graph"""
        logger.info("Resetting knowledge graph...")
        
        if not self.driver:
            logger.warning("Neo4j not available")
            return
        
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("Knowledge graph cleared")
        except Exception as e:
            logger.error(f"Reset error: {e}")
        
        self.chunk_map = {}
        self.doc_type = None
        self.doc_structure = None