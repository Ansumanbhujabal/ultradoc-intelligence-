Please read the instructions carefully: this is not a timed test. This is the test project only. You can take as much time as needed. 

Ultra Doc-Intelligence
Goal
Build a POC AI system that allows a user to upload a logistics document and interact with it using natural language questions. The system must retrieve relevant content, answer grounded questions, apply guardrails, and return a confidence score. This simulates an AI assistant inside a Transportation Management System (TMS).
Tech Stack: Any major AI/ML stack allowed
Here is a folder with some sample logistics documents for you to test with AI Engineer Test - Sample Documents, but you can use your own sample test data as needed.
Core Features to Build
Document Upload & Processing
User(s) can upload a logistics document (PDF, DOCX, or TXT) such as: Rate Confirmation, BOL, Shipment Instructions, Invoice etc.
System must: Parse text, Chunk intelligently, Create embeddings, Store in a vector index
Ask Questions About the Document
User(s) can ask questions about the uploaded document. Examples: What is the carrier rate? When is pickup scheduled? Who is the consignee?
System must: Use retrieval (RAG), answer only from document context, and return: answer, supporting source text, confidence score.
Guardrails and Confidence Score
System must include at least one hallucination guardrail:
Examples:
    • Refuse to answer if confidence is low
    • Answer only if retrieval similarity passes threshold
    • Say “Not found in document” when context missing
Return a confidence score with each answer. Can be based on:
    • retrieval similarity
    • chunk agreement
    • answer coverage
    • heuristic scoring
Structured Extraction
Provide an option to extract structured shipment data from the uploaded document below: Shipment_id, shipper, consignee, pickup_datetime, delivery_datetime, equipment_type, mode, rate, currency, weight, carrier_name
Return JSON with nulls if missing.
Minimal UI (Required)
Provide a lightweight UI where a reviewer can:
    • Upload a document
    • Ask questions
    • View answers + sources + confidence
    • Run structured extraction
UI design is not graded — usability is.
API Endpoints
POST /upload | POST /ask | POST /extract
Submission Requirements
Provide:
    • GitHub repository
    • Hosted UI link
    • Runnable locally
    • README including: architecture, chunking strategy, retrieval method, guardrails approach, confidence scoring method, failure cases, improvement ideas

Evaluation Criteria
We evaluate:
    • Retrieval grounding quality
    • Extraction accuracy
    • Guardrail effectiveness
    • Confidence scoring logic
    • Code structure
    • End-to-end usability
    • Practical AI engineering judgment
Clarity and correctness matter more than framework complexity.

