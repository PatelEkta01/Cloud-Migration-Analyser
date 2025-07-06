# AWS Migration Evaluator using RAG-based GenAI System

This repository contains a cloud-native application that helps users evaluate AWS migration strategies using a Retrieval-Augmented Generation (RAG) architecture powered by Generative AI. It is optimized for educational and small-business use cases.

---

## Project Summary

Users interact with a static web-based frontend hosted on Amazon S3 to submit cloud migration questions in natural language (e.g., _"How should I move my SQL Server to AWS?_"). The system responds with AI-generated advice on:

- Migration plans
- Recommended AWS services
- Estimated costs

This is achieved by combining vector search with text generation:

- **MiniLM Embedding** via SageMaker
- **FAISS Retrieval** via EC2
- **Text Generation** via SageMaker (Flan-T5)
- **Serverless orchestration** via Lambda + API Gateway

---

## Architecture Overview

### AWS Components:
- **S3** — Hosts static frontend & knowledge base chunks
- **API Gateway** — Accepts user questions
- **Lambda** — Coordinates retrieval + generation steps
- **SageMaker** — Embedding + Generation endpoints
- **EC2 (FAISS)** — Vector search on embeddings
- **CloudWatch** — Monitoring & alarms

---

## 💡 Key Features

-  **MiniLM Embedding + FAISS Retrieval**
-  **Flan-T5-based Answer Generation**
-  **Fully Serverless Frontend on S3**
-  **ETL Summary via SNS Email Alerts**
-  **CloudWatch Alarms on Lambda Failure/Latency**
