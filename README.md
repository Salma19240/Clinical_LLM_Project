Clinical LLM Evaluation Pipeline

Project Overview

This project focuses on evaluating the reliability of Large Language Model (LLM)-generated responses in healthcare-related scenarios.

The pipeline uses GPT-2 to generate responses to clinical questions and applies semantic matching techniques to compare generated answers against a curated ground-truth dataset.

The project explores the challenges of factual correctness, semantic similarity, hallucinations, and the reliability of AI-generated clinical information.

Objectives

- Evaluate AI-generated responses to healthcare-related questions.
- Analyze responses concerning medications, symptoms, and treatments.
- Compare generated responses with reference answers.
- Explore semantic similarity using Sentence Transformers.
- Identify potential factual inconsistencies and hallucination risks.
- Understand the limitations of language models in clinical applications.

Technologies Used

- Python
- GPT-2
- Hugging Face Transformers
- Sentence Transformers
- Pandas

Project Files

File| Description
"test_llm.py"| Python implementation of the evaluation pipeline
"test_xqsee.csv"| Clinical question and reference-answer dataset

Methodology

1. Input: Clinical questions are provided as test cases.
2. Response Generation: GPT-2 generates responses to the questions.
3. Semantic Matching: Sentence Transformers are used to analyze semantic similarity between generated responses and reference answers.
4. Evaluation: Generated responses are compared against the curated reference dataset.
5. Analysis: The evaluation explores response quality, factual consistency, and potential hallucination risks.

Research Focus

The project investigates an important challenge in healthcare-related AI applications: fluent language generation does not necessarily guarantee factual correctness.

It highlights the importance of evaluating LLM-generated content before considering its use in sensitive domains such as healthcare.

Limitations

- Semantic similarity does not guarantee factual correctness.
- GPT-2-generated clinical responses should not be considered medical advice.
- The evaluation depends on the quality and coverage of the reference dataset.
- This project is an experimental evaluation pipeline and is not clinically validated.

Future Enhancements

- Experiment with additional language models such as FLAN-T5.
- Introduce more comprehensive factual consistency metrics.
- Expand the clinical evaluation dataset.
- Explore automated hallucination detection.
- Improve evaluation through multiple complementary metrics.

Disclaimer

This project is intended for educational and research purposes only. It is not designed for clinical diagnosis, treatment recommendations, or medical decision-making.