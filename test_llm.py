
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("test_cases.csv")

questions = df["question"].tolist()

# -------------------------------
# Load models
# -------------------------------
generator = pipeline("text-generation", model="gpt2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings
question_embeddings = semantic_model.encode(questions, convert_to_tensor=True)

# -------------------------------
# Accuracy tracking
# -------------------------------
correct = 0
total = 0

risk_stats = {
    "High": {"correct": 0, "total": 0},
    "Medium": {"correct": 0, "total": 0},
    "Low": {"correct": 0, "total": 0}
}

print("Clinical LLM Evaluator")
print("Type 'exit' to stop\n")

# -------------------------------
# Chat loop
# -------------------------------
while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("\n FINAL RESULTS")
        print(f"Overall Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")

        for risk in risk_stats:
            r = risk_stats[risk]
            if r["total"] > 0:
                print(f"{risk} Risk Accuracy: {r['correct']}/{r['total']} = {(r['correct']/r['total'])*100:.2f}%")

        break

    # -------------------------------
    # Generate LLM answer
    # -------------------------------
    prompt = f"Question: {user_input}\nAnswer:"

    result = generator(
        prompt,
        max_new_tokens=60,
        temperature=0.7,
        repetition_penalty=1.5,
        no_repeat_ngram_size=2,
        pad_token_id=50256
    )

    full_output = result[0]["generated_text"]
    llm_answer = full_output.split("Answer:")[-1].strip()

    # Clean incomplete sentence
    if "." in llm_answer:
        llm_answer = llm_answer[:llm_answer.rfind(".")+1]

    print("\nBot:", llm_answer)

    # -------------------------------
    # Semantic matching
    # -------------------------------
    user_embedding = semantic_model.encode(user_input, convert_to_tensor=True)
    scores = util.cos_sim(user_embedding, question_embeddings)[0]

    best_idx = scores.argmax().item()
    best_score = scores[best_idx]

    if best_score > 0.6:
        matched_row = df.iloc[best_idx]
        ground_truth = matched_row["ground_truth"]
        risk = matched_row["risk_level"]

        print("Matched Question:", matched_row["question"])
        print(" Ground Truth:", ground_truth)
        print("Risk Level:", risk)

        total += 1
        risk_stats[risk]["total"] += 1

        if ground_truth.lower() in llm_answer.lower():
            print("Correct")
            correct += 1
            risk_stats[risk]["correct"] += 1
        else:
            print("Incorrect")

        print(f"Accuracy: {correct}/{total} = {(correct/total)*100:.2f}%")

    else:
        print("⚠ No good semantic match found")

    print("-" * 50)