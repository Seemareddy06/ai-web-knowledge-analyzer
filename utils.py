from rag_pipeline import client


def simple_llm_call(prompt):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def generate_summary(text):
    return simple_llm_call(f"Provide a clear summary:\n{text}")


def extract_insights(text):
    return simple_llm_call(f"Extract top 5 key insights:\n{text}")


def compare_docs(text):
    return simple_llm_call(f"Compare the following content and give similarities and differences:\n{text}")


def generate_questions(text):
    return simple_llm_call(f"Generate 5 important questions:\n{text}")