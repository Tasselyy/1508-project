import os
from openai import OpenAI

# Initialize API client.
# We use standard OpenAI package.
# It works for Groq or DeepSeek by changing base_url.
client = OpenAI(
    api_key="YOUR_API_KEY_HERE",
    base_url="https://api.groq.com/openai/v1"
)

def get_chunk_texts(retrieved_ids, chunks):
    # Get text for retrieved chunk IDs.
    # Create a lookup dictionary.
    # Key is chunk_id. Value is text.
    chunk_dict = {c["chunk_id"]: c["text"] for c in chunks}
    
    texts = []
    
    # Loop through retrieved IDs.
    for cid in retrieved_ids:
        if cid in chunk_dict:
            # Add text to list.
            texts.append(chunk_dict[cid])
            
    return texts

def generate_answer(query, context_texts):
    # Call LLM to generate an answer.
    # Join all chunks into one string.
    # Separate chunks with double newlines.
    context_str = "\n\n---\n\n".join(context_texts)
    
    # Build the system prompt.
    # Tell the model how to behave.
    system_prompt = (
        "You are a helpful QA bot. "
        "Answer the question using ONLY the provided context. "
        "If context has no answer, say 'I do not know'. "
        "Briefly cite your sources."
    )
    
    # Build the user prompt.
    user_prompt = f"Context:\n{context_str}\n\nQuestion:\n{query}"
    
    try:
        # Call the API.
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        
        # Return the generated text.
        return response.choices[0].message.content
        
    except Exception as e:
        # Catch any API errors.
        print(f"API Error: {e}")
        return "Error generating answer."

def run_generation_step(sampled_queries, chunks, top_k=5):
    # Run generation for all sampled queries.
    
    print(f"  Starting LLM generation for {len(sampled_queries)} queries...")
    
    for q in sampled_queries:
        # Get the top K IDs from ColBERT.
        # You can change this to biencoder_retrieved_ids if needed.
        top_ids = q.get("colbert_retrieved_ids", [])[:top_k]
        
        # Look up the actual text for these IDs.
        context_texts = get_chunk_texts(top_ids, chunks)
        
        # Generate the final answer.
        answer = generate_answer(q["query"], context_texts)
        
        # Save the answer back into the query dictionary.
        q["generated_answer"] = answer
        
    print("  Generation step finished.")