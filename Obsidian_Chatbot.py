import os
import glob
import openai
import faiss
import numpy as np
import chainlit as cl
from chainlit.input_widget import Select
from dotenv import load_dotenv
from urllib.parse import quote
import subprocess
import requests
import json

load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to your Obsidian vault
vault_path = os.getenv("OBSIDIAN_VAULT_PATH")

# Name of your Obsidian vault
vault_name = os.getenv("OBSIDIAN_VAULT_NAME")

# Path to a specific file in the vault
tasks_file_path = os.path.join(vault_path, 'Tasks', 'Tasks.md')

# Load the content of the file
with open(tasks_file_path, 'r', encoding='utf-8') as f:
    tasks_content = f.read()

# Function to load all .md files from the vault
def load_notes(vault_path):
    notes = []
    titles = []
    file_paths = glob.glob(os.path.join(vault_path, '**/*.md'), recursive=True)
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            notes.append(content)
            title = os.path.splitext(os.path.basename(file_path))[0]
            titles.append(title)
    return notes, titles

# Function to create embeddings
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model='text-embedding-3-small'
    )
    return response.data[0].embedding

# Check if embeddings and index already exist
if not (os.path.exists('embeddings.npy') and os.path.exists('titles.npy') and os.path.exists('notes_index.faiss') and os.path.exists('notes.npy')):
    print("Generating embeddings and creating FAISS index...")
    # Load notes and titles
    notes, titles = load_notes(vault_path)

    # Create embeddings for all notes
    embeddings = []
    for note in notes:
        embedding = get_embedding(note)
        embeddings.append(embedding)

    # Convert embeddings to numpy array
    embedding_matrix = np.array(embeddings).astype('float32')

    # Create FAISS index
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    # Save embeddings, titles, notes, and index
    np.save('embeddings.npy', embedding_matrix)
    np.save('titles.npy', np.array(titles))
    np.save('notes.npy', np.array(notes))
    faiss.write_index(index, 'notes_index.faiss')
else:
    print("Loading embeddings and FAISS index from disk...")
    # Load embeddings, titles, notes, and index
    embedding_matrix = np.load('embeddings.npy')
    titles = np.load('titles.npy').tolist()
    notes = np.load('notes.npy').tolist()
    index = faiss.read_index('notes_index.faiss')

# Function to initiate chat with model selection
@cl.on_chat_start
async def start():
    msg = await cl.Message(content="I'm your Obsidian Assistant!").send()
    # Display model selection widget
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-4o-mini", "gpt-4o", "llama3"],
                initial_index=0,
            )
        ]
    ).send()

    # Initialize session memory
    cl.user_session.set("message_history", [])
    cl.user_session.set("Model", settings.get("Model", "gpt-4o-mini"))
    cl.user_session.set("previous_notes", [])  # Initialize previous notes list

    # Display the "Update notes" button
    await cl.Action(name="rerun_embeddings", label="Update notes", value="rerun_embeddings").send(for_id=msg.id)

# Function to handle user queries
@cl.on_message
async def main(message):

    # Retrieve selected model from user session
    selected_model = cl.user_session.get("Model")
    print(f"Selected model: {selected_model}")

    # Retrieve message history from session memory
    message_history = cl.user_session.get("message_history", [])

    user_input = message.content

    # Create embedding for the user's query
    query_embedding = get_embedding(user_input)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    # Search for the top 3 most similar notes
    k = 3
    distances, indices = index.search(query_embedding, k)

    # Filter results based on similarity threshold and compute similarity as a percentage
    similarity_threshold = 0  # Adjust as needed
    retrieved_notes = []
    retrieved_titles = []
    retrieved_similarities = []
    for i, distance in zip(indices[0], distances[0]):
        # Convert distance to similarity percentage
        similarity = (1 / (1 + distance)) * 100

        # Add note only if similarity exceeds the threshold
        if similarity >= similarity_threshold:
            retrieved_notes.append(notes[i])
            retrieved_titles.append(titles[i])
            retrieved_similarities.append(similarity)  # Save similarity value

    # Store retrieved notes and titles in the session
    cl.user_session.set("retrieved_notes", retrieved_notes)
    cl.user_session.set("retrieved_titles", retrieved_titles)
    cl.user_session.set("retrieved_similarities", retrieved_similarities)
    cl.user_session.set("user_input", user_input)

    # Display each note as a toggle button
    msg = await cl.Message(content="Select notes to use as context by clicking on each. When done, press 'Continue'.").send()

    for i, (title, similarity) in enumerate(zip(retrieved_titles, retrieved_similarities)):
        await cl.Action(
            name="toggle_note",
            label=f"{title} - Similarity: {similarity:.2f}%",
            value=str(i)
        ).send(for_id=msg.id)

    # Add a "Continue" button to finalize selection
    await cl.Action(
        name="finalize_selection",
        label="Continue",
        value="finalize"
    ).send(for_id=msg.id)

    # Initialize session storage for selected notes
    cl.user_session.set("selected_note_indices", [])

# Callback function when a note is selected
async def process_selected_notes(selected_indices):
    retrieved_notes = cl.user_session.get("retrieved_notes")
    retrieved_titles = cl.user_session.get("retrieved_titles")
    user_input = cl.user_session.get("user_input")
    previous_notes = cl.user_session.get("previous_notes")
    message_history = cl.user_session.get("message_history", [])

    # If no notes selected, use previous notes or just chat history
    if not selected_indices:
        if previous_notes:
            selected_notes = previous_notes
            print("Using previous notes as context.")
        else:
            selected_notes = []
            print("No previous notes found. Using chat history only.")
    else:
        selected_notes = [
            {'title': retrieved_titles[idx], 'content': retrieved_notes[idx]}
            for idx in selected_indices
        ]
        cl.user_session.set("previous_notes", selected_notes)


    # Prepare the context
    if selected_notes:
        notes_context = '\n\n'.join([f"Title: {note['title']}\nContent: {note['content']}" for note in selected_notes])
        # Create the prompt for the GPT model
        prompt = f"""Use the following notes and the conversation history to answer the question. Refer to the titles of the notes in your answer.

Notes:
{notes_context}

Tasks:

{tasks_content}

Conversation History:
{format_message_history(message_history)}

Question: {user_input}

Answer:"""
    else:
        # No notes selected, use only chat history
        prompt = f"""Use the conversation history to answer the question.

Tasks:

{tasks_content}

Conversation History:
{format_message_history(message_history)}

Question: {user_input}

Answer:"""

    # Append user's question to message history
    message_history.append({"role": "user", "content": user_input})
    cl.user_session.set("message_history", message_history)

    msg = cl.Message(content="")
    await msg.send()

    # Retrieve the selected model
    selected_model = cl.user_session.get("Model")
    print(f"Selected model: {selected_model}")

    # Generate the answer
    model_header = f"Selected model: {selected_model}\n\n---\n\n"

    if selected_model in ["gpt-4o-mini", "gpt-4o"]:
        # Get the response from the GPT model
        response = openai.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that helps answer questions based on the user's notes and conversation history."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            stream=True
        )

        # Process the streaming response
        collected_answer = model_header
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                collected_answer += chunk_content
                msg.content = collected_answer
                await msg.update()

    elif selected_model == "llama3":
        ollama_url = "http://localhost:11434/api/generate"  # Update with your Ollama server URL and port
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": "llama3",  # The model name as per your Ollama configuration
            "prompt": prompt
        }

        # Send the request to Ollama
        response = requests.post(ollama_url, headers=headers, json=data, stream=True)

        # Check for errors
        if response.status_code != 200:
            await cl.Message(content=f"Error: {response.status_code} - {response.text}").send()
            return

        # Process the streaming response
        collected_answer = model_header
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    line_data = json.loads(line)
                    chunk_content = line_data.get('response', '')
                    if chunk_content:
                        collected_answer += chunk_content
                        msg.content = collected_answer
                        await msg.update()
                except json.JSONDecodeError:
                    continue

    else:
        await cl.Message(content="Selected model is not supported.").send()
        return

    # Append assistant's response to message history
    message_history.append({"role": "assistant", "content": collected_answer})
    cl.user_session.set("message_history", message_history)

    # Display the action to open the note in Obsidian, if a note was used
    if selected_notes:
        for note in selected_notes:
            title = note['title']
            link = f"obsidian://open?vault={quote(vault_name)}&file={quote(title)}"
            await cl.Action(name="answer.satisfy", label=f"Open note: {title}", value=link).send(for_id=msg.id)

# Action callback to open the note in Obsidian
@cl.action_callback("answer.satisfy")
async def action_callback(action):
    link = action.value
    subprocess.run(["open", link]) 

# Action callback to update notes
@cl.action_callback("rerun_embeddings")
async def rerun_embeddings_callback(action):
    await cl.Message(content="Updating your notes...").send()

    # Reload notes and regenerate embeddings
    notes, titles = load_notes(vault_path)
    embeddings = [get_embedding(note) for note in notes]
    embedding_matrix = np.array(embeddings).astype('float32')
    
    # Rebuild FAISS index
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    # Save new embeddings, titles, and index
    np.save('embeddings.npy', embedding_matrix)
    np.save('titles.npy', np.array(titles))
    np.save('notes.npy', np.array(notes))
    faiss.write_index(index, 'notes_index.faiss')

    await cl.Message(content="All notes are up to date!").send()

# Handle settings update
@cl.on_settings_update
async def on_settings_update(settings):
    print("Settings updated ", settings)
    selected_model = settings["Model"]
    cl.user_session.set("Model", selected_model)
    print(f"Selected model: {selected_model}")

# Function to format message history
def format_message_history(message_history):
    formatted_history = ''
    for message in message_history:
        role = message.get('role', '')
        content = message.get('content', '')
        formatted_history += f"{role.capitalize()}: {content}\n"
    return formatted_history

@cl.action_callback("toggle_note")
async def toggle_note_callback(action):
    selected_note_indices = cl.user_session.get("selected_note_indices", [])
    note_index = int(action.value)

    # Retrieve the list of titles from the session
    retrieved_titles = cl.user_session.get("retrieved_titles")

    # Toggle selection: add if not in list, remove if already selected
    if note_index in selected_note_indices:
        selected_note_indices.remove(note_index)
    else:
        selected_note_indices.append(note_index)
    
    # Update session memory
    cl.user_session.set("selected_note_indices", selected_note_indices)

    # Get the titles of the selected notes
    selected_titles = [retrieved_titles[i] for i in selected_note_indices]
    
    # Display a message with the selected titles
    await cl.Message(content=f"Selected notes updated: {', '.join(selected_titles)}").send()

@cl.action_callback("finalize_selection")
async def finalize_selection_callback(action):
    selected_indices = cl.user_session.get("selected_note_indices")
    await process_selected_notes(selected_indices)