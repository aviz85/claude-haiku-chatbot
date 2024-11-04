import streamlit as st
import anthropic
import time
import re

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Set page config
st.set_page_config(page_title="Claude Chat", page_icon="🤖")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0

st.title("Claude 3.5 Haiku Chat")

# Add empty element in sidebar that will be updated
cost_placeholder = st.sidebar.empty()
cost_placeholder.markdown(f"💰 Total Cost: ${st.session_state.total_cost:.4f}")

def format_paragraph_direction(text):
    paragraphs = text.split('\n')
    formatted_paragraphs = []
    
    for p in paragraphs:
        if not p.strip():  # Handle empty lines
            formatted_paragraphs.append(p)
            continue
            
        # Find first letter in paragraph
        letter_match = re.search(r'[א-תA-Za-z]', p)
        
        if not letter_match:  # No letters found
            formatted_paragraphs.append(f'<div style="text-align: left; direction: ltr;">{p}</div>')
            continue
            
        first_letter = letter_match.group()
        
        # Check if first letter is Hebrew
        if '\u0590' <= first_letter <= '\u05FF':
            formatted_paragraphs.append(f'<div style="text-align: right; direction: rtl;">{p}</div>')
        else:  # English or other
            formatted_paragraphs.append(f'<div style="text-align: left; direction: ltr;">{p}</div>')
            
    return '\n'.join(formatted_paragraphs)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(format_paragraph_direction(message["content"]), unsafe_allow_html=True)
        else:
            st.write(message["content"])
        if "metrics" in message:
            st.caption(f"⚡ {message['metrics']['tokens_per_second']:.1f} tokens/sec | "
                      f"💰 ${message['metrics']['cost']:.4f}")

# Chat input
if prompt := st.chat_input("איך אני יכול לעזור לך היום?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get Claude's response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        token_count = 0
        start_time = time.time()
        
        # Create streaming response
        stream = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4000,
            messages=[
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.messages
            ],
            stream=True
        )
        
        # Process the stream
        for chunk in stream:
            if chunk.type == 'content_block_delta':
                full_response += chunk.delta.text
                token_count += 1
                message_placeholder.markdown(
                    format_paragraph_direction(full_response + "▌"), 
                    unsafe_allow_html=True
                )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get usage statistics from the final message
        final_message = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=4000,
            messages=[
                {"role": m["role"], "content": m["content"]} 
                for m in st.session_state.messages
            ]
        )
        
        input_tokens = final_message.usage.input_tokens
        output_tokens = final_message.usage.output_tokens
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * 1.0  # $1 per million tokens
        output_cost = (output_tokens / 1_000_000) * 5.0  # $5 per million tokens
        total_cost = input_cost + output_cost
        
        # Update total cost
        st.session_state.total_cost += total_cost
        cost_placeholder.markdown(f"💰 Total Cost: ${st.session_state.total_cost:.4f}")
        
        # Calculate tokens per second
        tokens_per_second = output_tokens / duration if duration > 0 else 0
        
        # Store metrics
        metrics = {
            "tokens_per_second": tokens_per_second,
            "cost": total_cost,
            "duration": duration,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        
        # Update final response without cursor
        message_placeholder.markdown(
            format_paragraph_direction(full_response),
            unsafe_allow_html=True
        )
        st.caption(f"⚡ {tokens_per_second:.1f} tokens/sec | 💰 ${total_cost:.4f}")
            
    # Add assistant response to chat history with metrics
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "metrics": metrics
    })