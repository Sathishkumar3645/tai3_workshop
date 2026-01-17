from jinja2 import Environment, FileSystemLoader
import os

PROMPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')

def render_chat_prompt(user_query, formatted_history):
    env = Environment(loader=FileSystemLoader(PROMPT_DIR))
    template = env.get_template('chat_prompt.j2')
    return template.render(user_query=user_query, formatted_history=formatted_history)