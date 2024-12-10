import argparse
import os
import platform
import shutil
from copy import deepcopy

import torch
from lavis.models import load_model_and_preprocess
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
from functools import partial

import wikipediaapi
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import re


class ImprovedWikipediaRetriever:
    def __init__(self, language='en', user_agent="RAGApp/1.0"):
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=user_agent
        )

        self.nlp = spacy.load("en_core_web_sm")

        # Expand landmark mappings
        self.landmark_mappings = {
            'eiffel tower': 'Eiffel Tower',
            'great wall of china': 'Great Wall of China',
            'taj mahal': 'Taj Mahal',
            'statue of liberty': 'Statue of Liberty',
            # Add more common landmark names and their canonical forms
        }

    def extract_most_important_entity(self, text):
        """
        Enhanced entity extraction with multi-stage approach
        """
        # Preprocess text
        text = text.lower()

        # Check for exact landmark matches first
        for landmark, canonical_name in self.landmark_mappings.items():
            if landmark in text:
                return canonical_name

        # Process text with spaCy
        doc = self.nlp(text)

        # Priority order of entity extraction
        entity_priority = [
            'PRODUCT',  # Landmarks, artifacts
            'ORG',  # Organizations
            'PERSON',  # People
            'GPE',  # Geo-political entities
            'LOC',  # Locations
        ]

        # First, try to find entities in priority order
        for label in entity_priority:
            entities = [ent.text for ent in doc.ents if ent.label_ == label]
            if entities:
                return entities[0]

        # Fallback to most important nouns
        nouns = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
        if nouns:
            return nouns[0]

        # Last resort: return first meaningful word
        meaningful_words = [word for word in text.split() if len(word) > 2]
        return meaningful_words[0] if meaningful_words else text

    def search_wikipedia(self, query, context=None, top_paragraphs=2):
        """
        Advanced Wikipedia search with improved entity matching
        """
        # Combine query and context
        full_text = f"{query} {context}" if context else query

        # Extract the most important entity
        primary_entity = self.extract_most_important_entity(full_text)

        # Try variations of the entity to maximize search chances
        search_variations = [
            primary_entity,
            primary_entity.lower(),
            primary_entity.title(),
            re.sub(r'\s+', '_', primary_entity)
        ]

        # Expanded search strategy
        search_variations += [
            f"{primary_entity} landmark",
            f"{primary_entity} history",
            primary_entity.replace(" ", "_")
        ]

        # Search through variations
        for variation in search_variations:
            try:
                page = self.wiki_wiki.page(variation)
                if page.exists():
                    # Prefer longer summaries
                    paragraphs = page.summary.split('\n')
                    truncated_summary = '\n'.join(paragraphs[:top_paragraphs])

                    return {
                        'title': page.title,
                        'summary': truncated_summary,
                        'full_summary': page.summary,
                        'url': page.fullurl,
                        'entity': variation
                    }
            except Exception as e:
                print(f"Search error for {variation}: {e}")

        return {
            'title': 'Not Found',
            'summary': 'No relevant Wikipedia information found.',
            'full_summary': '',
            'url': None,
            'entity': primary_entity
        }


class RAGPipeline:
    def __init__(self, wiki_retriever=None):
        self.wiki_retriever = wiki_retriever or ImprovedWikipediaRetriever()

    def enhance_query(self, original_query, image_description=None):
        """
        Enhance the query with Wikipedia context
        """
        # Combine image description and query for context
        combined_query = f"{image_description or ''} {original_query}"

        # Retrieve Wikipedia context
        wiki_result = self.wiki_retriever.search_wikipedia(combined_query)

        # Construct an enhanced query with Wikipedia context
        enhanced_query = (
            f"Context from Wikipedia about '{wiki_result['entity']}': {wiki_result['summary']}\n"
            f"Original Query: {original_query}\n"
            "Provide a concise and informative response considering the context:"
        )

        return enhanced_query, wiki_result


def _load_model_processor(args):
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"

    global load_model_and_preprocess
    load_model_and_preprocess = partial(load_model_and_preprocess, is_eval=True, device=device_map)

    model, vis_processors, _ = load_model_and_preprocess("minigpt4qwen", args.model_type,
                                                         llm_device_map=args.llm_device_map)
    model.load_checkpoint(args.checkpoint_path)

    model.llm_model.transformer.bfloat16()
    model.llm_model.lm_head.bfloat16()

    generation_config = {
        "chat_format": "chatml",
        "eos_token_id": 151643,
        "pad_token_id": 151643,
        "max_window_size": 6144,
        "max_new_tokens": 512,
        "do_sample": False,
        "transformers_version": "4.31.0"
    }

    generation_config = GenerationConfig.from_dict(generation_config)

    return model, vis_processors, generation_config


_WELCOME_MSG = '''
Welcome to RAG-Enhanced Interactive Chat! 
Type text to start chat, type :h to show command help.
'''

_HELP_MSG = '''
Commands:
    :help / :h          Show this help message
    :exit / :quit / :q  Exit the demo
    :clear / :cl        Clear screen
    :clear-his / :clh   Clear history
    :history / :his     Show history
'''


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f'History ({len(history)})'.center(terminal_width, '='))
    for index, (query, response, rag_info) in enumerate(history):
        print(f'User[{index}]: {query}')
        print(f'MPP-Qwen[{index}]: {response}')
        print(f'Wiki Context[{index}]: {rag_info["wiki_context"]["title"]}')
        if rag_info["wiki_context"]["url"]:
            print(f'Wiki URL[{index}]: {rag_info["wiki_context"]["url"]}')
    print('=' * terminal_width)


def _get_input() -> str:
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty')


def _get_image_input():
    images, messages = [], []
    while True:
        try:
            message = input(
                'Please input the path of images (you can input multiple image paths, use `:f` to finish input):> ')
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            if message == ":f":
                print("[Finished] You've finished inputting images!")
                return images, messages
            try:
                from PIL import Image
                image = Image.open(message).convert("RGB")
            except Exception as e:
                print(e)
                continue
            images.append(image)
            messages.append(message)
        else:
            print('[ERROR] Query is empty')


def main():
    parser = argparse.ArgumentParser(
        description='RAG-Enhanced Interactive Chat Demo.')
    parser.add_argument("--model-type", type=str, default='qwen7b_chat',
                        choices=['qwen7b_chat', 'qwen14b_chat'])
    parser.add_argument("-c", "--checkpoint-path", type=str,
                        help="Checkpoint name or path")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Run demo with CPU only")
    parser.add_argument("--llm_device_map", type=str, default="cpu")
    args = parser.parse_args()

    # Initialize model and RAG pipeline
    model, vis_processors, generation_config = _load_model_processor(args)
    rag_pipeline = RAGPipeline()

    # Image input
    images, image_paths = _get_image_input()
    image_tensor = torch.stack([vis_processors['eval'](image) for image in images], dim=0).to(model.device)

    # Initialize history and seed
    history = []
    seed = args.seed
    set_seed(seed)

    _clear_screen()
    print(_WELCOME_MSG)

    first = True
    while True:
        query = _get_input()

        # Process commands (similar to original implementation)
        if query.startswith(':'):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ''
            else:
                command = command_words[0]

            if command in ['exit', 'quit', 'q']:
                break
            elif command in ['clear', 'cl']:
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()
                continue
            elif command in ['clear-history', 'clh']:
                print(f'[INFO] All {len(history)} history cleared')
                history.clear()
                _gc()
                continue
            elif command in ['help', 'h']:
                print(_HELP_MSG)
                continue
            elif command in ['history', 'his']:
                _print_history(history)
                continue
            else:
                # As normal query.
                pass

        # Prepare query (similar to original implementation)
        if first:
            if '<ImageHere>' not in query:
                img_query = ""
                for _ in image_paths:
                    img_query += '<Img><ImageHere></Img>'
                query = img_query + query
            first = False

        try:
            # Enhance query with RAG context
            enhanced_query, wiki_result = rag_pipeline.enhance_query(
                query,
                image_description=' '.join(image_paths)
            )

            # Use model.chat with the enhanced query
            if args.cpu_only:
                model.bfloat16()
                response, updated_history = model.chat(
                    enhanced_query,
                    history=history,
                    image_tensor=image_tensor.bfloat16(),
                    generation_config=generation_config
                )
            else:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    response, updated_history = model.chat(
                        enhanced_query,
                        history=history,
                        image_tensor=image_tensor,
                        generation_config=generation_config
                    )

            # Store RAG-enhanced interaction
            rag_info = {
                'wiki_context': wiki_result,
                'enhanced_query': enhanced_query
            }
            history = updated_history

            # Clear screen and display response
            _clear_screen()
            print(f"\n\033[33mUser:\033[0m {query}")
            print(f"\n\033[31mMPP-Qwen:\033[0m {response}")
            print(f"\n\033[34mWikipedia Context:\033[0m {wiki_result['summary']}")
            if wiki_result['url']:
                print(f"\n\033[32mMore Info:\033[0m {wiki_result['url']}")

        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue


if __name__ == "__main__":
    main()