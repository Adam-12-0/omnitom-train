try:
    from .benchmark_prompting import BENCHMARK_PATH, story_text
except ImportError:
    from benchmark_prompting import BENCHMARK_PATH, story_text


PROMPT = """
You are a Theory of Mind expert whose task is to extract multi-order actor beliefs from the narrative and output a table with columns Actor, Belief, and Order by performing the following steps. A belief is a minimal proposition expressing what an actor takes to be true.
1. Identify narrated events and states that the story presents as facts, and record them as world-level beliefs attributed to the special actor 'world' (order 0).
2. Identify all actors, including characters or groups, who appear in the narrative and are capable of holding beliefs.
3. For each actor, extract beliefs about the narrated events or states of the world, and record them as first-order beliefs (order 1).
4. For each actor, extract beliefs about other actors’ beliefs, applying this notion recursively for nested beliefs, and record them as higher-order beliefs (order 2 or higher).
""".strip()


OUTRO = """
Now output only the table.
""".strip()


def build_extract_messages(story_id: int, dataset_path=None) -> tuple[str, str]:
    story = story_text(story_id, dataset_path)

    system_prompt = PROMPT
    user_prompt = (
        "Given the story narrative below, extract multi-order actor beliefs and output only the pipe-separated table with header: Actor | Belief | Order.\n\n"
        + "Story Narrative:\n"
        + story
        + "\n\n"
        + OUTRO
    )
    return system_prompt, user_prompt
