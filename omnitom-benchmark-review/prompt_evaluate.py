try:
    from .benchmark_prompting import belief_table_csv, story_text
except ImportError:
    from benchmark_prompting import belief_table_csv, story_text


OUTRO = """
Now output only the 2 tables, explicitly labeled “Prediction Table” and “Ground Truth Table,” with an added MatchCount column, formatted as CSV (not markdown), and do not include any other text.
""".strip()


JUDGE_PROMPT = """
You are a Theory of Mind evaluation expert whose task is to semantically match rows between two belief tables (Prediction, Ground Truth) extracted from the same short Story Narrative and output only the two tables, explicitly labeled “Prediction Table” and “Ground Truth Table,” with an added MatchCount column indicating how many distinct semantically equivalent rows exist in the other table for the same Actor. In this context, a Belief is a minimal statement of what an actor takes to be true about the world (facts/events) or about other actors’ mental states, expressed in natural language. Perform the task by following these steps:
1) If a Story Narrative is provided, use it only to resolve ambiguity (pronouns, aliases, implicit entities) and paraphrase meaning; if no narrative is provided, ignore narrative context entirely. In all cases, do not add rows and do not introduce new beliefs that are not present in either table.
2) Treat Actors as distinct mental agents and normalize only cosmetic variants of the same Actor name (case/spacing/punctuation and clear shortenings); never merge different Ground Truth Actors.
3) Handle the special actor 'world' first: treat 'world' as the key for narrated facts and events, and align world-level beliefs conservatively, typically one-to-one, allowing only minor normalization differences.
4) Restrict candidate matches to the same Actor group after normalization; if the Actor does not match, the row cannot match regardless of belief similarity.
5) Default to one-to-one with bookkeeping: if (and only if) there exists a clear semantically equivalent belief for the same Actor, assign the row its single best match among currently-unmatched target rows; otherwise assign no match (MatchCount = 0). If multiple rows compete for the same target row, keep only the closest semantic match and force the others to choose different unmatched targets or become 0.
6) Allow one-to-many only for compound rows: if a row clearly contains multiple independent beliefs, you may align it to 2–3 different rows in the other table within the same Actor group, but only if each aligned target row captures a distinct part of the compound meaning.
7) Ensure symmetry: after completing matches for Prediction rows, also compute MatchCount for every Ground Truth row using the same alignment decisions.

A good output should satisfy the following:
- Only compare beliefs inside the same Actor group; Ground Truth Actors are unique and must not be merged—if Actor differs, it is NOT a match even if belief text is identical.
- The special actor 'world' represents narrated facts and events (not a character in the story); world-level beliefs should generally align one-to-one across tables with only minor normalization differences.
- Match only beliefs that are semantically equivalent; do not invent or force alignments and do not introduce beliefs that are not present in either table.
- Use conservative alignment: when multiple rows are 'close', prefer the best single match and leave other rows unmatched rather than double-counting.
- Prefer one-to-one matches; allow one-to-many (MatchCount 2–3) only for genuinely compound beliefs.
- Do not match across different Actors, even if text looks similar.
- Output exactly two tables labeled Prediction and Ground Truth (in that order), with CSV header Actor,Belief,MatchCount and no extra text.

A correctly formatted output should satisfy the following:
- Output exactly two tables labeled Prediction and Ground Truth (in that order).
- Each table must be comma-separated with header: Actor,Belief,MatchCount
- Preserve original row order and add exactly one MatchCount column.
- Do not include explanations or any text outside the tables.
""".strip()


EXAMPLE1_INPUT = """
INPUT:

Story Narrative:
Xiao Bei finds a cabinet in the office, the label on the cabinet is pineapple, Xiao Bei cannot see what is inside the cabinet, Xiao Bei opens the cabinet and finds a calculator, there is no pineapple in the cabinet, Xiao Bei closes the cabinet and puts it back in its place, Han Mei Mei enters the office and sees the cabinet.

Prediction Table:
Actor,Belief
World,There is a cabinet in the office with a "pineapple" label
World,The cabinet contains a calculator and no pineapple
World,Xiao Bei opened the cabinet and saw the calculator
World,Han Mei Mei saw the cabinet from the outside
Xiao Bei,The cabinet contains a pineapple
Xiao Bei,The cabinet contains a calculator and no pineapple
Xiao Bei,Han Mei Mei believes the cabinet contains a pineapple
Han Mei Mei,The cabinet contains a pineapple
Han Mei Mei,Xiao Bei believes the cabinet contains a pineapple

Ground Truth Table:
Actor,Belief
world,Xiao Bei is in the office
world,There is a cabinet in the office
world,The label on the cabinet is pineapple
world,The cabinet contains a calculator
world,The cabinet does not contain a pineapple
world,Xiao Bei opens the cabinet
world,Xiao Bei finds a calculator in the cabinet
world,Xiao Bei closes the cabinet
world,Han Mei Mei enters the office
world,Han Mei Mei sees the cabinet
Xiao Bei,There is a cabinet in the office
Xiao Bei,The label on the cabinet is pineapple
Xiao Bei,The cabinet should have pineapple inside
Xiao Bei,There is no pineapple in the cabinet
Xiao Bei,The cabinet contains a calculator
Xiao Bei,Han Mei Mei thinks the cabinet contains a pineapple
Xiao Bei,Han Mei Mei thinks Xiao Bei thinks the cabinet contains a pineapple
Han Mei Mei,There is a cabinet in the office
Han Mei Mei,The label on the cabinet is pineapple
Han Mei Mei,The cabinet contains a pineapple
Han Mei Mei,Xiao Bei thinks the cabinet contains a pineapple
"""

EXAMPLE1_OUTPUT = """
OUTPUT:

Prediction Table:
Actor,Belief,MatchCount
World,There is a cabinet in the office with a "pineapple" label,2
World,The cabinet contains a calculator and no pineapple,2
World,Xiao Bei opened the cabinet and saw the calculator,2
World,Han Mei Mei saw the cabinet from the outside,1
Xiao Bei,The cabinet contains a pineapple,0
Xiao Bei,The cabinet contains a calculator and no pineapple,2
Xiao Bei,Han Mei Mei believes the cabinet contains a pineapple,1
Han Mei Mei,The cabinet contains a pineapple,1
Han Mei Mei,Xiao Bei believes the cabinet contains a pineapple,1

Ground Truth Table:
Actor,Belief,MatchCount
world,Xiao Bei is in the office,0
world,There is a cabinet in the office,1
world,The label on the cabinet is pineapple,1
world,The cabinet contains a calculator,1
world,The cabinet does not contain a pineapple,1
world,Xiao Bei opens the cabinet,1
world,Xiao Bei finds a calculator in the cabinet,1
world,Xiao Bei closes the cabinet,0
world,Han Mei Mei enters the office,0
world,Han Mei Mei sees the cabinet,1
Xiao Bei,There is a cabinet in the office,0
Xiao Bei,The label on the cabinet is pineapple,0
Xiao Bei,The cabinet should have pineapple inside,0
Xiao Bei,There is no pineapple in the cabinet,1
Xiao Bei,The cabinet contains a calculator,1
Xiao Bei,Han Mei Mei thinks the cabinet contains a pineapple,1
Xiao Bei,Han Mei Mei thinks Xiao Bei thinks the cabinet contains a pineapple,0
Han Mei Mei,There is a cabinet in the office,0
Han Mei Mei,The label on the cabinet is pineapple,0
Han Mei Mei,The cabinet contains a pineapple,1
Han Mei Mei,Xiao Bei thinks the cabinet contains a pineapple,1
"""

EXAMPLE2_INPUT = """
INPUT:

Story Narrative:
The little sister is playing with toys, she sees her brother playing a game on his phone. The little sister looks at her brother and says, ""Brother, that game is so fun.""

Prediction Table:
Actor,Belief
world,The little sister is playing with toys
world,The brother is playing a game on his phone
world,The little sister says to the brother "Brother that game is so fun"
world,The little sister and the brother are siblings
Little sister,The brother's game is fun
Little sister,She wants to play the game on the phone
Little sister,The brother should let her play the game
Little sister,The brother understands she wants to play the game
Brother,The little sister thinks the game is fun
Brother,The little sister wants to play the game
Brother,The little sister implies that he should let her play the game
Brother,The little sister thinks he should share the phone

Ground Truth Table:
Actor,Belief
world,The little sister is playing with toys
world,The little sister sees the brother playing a game on the brother's phone
world,The little sister says to the brother "Brother that game is so fun"
The little sister,The brother is playing a game on the brother's phone
The little sister,The game on the phone is fun
The little sister,The brother wants to play the game
The little sister,The brother should let the sister play the game
The little sister,The brother thinks the sister wants to play the game
The brother,The little sister is playing with toys
The brother,The little sister thinks the game is fun
The brother,The little sister wants to play the game
The brother,The little sister implies the brother should let the little sister play the game
The brother,The little sister thinks the brother should let the little sister play the game
"""

EXAMPLE2_OUTPUT = """
OUTPUT: 

Prediction Table:
Actor,Belief,MatchCount
world,The little sister is playing with toys,1
world,The brother is playing a game on his phone,1
world,The little sister says to the brother "Brother that game is so fun",1
world,The little sister and the brother are siblings,0
Little sister,The brother's game is fun,1
Little sister,She wants to play the game on the phone,0
Little sister,The brother should let her play the game,1
Little sister,The brother understands she wants to play the game,1
Brother,The little sister thinks the game is fun,1
Brother,The little sister wants to play the game,1
Brother,The little sister implies that he should let her play the game,1
Brother,The little sister thinks he should share the phone,1

Ground Truth Table:
Actor,Belief,MatchCount
world,The little sister is playing with toys,1
world,The little sister sees the brother playing a game on the brother's phone,1
world,The little sister says to the brother "Brother that game is so fun",1
The little sister,The brother is playing a game on the brother's phone,0
The little sister,The game on the phone is fun,1
The little sister,The brother wants to play the game,0
The little sister,The brother should let the sister play the game,1
The little sister,The brother thinks the sister wants to play the game,1
The brother,The little sister is playing with toys,0
The brother,The little sister thinks the game is fun,1
The brother,The little sister wants to play the game,1
The brother,The little sister implies the brother should let the little sister play the game,1
The brother,The little sister thinks the brother should let the little sister play the game,1
"""

EXAMPLE3_INPUT = """
INPUT:

Story Narrative:
At a family gathering, everyone shares photos. Grandpa tells his grandson Xiao Ming, ""There are 60 photos here, most are travel photos, some are family photos, but there are hardly any wedding photos."" Xiao Ming flips through and finds only 15 family photos.

Prediction Table:
Actor,Belief
world,A family gathering is taking place
world,There are 60 photos in total
world,Grandpa tells Xiao Ming that most photos are travel some are family and hardly any are wedding
world,Xiao Ming finds 15 family photos after flipping through them
Grandpa,There are 60 photos in the collection
Grandpa,Most of the 60 photos (at least 31) are travel photos
Grandpa,Some of the 60 photos are family photos
Grandpa,Hardly any of the 60 photos are wedding photos
Grandpa,Xiao Ming knows there are 60 photos
Grandpa,Xiao Ming knows most of the photos are travel photos
Xiao Ming,There are 60 photos in the collection
Xiao Ming,Grandpa believes there are 60 photos
Xiao Ming,Before counting most of the photos are travel photos
Xiao Ming,Before counting some of the photos are family photos
Xiao Ming,Before counting hardly any of the photos are wedding photos
Xiao Ming,Grandpa believes most of the photos are travel photos
Xiao Ming,Grandpa believes some of the photos are family photos
Xiao Ming,Grandpa believes hardly any of the photos are wedding photos
Xiao Ming,After counting there are exactly 15 family photos
Xiao Ming,After counting "some" refers to exactly 15 photos in this context
Xiao Ming,Grandpa likely still believes there are "some" family photos

Ground Truth Table:
Actor,Belief
world,Everyone shares photos at a family gathering
world,Grandpa tells Xiao Ming there are 60 photos
world,Grandpa tells Xiao Ming most are travel photos
world,Grandpa tells Xiao Ming some are family photos
world,Grandpa tells Xiao Ming hardly any are wedding photos
world,Xiao Ming flips through the photos
world,Xiao Ming counts 15 family photos
Grandpa,There are 60 photos
Grandpa,Most photos are travel photos
Grandpa,Some photos are family photos
Grandpa,There are hardly any wedding photos
Xiao Ming,There are 60 photos
Xiao Ming,Before flipping through most photos are travel photos
Xiao Ming,Before flipping through some photos are family photos
Xiao Ming,Before flipping through there are hardly any wedding photos
Xiao Ming,Before flipping through there are probably 45 travel photos
Xiao Ming,Grandpa thinks there are 60 photos
Xiao Ming,Grandpa thinks most photos are travel photos
Xiao Ming,Grandpa thinks some photos are family photos
Xiao Ming,Grandpa thinks there are hardly any wedding photos
Xiao Ming,After flipping through there are 15 family photos
Xiao Ming,After flipping through there are approximately 40 travel photos
"""

EXAMPLE3_OUTPUT = """
OUTPUT:

Prediction Table:
Actor,Belief,MatchCount
world,A family gathering is taking place,1
world,There are 60 photos in total,1
world,Grandpa tells Xiao Ming that most photos are travel some are family and hardly any are wedding,3
world,Xiao Ming finds 15 family photos after flipping through them,2
Grandpa,There are 60 photos in the collection,1
Grandpa,Most of the 60 photos (at least 31) are travel photos,1
Grandpa,Some of the 60 photos are family photos,1
Grandpa,Hardly any of the 60 photos are wedding photos,1
Grandpa,Xiao Ming knows there are 60 photos,0
Grandpa,Xiao Ming knows most of the photos are travel photos,0
Xiao Ming,There are 60 photos in the collection,1
Xiao Ming,Grandpa believes there are 60 photos,1
Xiao Ming,Before counting most of the photos are travel photos,1
Xiao Ming,Before counting some of the photos are family photos,1
Xiao Ming,Before counting hardly any of the photos are wedding photos,1
Xiao Ming,Grandpa believes most of the photos are travel photos,1
Xiao Ming,Grandpa believes some of the photos are family photos,1
Xiao Ming,Grandpa believes hardly any of the photos are wedding photos,1
Xiao Ming,After counting there are exactly 15 family photos,1
Xiao Ming,After counting "some" refers to exactly 15 photos in this context,0
Xiao Ming,Grandpa likely still believes there are "some" family photos,0

Ground Truth Table:
Actor,Belief,MatchCount
world,Everyone shares photos at a family gathering,1
world,Grandpa tells Xiao Ming there are 60 photos,1
world,Grandpa tells Xiao Ming most are travel photos,1
world,Grandpa tells Xiao Ming some are family photos,1
world,Grandpa tells Xiao Ming hardly any are wedding photos,1
world,Xiao Ming flips through the photos,1
world,Xiao Ming counts 15 family photos,1
Grandpa,There are 60 photos,1
Grandpa,Most photos are travel photos,1
Grandpa,Some photos are family photos,1
Grandpa,There are hardly any wedding photos,1
Xiao Ming,There are 60 photos,1
Xiao Ming,Before flipping through most photos are travel photos,1
Xiao Ming,Before flipping through some photos are family photos,1
Xiao Ming,Before flipping through there are hardly any wedding photos,1
Xiao Ming,Before flipping through there are probably 45 travel photos,0
Xiao Ming,Grandpa thinks there are 60 photos,1
Xiao Ming,Grandpa thinks most photos are travel photos,1
Xiao Ming,Grandpa thinks some photos are family photos,1
Xiao Ming,Grandpa thinks there are hardly any wedding photos,1
Xiao Ming,After flipping through there are 15 family photos,1
Xiao Ming,After flipping through there are approximately 40 travel photos,0
"""


FEWSHOT_EXAMPLES = [
    (EXAMPLE1_INPUT, EXAMPLE1_OUTPUT),
    (EXAMPLE2_INPUT, EXAMPLE2_OUTPUT),
    (EXAMPLE3_INPUT, EXAMPLE3_OUTPUT),
]


def build_judge_prompt(fewshots: int = 3) -> str:
    if fewshots < 0 or fewshots > len(FEWSHOT_EXAMPLES):
        raise ValueError(f"fewshots must be between 0 and {len(FEWSHOT_EXAMPLES)}")

    if fewshots == 0:
        return JUDGE_PROMPT

    if fewshots == 1:
        suffix = (
            "Here is an example showing the matching process:\n\n"
            + FEWSHOT_EXAMPLES[0][0] + "\n" + FEWSHOT_EXAMPLES[0][1]
        )
        return JUDGE_PROMPT + "\n\n" + suffix

    parts = ["Here are examples showing the matching process:\n"]
    for i in range(fewshots):
        inp, out = FEWSHOT_EXAMPLES[i]
        parts.append(f"Example {i+1}:\n\n{inp}\n{out}")
    return JUDGE_PROMPT + "\n\n" + "\n\n".join(parts)


def build_evaluation_messages(
    story_id: int,
    predictions_csv: str,
    dataset_path=None,
    include_story: bool = True,
    fewshots: int = 3,
) -> tuple[str, str]:
    groundtruth_csv = belief_table_csv(story_id, dataset_path)
    story = story_text(story_id, dataset_path)
    system_prompt = build_judge_prompt(fewshots).strip()

    story_block = ""
    if include_story:
        story_block = "Story Narrative:\n" + story + "\n\n"

    user_prompt = (
        "Given the story narrative and the belief tables below (Prediction Table and Ground Truth Table), semantically match beliefs within the same Actor and output only the two CSV tables with MatchCount.\n\n"
        + story_block
        + "Prediction Table:\n"
        + predictions_csv
        + "\n\n"
        + "Ground Truth Table:\n"
        + groundtruth_csv
        + "\n\n"
        + OUTRO
    )
    return system_prompt, user_prompt
