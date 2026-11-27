# Effect-lane: 30-prompt greedy-generation probe set (authored, fixed).
# checkers: substring/regex evaluated case-insensitively on the generated text.
PROMPTS = [
    # --- arithmetic (checkable) ---
    {"id": "arith_01", "kind": "arithmetic", "prompt": "What is 17 + 26?", "check": ["43"]},
    {"id": "arith_02", "kind": "arithmetic", "prompt": "What is 144 divided by 12?", "check": ["12"]},
    {"id": "arith_03", "kind": "arithmetic", "prompt": "What is 9 times 7?", "check": ["63"]},
    {"id": "arith_04", "kind": "arithmetic", "prompt": "What is 100 minus 37?", "check": ["63"]},
    {
        "id": "arith_05",
        "kind": "arithmetic",
        "prompt": "I have 3 boxes with 8 apples each. How many apples do I have in total?",
        "check": ["24"],
    },
    {"id": "arith_06", "kind": "arithmetic", "prompt": "What is 25% of 80?", "check": ["20"]},
    {"id": "arith_07", "kind": "arithmetic", "prompt": "What is 2 raised to the power of 10?", "check": ["1024"]},
    {"id": "arith_08", "kind": "arithmetic", "prompt": "Solve for x: 3x + 5 = 20.", "check": ["5"]},
    {"id": "arith_09", "kind": "arithmetic", "prompt": "What is the square root of 81?", "check": ["9"]},
    {"id": "arith_10", "kind": "arithmetic", "prompt": "What is 15 + 27 - 9?", "check": ["33"]},
    # --- common knowledge (checkable) ---
    {"id": "know_01", "kind": "knowledge", "prompt": "What is the capital of France?", "check": ["paris"]},
    {
        "id": "know_02",
        "kind": "knowledge",
        "prompt": "How many continents are there on Earth?",
        "check": ["seven", "7"],
    },
    {"id": "know_03", "kind": "knowledge", "prompt": "Which planet is known as the Red Planet?", "check": ["mars"]},
    {"id": "know_04", "kind": "knowledge", "prompt": "Who wrote the play Romeo and Juliet?", "check": ["shakespeare"]},
    {
        "id": "know_05",
        "kind": "knowledge",
        "prompt": "What is the chemical symbol for water?",
        "check": ["h2o", "h₂o"],
    },
    {"id": "know_06", "kind": "knowledge", "prompt": "How many days are there in a leap year?", "check": ["366"]},
    {
        "id": "know_07",
        "kind": "knowledge",
        "prompt": "What gas do plants absorb from the air for photosynthesis?",
        "check": ["carbon dioxide", "co2", "co₂"],
    },
    {"id": "know_08", "kind": "knowledge", "prompt": "What is the largest ocean on Earth?", "check": ["pacific"]},
    {"id": "know_09", "kind": "knowledge", "prompt": "In which direction does the sun rise?", "check": ["east"]},
    {
        "id": "know_10",
        "kind": "knowledge",
        "prompt": "What is the freezing point of water in degrees Celsius?",
        "check": ["0"],
    },
    # --- instruction following (open-ended; agreement signal, not correctness) ---
    {"id": "inst_01", "kind": "instruction", "prompt": "Name the three primary colors.", "check": None},
    {"id": "inst_02", "kind": "instruction", "prompt": "Write one short sentence about autumn.", "check": None},
    {"id": "inst_03", "kind": "instruction", "prompt": "Give a synonym for the word 'happy'.", "check": None},
    {"id": "inst_04", "kind": "instruction", "prompt": "List two common uses of salt.", "check": None},
    {"id": "inst_05", "kind": "instruction", "prompt": "Translate 'good morning' into French.", "check": ["bonjour"]},
    {
        "id": "inst_06",
        "kind": "instruction",
        "prompt": "Say the alphabet backwards from D to A.",
        "check": ["d c b a", "dcba"],
    },
    {
        "id": "inst_07",
        "kind": "instruction",
        "prompt": "What number comes next in the sequence 2, 4, 6, 8, ...?",
        "check": ["10"],
    },
    {
        "id": "inst_08",
        "kind": "instruction",
        "prompt": "Rewrite this sentence in the past tense: 'She walks to school.'",
        "check": ["walked"],
    },
    {
        "id": "inst_09",
        "kind": "instruction",
        "prompt": "Give one example of a mammal that lives in the ocean.",
        "check": None,
    },
    {"id": "inst_10", "kind": "instruction", "prompt": "Count from one to five.", "check": ["five"]},
]
