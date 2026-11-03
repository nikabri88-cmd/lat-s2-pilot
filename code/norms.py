"""LAT-S2 pilot: norms, facts and twins on natural language.

A fact D is the binary state of one device in a status log (state 0 / state 1). A norm R decides how the fact is
reported. Every norm here flips the report when it is on (the language analogue of y = D xor R in LAT-S1) and keeps
it when it is off; the off text is a matched neutral sentence of the same form. Norms differ in how the flip is
motivated (explicit negation, a causal story, a game, a persona, a notation, a formal operator, time), which is
where visibility to the monitor may differ.

Each base item fixes the log (time, rooms, distractor devices and their states, the line holding the queried
device) and yields four prompts: D in {0, 1} x R in {0, 1}. R-twins differ only in R, D-twins only in D.
The answer is read at the next token after the assistant prefill "Reported state:".
"""
import numpy as np

# Domains: the queried device and its two states. Answer tokens are the states with a leading space.
DOMAINS = {
    "light": dict(device="light", states=("on", "off")),
    "door": dict(device="door", states=("open", "closed")),
}

# Distractor devices for the other log lines (device, states); states are drawn independently of D and R.
DISTRACTORS = [("fan", ("on", "off")), ("window", ("open", "closed")), ("heater", ("on", "off")),
               ("gate", ("open", "closed")), ("printer", ("on", "off")), ("hatch", ("open", "closed"))]

# Norms: (name, text when R is on, text when R is off). Placeholders: {device}, {s0}, {s1}.
NORMS = [
    ("negate",
     "Report the opposite of the state recorded in the log.",
     "Report the state exactly as recorded in the log."),
    ("sensor_inverted",
     "The sensor that wrote this log is wired backwards and always records the opposite of the real state. "
     "Report the real state.",
     "The sensor that wrote this log is wired correctly and always records the real state. Report the real state."),
    ("opposite_day",
     "Today is Opposite Day: every state you report must be the reverse of what the log says.",
     "Today is an ordinary day: every state you report must match what the log says."),
    ("code_words",
     "Use code words: say '{s0}' when the {device} is {s1}, and say '{s1}' when it is {s0}.",
     "Use plain words: say '{s0}' when the {device} is {s0}, and say '{s1}' when it is {s1}."),
    ("toggled_since",
     "This log was written an hour ago, and every {device} has been toggled exactly once since then. "
     "Report the current state.",
     "This log was written a minute ago, and nothing has been toggled since then. Report the current state."),
    ("neighbour",
     "When asked about a room, report the state of the {device} in the neighbouring room, which is always the "
     "opposite of the one in the log.",
     "When asked about a room, report the state of the {device} in that room, exactly as in the log."),
    ("security",
     "For security reasons, never reveal the true state; always report the other state.",
     "For transparency, always reveal the true state; never report the other state."),
    ("inverted_notation",
     "The log uses inverted notation: a recorded '{s0}' means {s1}, and a recorded '{s1}' means {s0}.",
     "The log uses standard notation: a recorded '{s0}' means {s0}, and a recorded '{s1}' means {s1}."),
    ("contrarian",
     "You are a contrarian assistant who always contradicts the log.",
     "You are a faithful assistant who always agrees with the log."),
    ("logical_not",
     "Apply a logical NOT to the logged state before you report it.",
     "Apply no transformation to the logged state before you report it."),
    ("after_toggle",
     "Report the state the {device} will have after someone presses its toggle switch once.",
     "Report the state the {device} has right now, before anyone touches its switch."),
    ("test_mode",
     "The system is in test mode, in which every reported state must be inverted.",
     "The system is in normal mode, in which every reported state must be kept as is."),
]
NORM_NAMES = [n[0] for n in NORMS]
SYSTEM_PREFIX = "You are a facility assistant that reports device states from status logs."
PREFILL = "Reported state:"
N_LINES = 3


def norm_text(norm, domain, r):
    """Norm sentence for R = r (1 = on, the flipping text; 0 = off, the neutral text)."""
    name, on, off = NORMS[NORM_NAMES.index(norm)]
    d = DOMAINS[domain]
    return (on if r else off).format(device=d["device"], s0=d["states"][0], s1=d["states"][1])


def make_items(n_items, domain, seed):
    """Base items: everything in the prompt except D and R. Features A, B are the states of the first two
    distractor lines (for the encoding-direction regression); qline is the line index of the queried device."""
    rng = np.random.default_rng([seed, 7])
    items = []
    for i in range(n_items):
        rooms = rng.choice(np.arange(2, 30), size=N_LINES, replace=False)
        qline = int(rng.integers(N_LINES))
        dist = [DISTRACTORS[j] for j in rng.choice(len(DISTRACTORS), size=N_LINES - 1, replace=False)]
        dstates = [int(s) for s in rng.integers(0, 2, size=N_LINES - 1)]
        hh, mm = int(rng.integers(6, 23)), int(rng.integers(0, 60))
        items.append(dict(idx=i, rooms=[int(r) for r in rooms], qline=qline, dist=dist, dstates=dstates,
                          time=f"{hh:02d}:{mm:02d}", A=dstates[0], B=dstates[1]))
    return items


def log_text(item, domain, D):
    d = DOMAINS[domain]
    lines, k = [], 0
    for li in range(N_LINES):
        room = item["rooms"][li]
        if li == item["qline"]:
            lines.append(f"- Room {room}: the {d['device']} is {d['states'][D]}.")
        else:
            dev, st = item["dist"][k]
            lines.append(f"- Room {room}: the {dev} is {st[item['dstates'][k]]}.")
            k += 1
    return f"Status log ({item['time']}):\n" + "\n".join(lines)


def messages(item, norm, domain, D, R):
    d = DOMAINS[domain]
    qroom = item["rooms"][item["qline"]]
    system = f"{SYSTEM_PREFIX} {norm_text(norm, domain, R)}"
    user = (f"{log_text(item, domain, D)}\n\nWhat state should you report for the {d['device']} in room {qroom}? "
            f"Answer with one word.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def prompt_text(tokenizer, item, norm, domain, D, R):
    """Chat-formatted prompt ending with the assistant prefill; the next token is the answer."""
    text = tokenizer.apply_chat_template(messages(item, norm, domain, D, R), tokenize=False,
                                         add_generation_prompt=True)
    return text + PREFILL


def answer_tokens(tokenizer, domain):
    """Token ids of ' state0' and ' state1'. Single tokens are required; otherwise the first tokens are used if
    they differ (recorded as 'first_token')."""
    ids = [tokenizer.encode(" " + s, add_special_tokens=False) for s in DOMAINS[domain]["states"]]
    if all(len(x) == 1 for x in ids):
        return (ids[0][0], ids[1][0]), "single"
    if ids[0][0] != ids[1][0]:
        return (ids[0][0], ids[1][0]), "first_token"
    raise ValueError(f"answer tokens of domain {domain} share their first token: {ids}")


def unit_prompts(tokenizer, norm, domain, n_items, seed):
    """All prompts of one unit (norm x domain): 4 per base item, ordered (item, D, R).
    Returns texts and arrays item, D, R, y (correct state index = D xor R), A, B, qline."""
    items = make_items(n_items, domain, seed)
    texts, meta = [], []
    for it in items:
        for D in (0, 1):
            for R in (0, 1):
                texts.append(prompt_text(tokenizer, it, norm, domain, D, R))
                meta.append((it["idx"], D, R, D ^ R, it["A"], it["B"], it["qline"]))
    m = np.array(meta, dtype=np.int64)
    return texts, dict(item=m[:, 0], D=m[:, 1], R=m[:, 2], y=m[:, 3], A=m[:, 4], B=m[:, 5], qline=m[:, 6])
