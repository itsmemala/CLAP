"""
Config variables for generating prompt responses
"""

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 6
COT_FLAG = True
DEBUG = False
ANSWER_TRIGGER = "So the answer is"
SHORT_ANSWER_TRIGGER = "answer is" # for long answer