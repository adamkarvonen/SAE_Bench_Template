from typing import Callable

from transformers import PreTrainedTokenizerFast

LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTERS_UPPER = LETTERS.upper()
ALL_ALPHA_LETTERS = LETTERS + LETTERS_UPPER


def get_tokens(
    tokenizer: PreTrainedTokenizerFast,
    filter: Callable[[str], bool] = lambda _token: True,
    replace_special_chars: bool = True,
) -> list[str]:
    result = []
    for token in tokenizer.vocab.keys():
        word = tokenizer.convert_tokens_to_string([token])
        if filter(word):
            result.append(word if replace_special_chars else token)
    return result


def get_alpha_tokens(
    tokenizer: PreTrainedTokenizerFast,
    allow_leading_space: bool = True,
    replace_special_chars: bool = True,
) -> list[str]:
    def filter_alpha(token: str) -> bool:
        if allow_leading_space and token.startswith(" "):
            token = token[1:]
        if len(token) == 0:
            return False
        return all(char in ALL_ALPHA_LETTERS for char in token)

    return get_tokens(
        tokenizer, filter_alpha, replace_special_chars=replace_special_chars
    )
