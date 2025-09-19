from textattack.augmentation.augmenter import Augmenter
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
DEFAULT_CONSTRAINTS = [RepeatModification(), StopwordModification()]


class TextBuggerAugmenter(Augmenter):
    """Li, J., Ji, S., Du, T., Li, B., and Wang, T. (2018).
    TextBugger: Generating Adversarial Text Against Real-world Applications.

    https://arxiv.org/abs/1812.05271
    """

    def __init__(
            self, **kwargs
    ):
        from textattack.transformations import (
            CompositeTransformation,
            WordSwapEmbedding,
            WordSwapHomoglyphSwap,
            WordSwapNeighboringCharacterSwap,
            WordSwapRandomCharacterDeletion,
            WordSwapRandomCharacterInsertion,
        )

        transformation = CompositeTransformation(
            [
                # (1) Insert: Insert a space into the word.
                # Generally, words are segmented by spaces in English. Therefore,
                # we can deceive classifiers by inserting spaces into words.
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Delete a random character of the word except for the first
                # and the last character.
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swap random two adjacent letters in the word but do not
                # alter the first or last letter. This is a common occurrence when
                # typing quickly and is easy to implement.
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-C (Sub-C): Replace characters with visually similar
                # characters (e.g., replacing “o” with “0”, “l” with “1”, “a” with “@”)
                # or adjacent characters in the keyboard (e.g., replacing “m” with “n”).
                WordSwapHomoglyphSwap(),
                # (5) Substitute-W
                # (Sub-W): Replace a word with its topk nearest neighbors in a
                # context-aware word vector space. Specifically, we use the pre-trained
                # GloVe model [30] provided by Stanford for word embedding and set
                # topk = 5 in the experiment.
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        use_constraint = UniversalSentenceEncoder(threshold=0.8)

        constraints = DEFAULT_CONSTRAINTS + [use_constraint]

        super().__init__(transformation, constraints=constraints, **kwargs)


if __name__ == "__main__":
    # Instantiate the augmenter
    augmenter = TextBuggerAugmenter()

    # Example input text
    # text = "The quick brown fox jumps over the lazy dog."
    text = ("Manor Marussia, the renowned Formula 1 team, has made a resounding declaration as they confirm their "
            "eagerly awaited comeback for the upcoming season. After a brief absence from the prestigious motorsport, "
            "the team is determined to reclaim their position on the starting grid. With an unwavering commitment to "
            "the sport, they have been working tirelessly to ensure their cars meet the stringent regulations and "
            "standards set by Formula 1. Fans and enthusiasts alike await the return of Manor Marussia with bated "
            "breath, eager to witness their dedication and competitiveness once again.")

    # Generate augmented versions
    augmented_texts = augmenter.augment(text)

    # Print the results
    print("Original text:", text)
    print("\nAugmented texts:")
    for i, aug in enumerate(augmented_texts, start=1):
        print(f"{i}. {aug}")