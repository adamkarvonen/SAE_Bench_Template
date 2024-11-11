from dataclasses import dataclass


@dataclass
class AutoInterpConfig:
    """
    Controls all parameters for how autointerp will work.

    Arguments:
        model_name:                     The name of the model to use
        device:                         The device to use
        n_latents:                      The number of latents to use
        override_latents:               The latents to use (overrides n_latents if supplied)
        dead_latent_threshold:          The log sparsity value below which we consider a latent to be dead
        seed:                           The seed to use for all randomness

        buffer:                         The size of the buffer to use for scoring
        no_overlap:                     Whether to allow overlapping sequences for scoring
        act_threshold_frac:             The fraction of the maximum activation to use as the activation threshold
        total_tokens:                   The total number of tokens we'll gather data for.
        batch_size:                     The batch size to use for the scoring phase
        scoring:                        Whether to perform the scoring phase, or just return explanation
        max_tokens_in_explanation:      The maximum number of tokens to allow in an explanation
        use_demos_in_explanation:       Whether to use demonstrations in the explanation prompt

        n_top_ex_for_generation:        The number of top activating sequences to use for the generation phase
        n_iw_sampled_ex_for_generation: The number of importance-sampled sequences to use for the generation phase (this
                                        is a replacement for quantile sampling)

        n_top_ex_for_scoring:           The number of top sequences to use for scoring
        n_random_ex_for_scoring:        The number of random sequences to use for scoring
        n_iw_sampled_ex_for_scoring:    The number of importance-sampled sequences to use for scoring
    """

    # High-level params (not specific to autointerp)
    model_name: str
    n_latents: int | None = None
    override_latents: list[int] | None = None
    dead_latent_threshold: float = -8
    seed: int = 0

    # Main autointerp params
    buffer: int = 10
    no_overlap: bool = True
    act_threshold_frac: float = 0.01
    total_tokens: int = 2_000_000
    batch_size: int = 512  # split up total tokens into batches of this size
    scoring: bool = True
    max_tokens_in_explanation: int = 30
    use_demos_in_explanation: bool = True

    # Sequences included in generation phase
    n_top_ex_for_generation: int = 10
    n_iw_sampled_ex_for_generation: int = 5

    # Sequences included in scoring phase
    n_top_ex_for_scoring: int = 2
    n_random_ex_for_scoring: int = 10
    n_iw_sampled_ex_for_scoring: int = 2

    def __post_init__(self):
        if self.n_latents is None:
            assert self.override_latents is not None
            self.latents = self.override_latents
            self.n_latents = len(self.latents)
        else:
            assert self.override_latents is None
            self.latents = None

    @property
    def n_top_ex(self):
        """When fetching data, we get the top examples for generation & scoring simultaneously."""
        return self.n_top_ex_for_generation + self.n_top_ex_for_scoring

    @property
    def max_tokens_in_prediction(self) -> int:
        """Predictions take the form of comma-separated numbers, which should all be single tokens."""
        return 2 * self.n_ex_for_scoring + 5

    @property
    def n_ex_for_generation(self) -> int:
        return self.n_top_ex_for_generation + self.n_iw_sampled_ex_for_generation

    @property
    def n_ex_for_scoring(self) -> int:
        """For scoring phase, we use a randomly shuffled mix of top-k activations and random sequences."""
        return (
            self.n_top_ex_for_scoring
            + self.n_random_ex_for_scoring
            + self.n_iw_sampled_ex_for_scoring
        )

    @property
    def n_iw_sampled_ex(self) -> int:
        return self.n_iw_sampled_ex_for_generation + self.n_iw_sampled_ex_for_scoring

    @property
    def n_correct_for_scoring(self) -> int:
        return self.n_top_ex_for_scoring + self.n_iw_sampled_ex_for_scoring
