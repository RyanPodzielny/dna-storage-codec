"""
Module simulates the errors a DNA strand picks up as it moves through the storage channel. There are four stages, each
with its own config: synthesis (writing the strands), aging (storing them), PCR (amplifying them), and sequencing
(reading them back). Each stage can add substitutions, insertions, and deletions, and the later stages can also lose,
copy, or truncate whole strands. 

Every config has two modes. Simple mode only uses the substitution, insertion, and deletion rates, and applies them once
to every strand. Advanced mode turns on everything else (per base rates, homopolymers, GC content, position bias,
truncation, dropout, and the stage specific settings like PCR cycles or years of aging). The defaults are neutral, so a
config built with only the three rates behaves exactly like simple mode even if advanced is turned on.

Most of this information is based off of:
    Heckel, Mikutis, Grass (2019) "A Characterization of the DNA Data Storage Channel", Scientific Reports.
Which gives amazing insights into how/where errors happen in the storage channel back up with experiments.
"""


import math
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

# Using pydantic instead of just dataclasses to validate the config, so we don't have to write our own validation code. 
#   It is a bit of a pain to learn, but it is worth it espically since it can validate when a user changes a field after
#   creation. Everything that is simple is used as types, otherwise we use model validatosr to ensure the config
#   remains valid. Also json serialization is so easy, so worth it.
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, \
    model_validator, ValidationError
from pydantic.dataclasses import dataclass

from Encode.Methods.Encoding import Base


#------------------------------------------------------------------------------#
#                                                                              #
#   TYPES                                                                      #
#                                                                              #
#------------------------------------------------------------------------------#


# A probability, checked by pydantic to be between 0 and 1
Probability = Annotated[float, Field(ge=0, le=1)]

# Probability where can't be 0
NonZeroProbability = Annotated[float, Field(gt=0, le=1)]

# Using Base from Encodings our type

# How copies (synthesis) or reads (sequencing) get handed out to each strand. Using enum instead of string so we can
#   use pydantic's built in validation instead of writing our own and still keep up with linting error.
class CoverageDistribution(StrEnum):
    # Every strand gets exactly the mean amount, no coverage bias at all
    FIXED = "fixed"
    # Random amount with the given mean, what you get from ideal random sampling
    POISSON = "poisson"
    # Poisson with extra spread, matches real data much better based on Heckel et al.
    NEGATIVE_BINOMIAL = "negative_binomial"


# Which stage of the storage channel a config is for. Each stage config has a stage field fixed to its own value, so a
#   saved config knows what it is, and loading can tell which config class to build from the file alone.
class Stage(StrEnum):
    SYNTHESIS = "synthesis"
    AGING = "aging"
    PCR = "pcr"
    SEQUENCING = "sequencing"


# A homopolymer run length, a run of 1 is just a base, so 2 is the shortest homopolymer
RunLength = Annotated[int, Field(ge=2)]


#------------------------------------------------------------------------------#
#                                                                              #
#   CONSTANTS                                                                  #
#                                                                              #
#------------------------------------------------------------------------------#


# Every base in the order the enum lists them (A, C, G, T), so the order of the dictionaries is the same every run
DNA_BASES: Final[tuple[Base, ...]] = tuple(Base)

# Pydantic copies mutable defaults for every instance, so these are never shared
UNIFORM_WEIGHTS: Final[dict[Base, float]] = {base: 1/4 for base in DNA_BASES}
UNIFORM_CONDITIONAL: Final[dict[Base, dict[Base, float]]] = {
    base:
        {
            other_base: 1/3 for other_base in DNA_BASES if other_base != base
        }
        for base in DNA_BASES
    }
DEFAULT_HOMOPOLYMER_MULTIPLIER: Final[dict[int, float]] = {4: 1.5, 5: 2.0, 6: 3.0}


#------------------------------------------------------------------------------#
#                                                                              #
#   HELPER FUNCTION                                                            #
#                                                                              #
#------------------------------------------------------------------------------#


def _check_sums_to_one(weights: dict[Base, float]) -> bool:
    """
    Checks a table of weights sums to 1. Negative weights are already caught by the field types.

    Args:
        weights: The weights to check.

    Returns:
        True if the weights sum to 1 (within floating point error), False otherwise.
    """
    sum_weights: float = 0.0
    for _, weight in weights.items():
        sum_weights += weight

    # Need to use isclose because of floating point error
    return math.isclose(sum_weights, 1)


def _default_substitution_rate_per_base(data: dict[str, Any]) -> dict[Base, float]:
    """
    Default for substitution_rate_per_base, every base gets the overall substitution rate. Pydantic sees this takes one
    argument and passes in the fields validated so far, so it has to be declared after substitution_rate in the config.

    Args:
        data: The fields pydantic has already validated, keyed by field name.

    Returns:
        A dictionary of each base to the overall substitution rate.
    """
    # Pydantic only calls this when the user gave no table at all, so the table isn't in data yet - every base gets the
    #   sub rate. A partial table from the user never comes here, the validator fills its missing bases instead.
    filled_values: dict[Base, float] = {}
    for base in DNA_BASES:
        filled_values[base] = data["substitution_rate"]

    return filled_values



@dataclass
class GCWindow(BaseModel):
    """
    GC content bounds used to scale the error rate at each base. GC content between 40% and 60% is usually the safe
    zone (at least in most papers I've read), below or above that strands are harder to work with.

    For each base we look at a window centered on that base (clipped at the ends of the strand) and get the fraction of
    G and C in it. Inside the bounds the multiplier is 1. Outside the bounds our multiplier grows linearly with how far
    we strive outside the bounds.
    """

    # Ensure we don't allow extra fields, and ensure we always validate on assignment (so if the user changes a field 
    #   after creation it is still validated). This is so much better than hand doing things. So happy I found this.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # The size of the window to look at, centered on the base we are looking at
    window_size: PositiveInt = 20
    # The lower and upper bounds of the GC fraction that we consider "safe" and don't scale the error rate
    lower_bound: Probability = 0.4
    upper_bound: Probability = 0.6

    # Our slope for how much the multiplier grows outside the bounds. Done linearly as I could not find any papers that
    #   give a specific function for how the error rate grows outside the bounds, and this is the simplest way.
    slope: NonNegativeFloat = 10.0


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS METHODS                                                          #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Ensure it's validated even after assignment
    @model_validator(mode="after")
    def _check_bounds(self) -> "GCWindow":
        """
        Checks the bounds are in order, the field types already check they are each between 0 and 1.

        Returns:
            The validated window.

        Raises:
            ValidationError: If the lower bound is above the upper bound.
        """
        if self.lower_bound > self.upper_bound:
            raise ValidationError(f"lower_bound must be <= upper_bound, got {self.lower_bound}, {self.upper_bound}")

        return self


    def multiplier(self, gc_fraction: float) -> float:
        """
        Gets the error multiplier for a given GC fraction.

        Args:
            gc_fraction: The fraction of G and C in the window, between 0 and 1.

        Returns:
            1 inside the bounds, otherwise 1 plus the slope times the distance outside the bounds.
        """
        # We are in bounds, no scaling
        if self.lower_bound <= gc_fraction <= self.upper_bound:
            return 1.0

        # Out of bounds - assume scaling
        distance: float = gc_fraction - self.upper_bound
        # We were actually lower, so fix distance
        if gc_fraction < self.lower_bound:
            distance = self.lower_bound - gc_fraction

        # Return the linear scale
        return 1.0 + self.slope * distance



@dataclass
class ErrorConfig(BaseModel):
    """
    The error settings shared by every stage. Each stage gets its own config, so the same settings can be used for
    different things per stage, e.g. different error rates for different stages of the sequencing process.

    All rates are per base. Simple mode uses the one overall rate for every base. In advanced mode the chance of an
    error at a base is:
        rate for that base * homopolymer multiplier * GC multiplier * position multiplier
    Defaults every multiplier is 1, so it is just the rate for that base, capped at 1.

    For advanced mode, each per base entry by the user is the actual rate for that base, P(error | base is X), so they
    do NOT sum to 1. The conditional tables are different, it's basically "given the error happens, what base do we
    get" - so sum to 1.
    """

    # Same as GC window, we do not allow extra field and we want to validate on assignment.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Allows to repeat same simulation with random numbers - none means different random numbers every time
    seed: int | None = None

    # Turns on everything other than the three rates. When False the simulator ignores the advanced settings. We keep
    #   the advanced settings valid even when advanced is off, so it will be able to store/load without breaking them.
    advanced: bool = False


    # SUBSTITUTIONS -----------------------------------------------------------#

    # The overall chance a base gets substituted. Used in simple mode, but also fills in the per base table in advanced
    #   mode if for any bases nothing is given.
    substitution_rate: Probability = 0.0015

    # Chance each base gets substituted, e.g. {"A": 0.003254}. Defaults to substitution_rate for every base, see
    #   _default_substitution_rate_per_base. If the user only gives some bases, the validator fills the missing ones.
    substitution_rate_per_base: dict[Base, Probability] = Field(default_factory=_default_substitution_rate_per_base)
    # Given a base is substituted, what it becomes - each row must sum to 1 and can't contain itself
    substitution_conditional: dict[Base, dict[Base, Probability]] = UNIFORM_CONDITIONAL


    # INSERTIONS --------------------------------------------------------------#

    # Overall chance a base gets an insertion right after it, used to cover no entries
    insertion_rate: Probability = 0.001

    # Chance of an insertion right after each base
    insertion_rate_after_base: dict[Base, Probability] = {}
    # Given an insertion happens, which base gets inserted - must sum to 1
    insertion_conditional: dict[Base, Probability] = UNIFORM_WEIGHTS
    # Chance the inserted base just repeats the base before it (called stuttering), very common in homopolymers on
    #   nanopore. Will override insertion_conditional when happens.
    insertion_repeat_previous: Probability = 0.0

    # Length of an insertion starts at the min length, then is geometric after it: P(length = min + j) = (1 - p)^j * p,
    #   so the mean is min + (1 - p) / p. A p of 1 means every insertion is exactly the min length. Capped at the max
    #   length. p of 0 would mean an infinite insertion, so p must be above 0, and it is a probability so it can't go
    #   past 1. Min must be at most the max, checked in the validator. Min of 1 means single base insertions can happen.
    insertion_min_length: PositiveInt = 1
    insertion_length_p: NonZeroProbability = 1.0
    insertion_max_length: PositiveInt = 5


    # DELETIONS ---------------------------------------------------------------#

    # Same as the other two rates, but for deletions. Used to fill in empties the per base table in advanced mode.
    deletion_rate: Probability = 0.001

    # Chance each base gets deleted. Missing bases use deletion_rate.
    deletion_rate_per_base: dict[Base, Probability] = {}

    # Same min length plus geometric length as insertions, same (0, 1] range for p
    deletion_min_length: PositiveInt = 1
    deletion_length_p: NonZeroProbability = 1.0
    # None means it can run to the end of the strand (which is just a truncation)
    deletion_max_length: PositiveInt | None = None


    # HOMOPOLYMERS ------------------------------------------------------------#

    # If we want to consider homopolymers - runs at/above the min length scale the error rate of every base in the run.
    homopolymer_check: bool = False
    homopolymer_min_length: RunLength = 4
    # Multiplier per run length for each type of error, the longest run listed is used for longer runs
    homopolymer_substitution_multiplier: dict[RunLength, NonNegativeFloat] = DEFAULT_HOMOPOLYMER_MULTIPLIER
    homopolymer_insertion_multiplier: dict[RunLength, NonNegativeFloat] = DEFAULT_HOMOPOLYMER_MULTIPLIER
    homopolymer_deletion_multiplier: dict[RunLength, NonNegativeFloat] = DEFAULT_HOMOPOLYMER_MULTIPLIER


    # GC CONTENT --------------------------------------------------------------#

    # If we want to consider GC content, see GCWindow for how the multiplier is worked out
    gc_content_check: bool = False
    gc_window: GCWindow = GCWindow()


    # POSITION BIAS -----------------------------------------------------------#

    # Errors typically go up toward the end of the strand. Multiplier goes linearly from 1 at the start of the strand
    #   to 1 + slope at the end.
    position_slope: NonNegativeFloat = 0.0


    # STRAND LOSS -------------------------------------------------------------#

    # Truncation rate - basically just a special case of deletion, the strand is cut and everything past is gone.
    #   This is a per base rate, so longer strands are more likely to be cut: P(strand intact) = (1 - rate)^length.
    truncation_rate: Probability = 0.0

    # Chance a whole strand is lost (stuck to a tube, didnt get swept, no PCR, never read, etc). Happens in simple too.
    #   This is a per strand rate, so it is not affected by length.
    dropout_rate: Probability = 0.001


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CLASS METHODS                                                          #
    #                                                                          #
    #--------------------------------------------------------------------------#



    # Ensure we validate everything to our standards and ensure we check after we change fields - just Guard Clauses.
    @model_validator(mode="after")
    def _check_rates(self) -> "ErrorConfig":
        """
        Checks the rules that look at more than one field. Every field has already passed its own type check by the
        time this runs, so we only need to check how they fit together.

        Returns:
            The validated config, with every per base table filled in to all four bases.

        Raises:
            ValidationError: If the three rates sum past 1 (overall or for one base), a min length is above its max 
                length, or a conditional row contains its own base, is missing a base, or doesn't sum to 1.
        """
        # Probabilities must sum to 1 or less
        if self.substitution_rate + self.insertion_rate + self.deletion_rate > 1:
            raise ValidationError("Substitution, insertion, and deletion rates must sum to 1 or less.")

        # Lengths start at the min, so the min can't be past the max
        if self.insertion_min_length > self.insertion_max_length:
            raise ValidationError(f"insertion_min_length must be <= insertion_max_length, "
                                  f"got {self.insertion_min_length}, {self.insertion_max_length}")
        
        # Deletions with no max can run to the end of the strand, so only check when there is one
        if self.deletion_max_length is not None and self.deletion_min_length > self.deletion_max_length:
            raise ValidationError(f"deletion_min_length must be <= deletion_max_length, "
                                  f"got {self.deletion_min_length}, {self.deletion_max_length}")

        # Fill in any bases the user left out with the overall rate, so after this every table has all four bases
        for base in DNA_BASES:
            self.substitution_rate_per_base.setdefault(base, self.substitution_rate)
            self.insertion_rate_after_base.setdefault(base, self.insertion_rate)
            self.deletion_rate_per_base.setdefault(base, self.deletion_rate)

            # Same as the overall check, but per base - one base can't have more than a 100% chance of some error
            if (self.substitution_rate_per_base[base] +
                self.insertion_rate_after_base[base] +
                self.deletion_rate_per_base[base] > 1):
                raise ValidationError(f"Substitution, insertion, and deletion rates for {base} must sum to 1 or less.")

        # Check that the conditional rates are valid
        if set(self.insertion_conditional) != set(DNA_BASES) or not _check_sums_to_one(self.insertion_conditional):
            raise ValidationError(f"insertion_conditional must have every base and sum to one, "
                                  f"got {set(self.insertion_conditional)}")

        # The substitution conditional is a table, so it must have every base as a row
        if set(self.substitution_conditional) != set(DNA_BASES):
            raise ValidationError(f"substitution_conditional must have a row for each of {DNA_BASES}")
        # But if it does, each row must have every base and sum to 1, and can't contain itself
        for base, row in self.substitution_conditional.items():
            # Cant be inside itself
            if base in row:
                raise ValidationError(f"substitution_conditional row {base} can't contain itself, got {set(row)}")

            # Must have every base besides itself and sum to 1
            if (set(row) != set(DNA_BASES) - {base}) or not _check_sums_to_one(row):
                raise ValidationError(f"substitution_conditional row {base} must have every base "
                                      f"and sum to one, got {set(row)}")

        return self



@dataclass
class SynthesisConfig(ErrorConfig):
    """
    Synthesis writes each strand base by base, and makes many copies of each one. Most insertions and deletions come
    from here (deletions are the most common), and chain termination (basically just stopping the synthesis of a strand)
    gives us truncated strands.
    """

    # Which stage this config is for - fixed, so it can't be changed. Basically so we know what config to build.
    stage: Literal[Stage.SYNTHESIS] = Stage.SYNTHESIS

    # Mean number of copies made of each strand
    sequence_depth: PositiveInt = 1

    # Heckel et al. shows it follows a negative binomial, so most realistic choice, but default will just be fixed
    copy_distribution: CoverageDistribution = CoverageDistribution.FIXED

    # Basically how uneven and nonuniform the copies are in the negative binomial (only used with neg binomial). A
    #   dispersion of 1 is a geometric distribution and nearing 100 is basically uniform.
    copy_dispersion: PositiveFloat = 1.0



@dataclass
class AgingConfig(ErrorConfig):
    """
    Aging is the strands sitting in storage. DNA mostly decays by depurination (molecule falls off and strand break),
    and by deamination (basically just substitution). Heckel et al. finds aging mostly loses strands rather than adding
    errors to them, but we shall leave that up to the user.
    """

    # Again, which stage this config is for
    stage: Literal[Stage.AGING] = Stage.AGING

    # How long the strands have been stored, in years. Used to scale the decay rate. Default is 0, so no decay.
    years: NonNegativeFloat = 0.0

    # Each strand survives with 0.5^(years / half life). How long that is depends on how the DNA is stored, so it is
    #   left to the user but ~500 years is cited for DNA in bone (Heckel et al.), encapsulated DNA lasts far longer.
    half_life_years: PositiveFloat = 500.0

    # Basically a additional substitution rate. Will be applied on top of the normal substitution rate.
    deamination_rate: Probability = 0.0



@dataclass
class PCRConfig(ErrorConfig):
    """
    PCR copies every strand each cycle. The rates here are per base per copy, and only hit the new copies made each
    cycle, so errors made early on get copied into everything after them.

    The bigger problem with PCR is not errors but bias. Each strand copies at a slightly different rate, and over many
    cycles that small difference turns into a huge difference in how many copies each strand has.
    """

    # Which stage this config is for - fixed, so it can't be changed, and saved with the config so loading knows
    #   which config to build
    stage: Literal[Stage.PCR] = Stage.PCR

    # Mimics the PCR process but we will not be making millions of strands
    cycles: NonNegativeInt = 3

    # Fraction of strands copied each cycle, 1 means perfect doubling.
    efficiency: Probability = 1.0
    # Standard deviation of the efficiency across strands, 0 means every strand copies at the same rate
    efficiency_std: NonNegativeFloat = 0.0
    # Our efficiency can also vary across cycles, this is the standard deviation of that variation
    efficiency_cycle_std: NonNegativeFloat = 0.0

    # Strands with GC outside the gc_window bounds amplify worse. Efficiency drops by this much per unit of the whole
    #   strand's GC fraction outside the bounds. Only used when gc_content_check is on.
    gc_efficiency_slope: NonNegativeFloat = 0.0



@dataclass
class SequencingConfig(ErrorConfig):
    """
    Sequencing reads strands from the pool. Errors depend a lot on the platform - but we shall leave that up to the
    biologist to set those rates.

    Reads are sampled from the pool, so strands with more copies (from synthesis and PCR bias) get read more, and some
    strands may never get read at all. Once again we use the same distribution types as synthesis.
    """

    # Same as before - fix our stage so we know what config it is
    stage: Literal[Stage.SEQUENCING] = Stage.SEQUENCING

    # Mean number of reads per unique strand - same as synthesis
    read_depth: PositiveInt = 1
    read_distribution: CoverageDistribution = CoverageDistribution.FIXED
    read_dispersion: PositiveFloat = 1.0



# TODO: finish this lol
class ErrorSimulator:
    def __init__(self,
                 synthesis: SynthesisConfig,
                 aging: AgingConfig,
                 PCR: PCRConfig,
                 sequencing: SequencingConfig,
                 record_errors: bool = True):
        pass



    def add_synthesis_error(self, sequences: list[str]) -> list[str]:
        # Preforms syntehsis error config

        return []

    def add_PCR_error(self, sequences: list[str]) -> list[str]:
        # Performs PCR error config (amplification errors

        # Usually dependent on the GC content and will go by cycles (PCRConfig.cycles). Mimics the PCR process but we
        # will not be millions of strand creations.

        return []

    def add_aging_error(self, sequences: list[str]) -> list[str]:
        # Performs aging error config (AgingConfig.years)

        return []

    def add_sequencing_error(self, sequences: list[str]) -> list[str]:
        # Performs sequencing error config

        return []


    def perform_simulation(self, sequences: list[str]) -> list[str]:


        return []
