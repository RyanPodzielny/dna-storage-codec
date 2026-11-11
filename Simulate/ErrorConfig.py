"""
Module Incliues the configuration classes for the DNA storage error simulation. Every config has two modes. Simple mode 
only uses the substitution, insertion, and deletion rates, and applies them once to every strand. Advanced mode turns on
everything else (per base rates, homopolymers, GC content, position bias, truncation, dropout, and the stage specific 
settings like PCR cycles or years of aging). The defaults are neutral, so a config built with only the three rates 
behaves exactly like simple mode even if advanced is turned on.

Most of this information is based off of:
    Heckel, Mikutis, Grass (2019) "A Characterization of the DNA Data Storage Channel", Scientific Reports.
Which gives amazing insights into how/where errors happen in the storage channel back up with experiments.

"""


import math
from enum import StrEnum
from typing import Annotated, Final, Literal


# Using pydantic instead of just dataclasses to validate the config, so we don't have to write our own validation code. 
#   It is a bit of a pain to learn, but it is worth it espically since it can validate when a user changes a field after
#   creation. Everything that is simple is used as types, otherwise we use model validatosr to ensure the config
#   remains valid. Also json serialization is so easy, so worth it.
from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, \
    model_validator

import numpy as np

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
    # Unknown
    UNKNOWN = "unknown"


# Which stage of the storage channel a config is for. Each stage config has a stage field fixed to its own value, so a
#   saved config knows what it is, and loading can tell which config class to build from the file alone.
class Stage(StrEnum):
    SYNTHESIS = "synthesis"
    AGING = "aging"
    PCR = "pcr"
    SEQUENCING = "sequencing"
    UNKNOWN = "unknown"


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

# Defaults for the conditional tables - every outcome equally likely
UNIFORM_WEIGHTS: Final[dict[Base, NonNegativeFloat]] = {base: 1/4 for base in DNA_BASES}
UNIFORM_CONDITIONAL: Final[dict[Base, dict[Base, NonNegativeFloat]]] = {
    base:
        {
            other_base: 1/3 for other_base in DNA_BASES if other_base != base
        }
        for base in DNA_BASES
    }
DEFAULT_HOMOPOLYMER_MULTIPLIER: Final[dict[PositiveInt, NonNegativeFloat]] = {4: 1.5, 5: 2.0, 6: 3.0}



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
            ValueError: If the lower bound is above the upper bound.
        """
        if self.lower_bound > self.upper_bound:
            raise ValueError(f"lower_bound must be <= upper_bound, got {self.lower_bound}, {self.upper_bound}")

        return self


    def distance(self, gc_fraction: float | np.ndarray) -> float | np.ndarray:
        """
        Gets how far a GC fraction is outside the bounds. Works on a single fraction, or a whole array of them at once
        (e.g. one for every window of a strand).

        Args:
            gc_fraction: The fraction of G and C, between 0 and 1 (or an array of them).

        Returns:
            0 inside the bounds, otherwise how far below the lower bound or above the upper bound it is.
        """
        # How far below the lower bound, and how far above the upper bound - anything not outside is cut to 0. The
        #   bounds are in order, so at most one of them is above 0 (and both are 0 inside the bounds). Using np.maximum
        #   works on single value or on arrays, which speeds things up greatly when we literally have to check windows
        #   of every base in a strand.
        too_low: float | np.ndarray = np.maximum(self.lower_bound - gc_fraction, 0.0)
        too_high: float | np.ndarray = np.maximum(gc_fraction - self.upper_bound, 0.0)

        return too_low + too_high


    def multiplier(self, gc_fraction: float | np.ndarray) -> float | np.ndarray:
        """
        Gets the error multiplier for a given GC fraction. Same as distance, works on a single fraction or an array.

        Args:
            gc_fraction: The fraction of G and C in the window, between 0 and 1 (or an array of them).

        Returns:
            1 inside the bounds, otherwise 1 plus the slope times the distance outside the bounds.
        """
        # Distance is 0 when we are in bounds, so no scaling there - otherwise it is the linear scale
        return 1.0 + self.slope * self.distance(gc_fraction)



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

    # Overriden in subclasses
    stage: Stage = Stage.UNKNOWN

    # Allows to repeat same simulation with random numbers - none means different random numbers every time
    seed: int | None = None

    # Turns on everything other than the three rates. When False the simulator ignores the advanced settings. We keep
    #   the advanced settings valid even when advanced is off, so it will be able to store/load without breaking them.
    advanced_mode: bool = False


    # SUBSTITUTIONS -----------------------------------------------------------#

    # The overall chance a base gets substituted. Used in simple mode, but also fills in the per base table in advanced
    #   mode if for any bases nothing is given. Depends heavily on the stage, this just given as a ballpark.
    substitution_rate: Probability = 0.0005

    # Chance each base gets substituted, e.g. {"A": 0.003254}
    substitution_rate_per_base: dict[Base, Probability] = {}
    # Given a base is substituted, what it becomes - each row must sum to 1 and can't contain itself. Uniform by
    #   default, each other base equally likely.
    substitution_conditional: dict[Base, dict[Base, Probability]] = UNIFORM_CONDITIONAL


    # INSERTIONS --------------------------------------------------------------#

    # Overall chance a base gets an insertion right after it, used to cover no entries. Once again ballpark - depends
    insertion_rate: Probability = 0.001

    # Chance of an insertion right after each base
    insertion_rate_after_base: dict[Base, Probability] = {}
    # Given an insertion happens, which base inserted - must sum to 1. Uniform by default, each base equally likely.
    insertion_conditional: dict[Base, Probability] = UNIFORM_WEIGHTS
    # Chance the inserted base just repeats the base before it (called stuttering), very common in homopolymers on
    #   nanopore. Will override insertion_conditional when happens.
    insertion_repeat_previous: Probability = 0.1

    # Length of an insertion starts at the min length, then is geometric after it: P(length = min + j) = (1 - p)^j * p,
    #   so the mean is min + (1 - p) / p. A p of 1 means every insertion is exactly the min length. Capped at the max
    #   length. p of 0 would mean an infinite insertion, so p must be above 0, and it is a probability so it can't go
    #   past 1. Min must be at most the max, checked in the validator. Min of 1 means single base insertions can happen.
    insertion_min_length: PositiveInt = 1
    # Probability of insertion being a single base (1 = 100% chance of single). Nearly all insertions are a single base
    insertion_length_probability: NonZeroProbability = 0.98
    # Capping how long an inesertion can possibly be
    insertion_max_length: PositiveInt = 5


    # DELETIONS ---------------------------------------------------------------#

    # Same as the other two rates, but for deletions. Ballpark again, depends on stage
    deletion_rate: Probability = 0.001

    # Chance each base gets deleted
    deletion_rate_per_base: dict[Base, Probability] = {}

    # Same min length plus geometric length as insertions - 
    deletion_min_length: PositiveInt = 1
    # Probability of deletion being a single base (1 = 100% chance of single) - depends on stage but can be quite high
    deletion_length_probability: NonZeroProbability = .95
    # None means it can run to the end of the strand (which is just a truncation)
    deletion_max_length: PositiveInt | None = None


    # HOMOPOLYMERS ------------------------------------------------------------#

    # If we want to consider homopolymers - once a run reaches the min length, every base from there on has its rate
    #   scaled by how long the run is up to that base (so A A A A A is 1 2 3 4 5, and only the 4th and 5th are scaled).
    #   Only used in advanced mode, so on by default just means it is on whenever advanced mode is.
    homopolymer_check: bool = True
    # Min length before we start useing the multiplier
    homopolymer_min_length: RunLength = 4
    # Multiplier per run length for each type of error, the longest run listed is used for longer runs. Depends on stage
    #   and on techonology being used - defaults are enough to show the effect.
    homopolymer_substitution_multiplier: dict[RunLength, NonNegativeFloat] = DEFAULT_HOMOPOLYMER_MULTIPLIER
    homopolymer_insertion_multiplier: dict[RunLength, NonNegativeFloat] = DEFAULT_HOMOPOLYMER_MULTIPLIER
    homopolymer_deletion_multiplier: dict[RunLength, NonNegativeFloat] = DEFAULT_HOMOPOLYMER_MULTIPLIER


    # GC CONTENT --------------------------------------------------------------#

    # If we want to consider GC content, see GCWindow for how the multiplier is worked out. Same as homopolymers, only
    #   used in advanced mode so on by default means on whenever advanced mode is.
    gc_content_check: bool = True
    gc_window: GCWindow = GCWindow()


    # POSITION BIAS -----------------------------------------------------------#

    # Errors typically go up toward the end of the strand. Multiplier goes linearly from 1 at the start of the strand
    #   to 1 + slope at the end. Again depends on stage and technology, defaults are enough to show the effect.
    position_slope: NonNegativeFloat = 0.05


    # STRAND LOSS -------------------------------------------------------------#

    # Truncation rate - basically just a special case of deletion, the strand is cut and everything past is gone.
    #   This is a per base rate, so longer strands are more likely to be cut: P(strand intact) = (1 - rate)^length.
    truncation_rate: Probability = 0.00004

    # Chance a whole strand is lost (stuck to a tube, didnt get swept, no PCR, never read, etc). Happens in simple too.
    #   This is a per strand rate, so it is not affected by length. Again depends on stage
    dropout_rate: Probability = 0.0001


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
            The validated config.

        Raises:
            ValueError: If the three rates sum past 1 (overall or for one base), a min length is above its max 
                length, or a conditional row contains its own base, is missing a base, or doesn't sum to 1.
        """

        # Perform gaurd clauses - valide rates > 1, min > max, and conditional tables complete, and sum = 1
        self._validate_probabilities()
        self._validate_lengths()
        self._validate_rates()
        self._validate_conditionals()

        return self


    def _validate_probabilities(self) -> None:
        """
        Checks the three rates sum to 1 or less, both overall and for each base. The field types already check they are
        between 0 and 1, so we only need to check the sum.

        Raises:
            ValueError: If the three rates sum past 1 (overall or for one base).
        """

        # Probabilities must sum to 1 or less
        if self.substitution_rate + self.insertion_rate + self.deletion_rate > 1:
            raise ValueError("Substitution, insertion, and deletion rates must sum to 1 or less.")


    def _validate_lengths(self) -> None:
        """
        Checks the min lengths are at most the max lengths. The field types already check they are
        positive, so we only need to check the min/max relationship.

        Raises:
            ValueError: If a min length is greater than its corresponding max length.
        """
         # Lengths start at the min, so the min can't be past the max
        if self.insertion_min_length > self.insertion_max_length:
            raise ValueError(f"insertion_min_length must be <= insertion_max_length, "
                                f"got {self.insertion_min_length}, {self.insertion_max_length}")
        
        # Deletions with no max can run to the end of the strand, so only check when there is one
        if self.deletion_max_length is not None and self.deletion_min_length > self.deletion_max_length:
            raise ValueError(f"deletion_min_length must be <= deletion_max_length, "
                                f"got {self.deletion_min_length}, {self.deletion_max_length}")


    def _validate_rates(self) -> None:
        """
        Checks the three rates for every base sum to 1 or less. Any base missing from a per base table uses the overall
        rate, but we don't write it into the table - otherwise changing the overall rate after the config is made
        wouldn't change those bases (the table would still hold the old one). See get_base_rates.

        Raises:
            ValueError: If the three rates for a base sum past 1.
        """
  
        for base in DNA_BASES:
            # Same as the overall check, but per base - one base can't have more than a 100% chance of some error
            if sum(self.get_base_rates(base)) > 1:
                raise ValueError(f"Substitution, insertion, and deletion rates for {base} must sum to 1 or less.")


    def get_base_rates(self, base: Base) -> tuple[float, float, float]:
        """
        Gets the three rates for one base, using the overall rate for any base the per base tables don't have.

        Args:
            base: The base to get the rates for.

        Returns:
            A tuple containing:
                - The chance the base is substituted.
                - The chance the base gets an insertion right after it.
                - The chance the base is deleted.
        """
        return (
            self.substitution_rate_per_base.get(base, self.substitution_rate),
            self.insertion_rate_after_base.get(base, self.insertion_rate),
            self.deletion_rate_per_base.get(base, self.deletion_rate)
        )


    def _validate_conditionals(self) -> None:
        """
        Checks the conditional tables are complete and each sums to 1. Nothing is filled in, the user has to give every
        base. The insertion conditional needs all four bases, and the substitution conditional needs a row for every
        base with all three other bases in it.

        Raises:
            ValueError: If a table is missing a base or row, a substitution row contains its own base, or a table
                doesn't sum to 1.
        """
        insertion_total: float = 0.0

        for base in DNA_BASES:
            # INSERTION - given an insertion happens, which base is inserted. Needs every base.
            if base not in self.insertion_conditional:
                raise ValueError(f"insertion_conditional is missing {base}, every base must be given")
            insertion_total += self.insertion_conditional[base]


            # SUBSTITUTION - given a base is substituted, what it becomes. Every base needs its own row.
            if base not in self.substitution_conditional:
                raise ValueError(f"substitution_conditional is missing the row for {base}")

            # Get the conditional row for this base
            row: dict[Base, float] = self.substitution_conditional[base]

            # A base can't be substituted with itself
            if base in row:
                raise ValueError(f"substitution_conditional {base} can't contain {base}, a base can't become itself")

            # Needs every other base, adding them up as we go
            row_total: float = 0.0
            for other_base in DNA_BASES:
                if other_base == base:
                    continue
                if other_base not in row:
                    raise ValueError(f"substitution_conditional row {base} is missing {other_base}, every other base "
                                     f"must be given")

                # Add up the row to check it sums to 1
                row_total += row[other_base]

            # The base has to become one of the others, so the row must sum to 1 (isclose for floating point error)
            if not math.isclose(row_total, 1):
                raise ValueError(f"substitution_conditional row {base} must sum to 1, got {round(row_total, 6)}")

        # One of the bases has to be inserted, so it must sum to 1 - only known once every base has been added up
        if not math.isclose(insertion_total, 1):
            raise ValueError(f"insertion_conditional must sum to 1, got {round(insertion_total, 6)}")



class SynthesisConfig(ErrorConfig):
    """
    Synthesis writes each strand base by base, and makes many copies of each one. Most insertions and deletions come
    from here (deletions are the most common), and chain termination (basically just stopping the synthesis of a strand)
    gives us truncated strands.
    """

    # Which stage this config is for - fixed, so it can't be changed. Basically so we know what config to build.
    stage: Literal[Stage.SYNTHESIS] = Stage.SYNTHESIS


    # SIMPLE MODE DEFAULTS
    # Deletions are the most common error when writing, then substitutions, and insertions are the rarest
    substitution_rate: Probability = 0.001
    insertion_rate: Probability = 0.0005
    deletion_rate: Probability = 0.002
    dropout_rate: Probability = 0.0001


    # Mean number of copies made of each strand. This is so much higher in real life - but can't do millions of strands
    synthesis_depth: PositiveInt = 3                    # INCLUDED IN SIMPLE MODE

    # Heckel et al. shows it follows a negative binomial, so most realistic choice, but default will just be fixed
    copy_distribution: CoverageDistribution = CoverageDistribution.FIXED

    # Basically how uneven and nonuniform the copies are in the negative binomial (only used with neg binomial). A
    #   dispersion of 1 is a geometric distribution and nearing 100 is basically uniform.
    copy_dispersion: PositiveFloat = 5.0



class AgingConfig(ErrorConfig):
    """
    Aging is the strands sitting in storage. DNA mostly decays by depurination (molecule falls off and strand break),
    and by deamination (basically just substitution). Heckel et al. finds aging mostly loses strands rather than adding
    errors to them, but we shall leave that up to the user.
    """

    # Again, which stage this config is for
    stage: Literal[Stage.AGING] = Stage.AGING


    # SIMPLE MODE DEFAULTS
    # Storage doesn't really add errors to a strand, it loses the whole strand or deaminates a base (typically)
    substitution_rate: Probability = 0.0
    insertion_rate: Probability = 0.0
    deletion_rate: Probability = 0.0
    dropout_rate: Probability = 0.000001


    # How long the strands have been stored, in years. Used to scale the decay rate - default is 0, so no decay.
    years: NonNegativeFloat = 0.0                   # INCLUDED IN SIMPLE MODE

    # Each strand survives with 0.5^(years / half life). How long that is depends on how the DNA is stored, so it is
    #   left to the user but ~500 years is cited for DNA in bone (Heckel et al.), encapsulated DNA lasts far longer.
    half_life_years: PositiveFloat = 500.0

    # Basically a additional substitution rate, but per year - so the longer the strands are stored the more of them we
    #   get. Chance a base is deaminated after all the years is 1 - (1 - rate)^years. This basically is just (1 - r) * 
    #   (1 - r) * ... (like half life) for all the years, i.e. it tells us that the chance of an error happened once
    deamination_rate: Probability = 0.00003



class PCRConfig(ErrorConfig):
    """
    PCR copies every strand each cycle. The rates here are per base per copy, and only hit the new copies made each
    cycle, so errors made early on get copied into everything after them.

    The bigger problem with PCR is not errors but bias. Each strand copies at a slightly different rate, and over many
    cycles that small difference turns into a huge difference in how many copies each strand has.
    """

    #   Same as before, fix our stage so we know what config it is
    stage: Literal[Stage.PCR] = Stage.PCR


    # SIMPLE MODE DEFAULTS
    # About what a high fidelity polymerase gives per base per copy. Very low, nearly all of them are substitutions.
    substitution_rate: Probability = 0.00001
    insertion_rate: Probability = 0.000001
    deletion_rate: Probability = 0.000001
    dropout_rate: Probability = 0.000001


    # Mimics the PCR process but we will not be making millions of strands
    cycles: NonNegativeInt = 3                  # INCLUDED IN SIMPLE MODE

    # Fraction of strands copied each cycle, 1 means perfect doubling.
    efficiency: Probability = 0.9
    # Standard deviation of the efficiency across strands, 0 means every strand copies at the same rate
    efficiency_std: NonNegativeFloat = 0.02
    # Our efficiency can also vary across cycles, this is the standard deviation of that variation
    efficiency_cycle_std: NonNegativeFloat = 0.01

    # Strands with GC outside the gc_window bounds amplify worse. Efficiency drops by this much per unit of the whole
    #   strand's GC fraction outside the bounds. Only used when gc_content_check is on.
    gc_efficiency_slope: NonNegativeFloat = 0.5



class SequencingConfig(ErrorConfig):
    """
    Sequencing reads strands from the pool. Errors depend a lot on the platform - but we shall leave that up to the
    biologist to set those rates.

    Reads are sampled from the pool, so strands with more copies (from synthesis and PCR bias) get read more, and some
    strands may never get read at all. Once again we use the same distribution types as synthesis.
    """

    # Same as before - fix our stage so we know what config it is
    stage: Literal[Stage.SEQUENCING] = Stage.SEQUENCING


    # SIMPLE MODE DEFAULTS
    # Most to all all of its errors are substitutions, insertions and deletions are very rare.
    substitution_rate: Probability = 0.002
    insertion_rate: Probability = 0.00001
    deletion_rate: Probability = 0.00001
    dropout_rate: Probability = 0.0001


    # Mean number of reads per unique strand - same as synthesis
    read_depth: PositiveInt = 5                 # INCLUDED IN SIMPLE MODE

    # Same as synthesis, but for reads
    read_distribution: CoverageDistribution = CoverageDistribution.FIXED
    read_dispersion: PositiveFloat = 5.0