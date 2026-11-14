"""
Module simulates the errors a DNA strand picks up as it moves through the storage channel. There are four stages, each
with its own config: synthesis (writing the strands), aging (storing them), PCR (amplifying them), and sequencing
(reading them back). Each stage can add substitutions, insertions, and deletions, and the later stages can also lose,
copy, or truncate whole strands. 

Most of this information is based off of:
    Heckel, Mikutis, Grass (2019) "A Characterization of the DNA Data Storage Channel", Scientific Reports.
Which gives amazing insights into how/where errors happen in the storage channel back up with experiments.
"""


from __future__ import annotations

from enum import StrEnum
from typing import cast

# Going to np for random numbers as it is both faster and we will need to draw from distributions for some stages.
import numpy as np

from Encode.Methods.Encoding import Base
from Helpers.Errors import SimulationError
from Simulate.ErrorConfig import DNA_BASES, CoverageDistribution, GCWindow, Stage, ErrorConfig, SynthesisConfig, \
    AgingConfig, PCRConfig, SequencingConfig



class ErrorRecord:
    """
    Holds the error counts for each stage. Each stage has its own record, and each record has a count of how many
    substitutions, insertions, deletions, truncations, and dropouts it added to the strands.
    """


    def __init__(self):
        self.counts: dict[ErrorType, int] = {
            self.ErrorType.SUBSTITUTION: 0,
            self.ErrorType.INSERTION: 0,
            self.ErrorType.DELETION: 0,
            self.ErrorType.TRUNCATION: 0,
            self.ErrorType.DROPOUT: 0
        }

        # And our list of records for each stage. Stored as (index, pos in strand, stage, type of error, strand)
        self.records: list[dict[str, (int | None) | Stage | ErrorType | str]] = []


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   CONSTANTS                                                              #
    #                                                                          #
    #--------------------------------------------------------------------------#


    # Types of errors that can occur
    class ErrorType(StrEnum):
        SUBSTITUTION = "substitution"
        INSERTION = "insertion"
        DELETION = "deletion"
        TRUNCATION = "truncation"
        DROPOUT = "dropout"


    # For recording so you know what each field means (better for serialization in a JSON)
    class ErrorDictKeys(StrEnum):
        INDEX = "index"
        POSITION = "position"
        STAGE = "stage"
        ERROR_TYPE = "error_type"
        SEQUENCE = "sequence"


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def record_error(self, 
                     index: int, 
                     position: int | None, 
                     stage: Stage, 
                     error_type: ErrorType, 
                     sequence: str) -> None:
        """
        Records an error that occured on a strand. Increments the count for the error type, and adds a record to list.

        Args:
            index: The index of the strand in the original list.
            position: The position of the error in the strand.
            stage: The stage the error occured on.
            error_type: The type of error that occured.
            sequence: The sequence of the strand after the error occured.
        """
        self.counts[error_type] += 1
        self.records.append({
            ErrorDictKeys.INDEX: index,
            ErrorDictKeys.POSITION: position,
            ErrorDictKeys.STAGE: stage,
            ErrorDictKeys.ERROR_TYPE: error_type,
            ErrorDictKeys.SEQUENCE: sequence
        })

        
# Short name for the types/leys, so we don't have to write ErrorRecord. every time
ErrorType = ErrorRecord.ErrorType
ErrorDictKeys = ErrorRecord.ErrorDictKeys



class ErrorSimulator:
    """
    Base for the simulators. Runs strands through the four stages of the storage channel, in order: synthesis, aging,
    PCR, sequencing. Each stage uses its own config, so e.g. sequencing can have far higher rates than aging.

    Holds everything both modes share (configs, random generators, records, the stage order). Each mode only has to
    say how one stage is applied to the strand.
    """


    def __init__(self,
                 synthesis: SynthesisConfig,
                 aging: AgingConfig,
                 PCR: PCRConfig,
                 sequencing: SequencingConfig):
        """
        Initializes the simulator with a config for every stage. Configs are already validated by pydantic, so there
        is nothing to check here.

        Args:
            synthesis: The config for writing the strands.
            aging: The config for storing the strands.
            PCR: The config for amplifying the strands.
            sequencing: The config for reading the strands back.
        """

        # Set up all our configs
        self.synthesis: SynthesisConfig = synthesis
        self.aging: AgingConfig = aging
        self.PCR: PCRConfig = PCR
        self.sequencing: SequencingConfig = sequencing

        # The order the stages are applied in, and the config for each stage - makes it easy to loop through
        self.stages: list[tuple[Stage, ErrorConfig]] = [
            (self.synthesis.stage, self.synthesis),
            (self.aging.stage, self.aging),
            (self.PCR.stage, self.PCR),
            (self.sequencing.stage, self.sequencing)
        ]

        # Create our random generators now - each stage got their own. If None - stage gets a new random gene every time
        self.generators: dict[Stage, np.random.Generator] = {}
        for stage, config in self.stages:
            self.generators[stage] = np.random.default_rng(config.seed)

        # Error record created on simulate, which will hold the counts and records of every error that happens
        self.error_record: ErrorRecord 


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE SHARED METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def _get_aged_rate(rate: float, years: float) -> float:
        """
        Turns a per year rate into the chance it happens at least once over all the years. Something only doesn't
        happen if it makes it through every year, so the chance is 1 - (1 - rate)^years. Basically just (1 - r) *
        (1 - r) * ... for the years, i.e. tells us that the chance of an error happened once on that base over years

        Args:
            rate: The chance it happens in a single year.
            years: How many years have gone by.

        Returns:
            The chance an error happened at least once, 0 when no years have gone by.
        """
        return 1 - (1 - rate) ** years


    @staticmethod
    def _get_gc_fraction(sequence: str) -> float:
        """
        Gets the fraction of G and C in a sequence - doesn't need to be full strand can be just a window.

        Args:
            sequence: The strand to look at.

        Returns:
            The fraction of the strand that is G or C, between 0 and 1.
        """
        return (sequence.count(Base.G) + sequence.count(Base.C)) / len(sequence)


    @staticmethod
    def _repeat_strands(counts: np.ndarray,
                        indices: list[int],
                        sequences: list[str]) -> tuple[list[int], list[str]]:
        """
        Repeats every strand by its count, used to make the copies synth and seq.

        Args:
            counts: How many times each strand is repeated - lock step with sequences.
            indices: The index of the original strand each strand came from, lock step with sequences.
            sequences: The strands to repeat.

        Returns:
            A tuple containing:
                - The index of the original strand each repeated strand came from.
                - The repeated strands, a strand with a count of 0 is not in them.

        Raises:
            SimulatorError: If a strand has a count less than 0, which shouldn't happen.
        """
        output_indices: list[int] = []
        output: list[str] = []
        # For every strand, add it to the output as many times as its count says
        for count, index, sequence in zip(counts, indices, sequences):
            # Shouldn't happen, but just in case
            if count < 0:
                raise SimulationError(
                    message=f"Count for strand {index} is {count}, must be above 0 to repeat it"
                )
            
            # Add the strand to the output as many times as its count says
            for _ in range(count):
                output_indices.append(index)
                output.append(sequence)

        return output_indices, output


    @staticmethod
    def _draw_copy_counts(rng: np.random.Generator,
                          distribution: CoverageDistribution,
                          mean: float,
                          dispersion: float,
                          size: int) -> np.ndarray:
        """
        Draws how many copies or reads each strand gets based on the selected coverage distribution. 

        Args:
            rng: The random generator of the stage we are in.
            distribution: Which distribution the counts follow.
            mean: The mean count per strand.
            dispersion: How uneven the counts are, only used by the negative binomial.
            size: How many strands to draw a count for.

        Returns:
            An array with the count for each strand, a count of 0 means the strand got nothing. Easier just to deal with
            it as an np array than cast to a list - going to need to apply to strands with np anways.

        Raises:
            SimulatorError: If the distribution is not a valid coverage distribution.
        """
        # In case something silly happens (shouldn't get based typing) (also mean will always be validated by pydantic)
        if distribution not in CoverageDistribution or distribution == CoverageDistribution.UNKNOWN:
            raise SimulationError(
                message=f"Distribution {distribution} is not a valid coverage distribution"
            )

        # POISSON - what you get from ideal random sampling
        if distribution == CoverageDistribution.POISSON:
            return rng.poisson(mean, size)

        # NEGATIVE BINOMIAL - numpy wants the number of successes (our dispersion) and the chance of success. With
        #   p = dispersion / (dispersion + mean) the mean works out to our mean, and a small dispersion spreads it out.
        if distribution == CoverageDistribution.NEGATIVE_BINOMIAL:
            probability: float = dispersion / (dispersion + mean)
            return rng.negative_binomial(dispersion, probability, size)

        # FIXED - every strand gets exactly the mean
        return np.full(size, round(mean))


    def _copy_strands(self,
                      indices: list[int],
                      sequences: list[str],
                      config: SynthesisConfig,
                      distribution: CoverageDistribution) -> tuple[list[int], list[str]]:
        """
        Synthesis makes many copies of each strand, how many each one gets comes from the copy distribution. The copies
        are made here before any errors are added, since every copy is written on its own and gets its own errors.

        Args:
            indices: The index of the original strand each strand came from, lines up with sequences.
            sequences: The strands to copy.
            config: The synthesis config.
            distribution: Which distribution the copies follow. Simple mode is always fixed (hence why we pass it by in
                params and not by config)

        Returns:
            A tuple containing:
                - The index of the original strand each copy came from.
                - The copies, a strand that got 0 copies is gone.
        """
        # Get stages random gen
        rng: np.random.Generator = self.generators[config.stage]

        # Basically just seeing how many copies each strand gets
        counts: np.ndarray = self._draw_copy_counts(
            rng=rng,
            distribution=distribution,
            mean=config.synthesis_depth,
            dispersion=config.copy_dispersion,
            size=len(sequences)
        )

        # Go through every strand to see if we lost a strand (not read or synthesized)
        for count, index, sequence in zip(counts, indices, sequences):
            # Strand was never written, same as losing the whole strand
            if count == 0:
                self.error_record.record_error(index, None, config.stage, ErrorType.DROPOUT, sequence)

        # And return the copies, stand with their proper counts 
        return self._repeat_strands(counts, indices, sequences)


    def _apply_dropout_errors(self,
                              indices: list[int],
                              sequences: list[str],
                              config: ErrorConfig,
                              dropout_rate: float,
                              survival_chance: float = 1.0) -> tuple[list[int], list[str]]:
        """
        Drops whole strands. Every strand has the dropout rate, and a stage can add its own chance of the strand
        surviving on top of it (advanced aging uses this for decay). Happens basically at every stage, we do this
        before each stage does it own this so we can kinda cheat and not be forced to put errors on to strands we
        are then planning to delete.

        Args:
            indices: The index of the original strand each strand came from, lines up with sequences.
            sequences: The strands going into the stage.
            config: The config of the stage.
            dropout_rate: Chance each strand is lost. Usually the config's, simple aging scales it by the years 
            survival_chance: Extra chance a strand survives the stage, 1 means only the dropout rate is used.

        Returns:
            A tuple containing:
                - The index of the original strand each surviving strand came from.
                - The strands that survived.
        """

        rng: np.random.Generator = self.generators[config.stage]

        # Strand has to make it past both, so the chances multiply
        keep_chance: float = (1 - dropout_rate) * survival_chance
        rolls: np.ndarray = rng.random(len(sequences))

        output_indices: list[int] = []
        output: list[str] = []
        # Go through each strand to see if it survives (so we can record errors for the ones that don't)
        for roll, index, sequence in zip(rolls, indices, sequences):
            # Whole strand is lost, bye bye
            if roll >= keep_chance:
                self.error_record.record_error(index, None, config.stage, ErrorType.DROPOUT, sequence)
                continue

            # Survived, so keep them
            output_indices.append(index)
            output.append(sequence)

        return output_indices, output


    def _amplify_strands(self,
                         indices: list[int],
                         sequences: list[str],
                         config: PCRConfig) -> tuple[list[int], list[str]]:
        """
        PCR trys to copy the strands every cycle. A strand is copied with the chance given by its efficiency, and only 
        the new copy gets errors - so an error made in an early cycle is copied into everything made from it after.

        The efficiency of a strand is the configs efficiency, plus how much that strand differs (the same for every 
        copy of the same original strand), plus how much the cycle differs, minus the GC penalty. Kept between 0 and 1.
        Efficiency is advanced only, in simple mode every strand is copied every cycle (perfect doubling).

        Args:
            indices: The index of the original strand each strand came from, lines up with sequences.
            sequences: The strands to amplify.
            config: The PCR config.

        Returns:
            A tuple containing:
                - The index of the original strand each strand in the pool came from.
                - The pool after every cycle, the strands that came in plus every copy made.
        """

        rng: np.random.Generator = self.generators[config.stage]

        # Copy the lists so adding the new copies onto the list doesn't change the lists we were given
        indices = indices.copy()
        sequences = sequences.copy()

        # How much each original strand differs from the mean efficiency. Drawn once per original strand, so every
        #   copy of it amplifies just as well (or badly) in every cycle - this is what turns into bias over the cycles.
        strand_offsets: dict[int, float] = {}
        if config.advanced_mode:
            for index in indices:
                if index not in strand_offsets:
                    strand_offsets[index] = float(rng.normal(0.0, config.efficiency_std))

        # Now basically just run the cycles, every cycle can be a bit better or worse than the mean too. The new copies 
        #   join the pool and can be copied themselves in the next cycle, so the pool grows exponentially.
        for _ in range(config.cycles):
            # The whole cycle can go a bit better or worse too - simple mode just doubles
            cycle_offset: float = 0.0
            if config.advanced_mode:
                cycle_offset = float(rng.normal(0.0, config.efficiency_cycle_std))

            new_indices: list[int] = []
            new_sequences: list[str] = []
            for index, sequence in zip(indices, sequences):
                # Simple mode copies every strand, so only advanced has to check if the strand was copied
                if config.advanced_mode:
                    efficiency: float = config.efficiency + strand_offsets[index] + cycle_offset

                    # Strands with GC outside the bounds amplify worse, by how far the whole strand is outside of them.
                    #   WE are using the efficiency slope here, not the GC slope since efficiency is different than the 
                    #   GC penalty we want to apply to per base. This is look at the strand as a whole, not each base.
                    if config.gc_content_check:
                        gc_fraction: float = self._get_gc_fraction(sequence)
                        efficiency -= config.gc_efficiency_slope * float(config.gc_window.distance(gc_fraction))

                    # Strand was not copied this cycle. Efficiency doesn't need to be clipped - roll is between 0 and 1]
                    if rng.random() >= efficiency:
                        continue

                # Only the new copy gets errors, the strand it was copied from stays as it is
                new_sequence: str = self._apply_errors(index, sequence, config)

                # Copy was cut or deleted down to nothing after applying errors, so there is no copy
                if not new_sequence:
                    continue

                new_indices.append(index)
                new_sequences.append(new_sequence)

            # The new copies join the pool, and can be copied themselves next cycle
            indices.extend(new_indices)
            sequences.extend(new_sequences)

        return indices, sequences


    def _age_strands(self,
                     indices: list[int],
                     sequences: list[str],
                     config: AgingConfig) -> tuple[list[int], list[str]]:
        """
        Applies aging errors to the strands.

        Args:
            indices: The index of the original strand each strand came from, lines up with sequences.
            sequences: The strands going into the stage.
            config: The config of the stage.

        Returns:
            A tuple containing:
                - The index of the original strand each of those strands came from.
                - The strands coming out of the stage, dropped strands are removed.
        """
        if config.stage != Stage.AGING:
            raise SimulationError(
                message=f"Stage {config.stage} is not aging, cannot apply aging errors"
            )

        # Default is 1, so if the stage doesn't have a survival chance it doesn't change the dropout rate
        survival_chance: float = 1.0
        dropout_rate: float = config.dropout_rate

        # Advanced mode uses the half life to calculate the chance a strand survives on top of the dropout rate
        if config.advanced_mode:
            survival_chance = 0.5 ** (config.years / config.half_life_years)
        # Simple just scales the dropout rate by the years
        else:
            dropout_rate = self._get_aged_rate(dropout_rate, config.years)

        # See what strands survive (we changing place but might as well return new lists)
        indices, sequences = self._apply_dropout_errors(indices, sequences, config, dropout_rate, survival_chance)
        
        return indices, sequences


    def _read_strands(self,
                      indices: list[int],
                      sequences: list[str],
                      config: SequencingConfig,
                      distribution: CoverageDistribution) -> tuple[list[int], list[str]]:
        """
        Sequencing samples reads from the 'pool' (basically where we are grabbing our strands from) how many reads each
        depends on user set distribution. The read depth is the mean number of reads per original strand, not per strand
        in the pool - meaning an original strand with more copies in the pool gets more reads. The reads are picked here
        before any error are added, since every read gets its own errors. Happens before applying seq errors.

        Args:
            indices: The index of the original strand each strand came from, lines up with sequences.
            sequences: The pool of strands to read from.
            config: The sequencing config.
            distribution: Which distribution to follow. Simple mode is always fixed (hence why we pass it by in params 
                and not by config)

        Returns:
            A tuple containing:
                - The index of the original strand each read came from.
                - The reads, a strand that got 0 reads is not in them.
        """

        rng: np.random.Generator = self.generators[config.stage]

        # Nothing in the pool to read - can happen if the previous stages lost everything (pretty unrealistic tho)
        if not sequences:
            return [], []

        # Where every copy of each original strand sits in the pool - so copies in a sense occupy the same space
        pool_spots: dict[int, list[int]] = {}
        for spot, index in enumerate(indices):
            if index not in pool_spots:
                pool_spots[index] = []
            pool_spots[index].append(spot)

        # Has to be ints (zeros gives floats by default), since we use the counts to repeat the strands
        counts: np.ndarray = np.zeros(len(sequences), dtype=int)

        # FIXED - every original strand gets exactly the read depth, no matter how many copies of it are in the pool.
        #   Each read is of a random one of its copies, since the copies can differ from the errors they picked up.
        if distribution == CoverageDistribution.FIXED:
            for spots in pool_spots.values():
                # Basically just make it so every copy of original strand gets 'read'
                picks: np.ndarray = rng.integers(len(spots), size=config.read_depth)
                for pick in picks:
                    counts[spots[pick]] += 1

        # Otherwise every strand in the pool draws its own number of reads
        else:
            # The mean is set so the reads average out to the read depth per original strand, and an original strand 
            #   with more copies ends up with more reads.
            mean: float = config.read_depth * len(pool_spots) / len(sequences)

            counts = self._draw_copy_counts(
                rng=rng,
                distribution=distribution,
                mean=mean,
                dispersion=config.read_dispersion,
                size=len(sequences)
            )

        # An original strand that was in the pool but had none of its copies read is lost, so record it
        for index, spots in pool_spots.items():
            total_reads: int = 0
            for spot in spots:
                total_reads += int(counts[spot])

            # We lost it, goodbye friend
            if total_reads == 0:
                self.error_record.record_error(index, None, config.stage, ErrorType.DROPOUT, sequences[spots[0]])

        # And return the reads, stand with their proper counts 
        return self._repeat_strands(counts, indices, sequences)


    def _apply_errors(self, index: int, sequence: str, config: ErrorConfig) -> str:
        """
        Applies the errors of the stage to one strand, picking simple or advanced mode from the config.

        Args:
            index: The index of the original strand this strand came from, for the records.
            sequence: The strand to add errors to.
            config: The config of the stage.

        Returns:
            The strand after the errors. Can be empty if the strand was cut or deleted down to nothing.
        """
        if config.advanced_mode:
            return self._apply_advanced_errors(index, sequence, config)

        return self._apply_simple_errors(index, sequence, config)


    def _apply_stage(self,
                     indices: list[int],
                     sequences: list[str],
                     config: ErrorConfig) -> tuple[list[int], list[str]]:
        """
        Modes first dropout (unless aging as its has a special drop out rate), then the stage goes like this:
            1. Whole strands are dropped (dropout rate, and decay for aging)
            2. The stage does its own thing - synthesis makes copies, PCR amplifies, sequencing samples reads
            3. Every strand gets its own errors
        PCR is the odd one out, its errors only go on the new copies so they are added while it amplifies.

        Advanced mode uses everything in the config. Simple mode only uses the three rates, dropout, and what is marked
        as simple in the configs (depths, years, cycles):
            synthesis   - every strand is copied synthesis_depth times
            aging       - nothing extra, but its rates are per year (scaled by years, so 0 years is no aging)
            PCR         - every strand doubles each cycle, errors only on the new copies
            sequencing  - every original strand gets read_depth reads, each of a random one of its copies

        Args:
            indices: The index of the original strand each strand came from, lines up with sequences.
            sequences: The strands going into the stage.
            config: The config of the stage.

        Returns:
            A tuple containing:
                - The index of the original strand each of those strands came from.
                - The strands coming out of the stage, dropped strands are removed.
        """

        # DROPOUT - every stage loses whole strands first, so we don't waste time adding errors to strands we are going
        #   to drop anyways. Aging is the odd one out, it works out its own dropout from the years (see _age_strands)
        if config.stage != Stage.AGING:
            indices, sequences = self._apply_dropout_errors(indices, sequences, config, config.dropout_rate)

        # Distributions are advanced only, so simple mode always uses a fixed number of copies and reads. Advanced mode
        #   swaps in the config's own distribution in synthesis and sequencing below.
        distribution: CoverageDistribution = CoverageDistribution.FIXED

        # The real meat and potatoes - apply stage's own thing to the strand before we error on it
        match config.stage:
            # SYNTHESIS - make the copies first, so each one gets its own errors below (like real life)
            case Stage.SYNTHESIS:
                synthesis: SynthesisConfig = cast(SynthesisConfig, config)
                if config.advanced_mode:
                    distribution = synthesis.copy_distribution

                indices, sequences = self._copy_strands(indices, sequences, synthesis, distribution)

            # AGING - Advanced mode: each strand survives with 0.5^(years / half life), on top of the normal dropout 
            #   rate. Simple mode: no half life, so the dropout rate is per year instead (hence why we skip before)
            case Stage.AGING:
                indices, sequences = self._age_strands(indices, sequences, cast(AgingConfig, config))

            # PCR - errors are added to the new copies as they are made, so there is nothing left to do after
            case Stage.PCR:
                indices, sequences = self._amplify_strands(indices, sequences, cast(PCRConfig, config))
                # Return early we are done PCR, which adds its own errors only to strands it amplifies 
                return indices, sequences

            # Sequencing - make the reads first, so each one gets its own errors below (like real life)
            case Stage.SEQUENCING:
                sequencing: SequencingConfig = cast(SequencingConfig, config)
                if config.advanced_mode:
                    distribution = sequencing.read_distribution

                indices, sequences = self._read_strands(indices, sequences, sequencing, distribution)
            

        # The index is carried alongside the strand instead of using where it sits in the list, since earlier stages
        #   can drop or copy strands - so the spot in the list no longer tells us which original strand it was.
        output_indices: list[int] = []
        output: list[str] = []
        for index, sequence in zip(indices, sequences):
            # Now we can finally apply our errors to the strands
            sequence = self._apply_errors(index, sequence, config)

            # Strand was cut or deleted down to nothing, so there is no strand to keep
            if not sequence:
                continue

            # Strand made it through the stage, so keep it (along with the original index for the next stage)
            output_indices.append(index)
            output.append(sequence)

        return output_indices, output


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE SIMPLE METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _pick_random_base(self, rng: np.random.Generator, exclude: str | None = None) -> str:
        """
        Picks a uniformly random base. Simple mode only, advanced will use the conditional tables instead.

        Args:
            rng: The random generator of the stage we are in.
            exclude: A base that can't be picked (used for substitutions so a base can't become itself).

        Returns:
            The picked base.
        """
        # Only the bases we are allowed to pick from
        choices: list[Base] = [base for base in DNA_BASES if base != exclude]

        # Pick a random base from the choices
        pick: Base = choices[rng.integers(len(choices))]

        return pick.value


    def _apply_simple_errors(self, index: int, sequence: str, config: ErrorConfig) -> str:
        """
        Applies substitutions, insertions, and deletions to one strand, with one flat rate for every base. We only
        apply one type of error per base, so if a base is substituted it can't also be deleted or have an insertion
        after it. Hard to tell if this is how it work in real life

        Args:
            index: The index of the original strand this strand came from, for the records.
            sequence: The strand to add errors to.
            config: The config of the stage.

        Returns:
            The strand after the errors.
        """

        rng: np.random.Generator = self.generators[config.stage]
        record: ErrorRecord = self.error_record

        # The rates we use for the stage, straight from the config
        substitution_rate: float = config.substitution_rate
        insertion_rate: float = config.insertion_rate
        deletion_rate: float = config.deletion_rate

        # AGING - the strands sit there for years, so every rate is per year and we need the chance over all of them
        if config.stage == Stage.AGING:
            years: float = cast(AgingConfig, config).years

            substitution_rate = self._get_aged_rate(substitution_rate, years)
            insertion_rate = self._get_aged_rate(insertion_rate, years)
            deletion_rate = self._get_aged_rate(deletion_rate, years)

            # Each rate is aged on its own, so together they can go past a 100% chance of an error. Only one error can
            #   happen per base, so scale them back down until they sum to 1 (keeps them in the same proportion). This
            #   will basically never happen unless user sets like 10000000 years.
            total: float = substitution_rate + insertion_rate + deletion_rate
            if total > 1:
                substitution_rate /= total
                insertion_rate /= total
                deletion_rate /= total

        # Draw all the rolls for the strand at once, much faster than one call per base. Basically we roll a number
        #   between 0 and 1 for each base, and if it is below the rate we apply that error.
        rolls: np.ndarray = rng.random(len(sequence))

        # To use the same roll we can think of each rate existing between 0 and 1, where the first part is sub
        #   next part is ins, and the last part is del. So we can just check if the roll is below the sum of the 
        #   rates we can see if we have that type of error. Shown below:
        #
        #   sub    sub+ins  sub+ins+del                              
        #|--sub--|---ins---|----del----|---------- no error ----------|
        substitution_cutoff: float = substitution_rate
        insertion_cutoff: float = insertion_rate + substitution_cutoff
        deletion_cutoff: float = deletion_rate + insertion_cutoff

        # To skip a lot of work we are only going to check the bases that got an error. Basically anything that is
        #   below the sum of the three rates is an error, and we can check which type it is by seeing which part
        #   of the range it is in.
        error_positions: np.ndarray = np.flatnonzero(rolls < deletion_cutoff)

        # Go backwards through the strand, so an insertion or deletion only shifts the bases we have already done
        #   and every position we haven't gotten to yet is still where it was
        for error_position in error_positions[::-1]:
            # Numpy gives us its own int type, so make it a normal int before it goes into the records
            position: int = int(error_position)
            base: str = sequence[position]

            # SUBSTITUTION - simply add a random base that isn't current one
            if rolls[position] < substitution_cutoff:
                new_base: str = self._pick_random_base(rng, exclude=base)
                sequence = sequence[:position] + new_base + sequence[position + 1:]
                record.record_error(index, position, config.stage, ErrorType.SUBSTITUTION, sequence)

            # INSERTION - add a random base after the current one, so it is at position + 1
            elif rolls[position] < insertion_cutoff:
                new_base: str = self._pick_random_base(rng)
                sequence = sequence[:position + 1] + new_base + sequence[position + 1:]
                record.record_error(index, position, config.stage, ErrorType.INSERTION, sequence)

            # DELETION - remove the base at the current position
            elif rolls[position] < deletion_cutoff:
                sequence = sequence[:position] + sequence[position + 1:]
                record.record_error(index, position, config.stage, ErrorType.DELETION, sequence)

        return sequence


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE ADVANCED METHODS                                               #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def _pick_weighted_base(rng: np.random.Generator, weights: dict[Base, float]) -> str:
        """
        Picks a base from a conditional table, e.g. given an insertion happens which base is inserted.

        Args:
            rng: The random generator of the stage we are in.
            weights: The chance of each base being picked, sums to 1 (checked by the config).

        Returns:
            The picked base.
        """
        # Get all our bases
        bases: list[Base] = list(weights.keys())
        # Get all the weights in the same order, so we can use them as probabilities for numpy's choice
        probabilities: np.ndarray = np.array(list(weights.values()))

        # The config only checks the sum is close to 1, numpy is more picky - so make it exact
        probabilities = probabilities / probabilities.sum()

        # And pick which base
        pick: Base = bases[rng.choice(len(bases), p=probabilities)]

        return pick.value


    @staticmethod
    def _draw_length(rng: np.random.Generator, 
                     min_length: int, 
                     length_probability: float, 
                     max_length: int | None) -> int:
        """
        Draws how long an insertion or deletion is. Starts at the min length and is geometric after it, so
        P(length = min + j) = (1 - p)^j * p, then capped at the max length.

        Args:
            rng: The random generator of the stage we are in.
            min_length: The shortest the error can be.
            length_probability: The p of the geometric part, 1 means it is always the min length.
            max_length: The longest the error can be, None means no cap.

        Returns:
            The length of the insertion or deletion.
        """
        # Numpy's geometric starts at 1 (number of tries until the first success), ours starts at 0 - so take 1 off
        length: int = min_length + int(rng.geometric(length_probability)) - 1

        # Cap everything to make to ensure it stays in users range if they set one, None means no cap
        if max_length is not None and length > max_length:
            length = max_length

        return length


    @staticmethod
    def _get_run_lengths(sequence: str) -> np.ndarray:
        """
        Gets how long the run is at each base, so counts up as the run goes on, e.g. "AACCCG" gives [1, 2, 1, 2, 3, 1].

        Args:
            sequence: The strand to look at.

        Returns:
            An array with the run length so far for every base in the strand.
        """
        run_lengths: list[int] = []

        # How long the run we are currently in is
        run_length: int = 0
        for position in range(len(sequence)):
            # Same base as the one before it, so the run keeps going - otherwise a new run starts at this base
            if position > 0 and sequence[position] == sequence[position - 1]:
                run_length += 1
            else:
                run_length = 1

            run_lengths.append(run_length)

        # Return as an array, so it can be used to look up the multiplier for every base at once
        return np.array(run_lengths)


    @staticmethod
    def _get_homopolymer_multiplier(run_length: int, multipliers: dict[int, float]) -> float:
        """
        Gets the error multiplier for a base sitting in a run of the given length.

        Args:
            run_length: The length of the run up to the base.
            multipliers: The multiplier per run length for the type of error.

        Returns:
            The multiplier of the longest listed run that is not longer than ours, so the longest run listed is used
            for anything past it. 1 if every listed run is longer than ours.
        """
        # Assume we have no homopolyermer, so the multiplier is 1.0 unless we find a longer run
        multiplier: float = 1.0
        closest_length: int = 0

        # Choose the longest listed run that is not longer than ours, so longest run is anything past it
        for listed_length, listed_multiplier in multipliers.items():
            if closest_length < listed_length <= run_length:
                closest_length = listed_length
                multiplier = listed_multiplier

        return multiplier


    @staticmethod
    def _get_homopolymer_multipliers(run_lengths: np.ndarray,
                                     min_length: int,
                                     multipliers: dict[int, float]) -> np.ndarray:
        """
        Gets the error multiplier for every base of a strand from the run length up to it.

        Args:
            run_lengths: The run length up to every base, from _get_run_lengths.
            min_length: The shortest run that counts as a homopolymer, anything shorter is not scaled.
            multipliers: The multiplier per run length for the type of error.

        Returns:
            An array with the multiplier for every base in the strand, 1 for any base not far enough into a run.
        """
        # Only a handful of run lengths show up in a strand, so work out the multiplier once per length into a small
        #   table, then every base just looks up its own length in it. Anything below the min length stays at 1.
        longest_run: int = int(run_lengths.max())
        by_length: np.ndarray = np.ones(longest_run + 1)
        for length in range(min_length, longest_run + 1):
            by_length[length] = ErrorSimulator._get_homopolymer_multiplier(length, multipliers)

        return by_length[run_lengths]


    @staticmethod
    def _get_gc_multipliers(sequence: str, gc_window: GCWindow) -> np.ndarray:
        """
        Gets the GC error multiplier for every base. Each base looks at a window centered on it (clipped at the ends
        of the strand), and the GC fraction of that window decides the multiplier (see GCWindow).

        Args:
            sequence: The strand to look at.
            gc_window: The window settings and bounds.

        Returns:
            An array with the multiplier for every base in the strand.
        """
        # Running total of G and C up to each base, so the GC in any window is just one subtraction instead of counting
        #   the whole window again for every base. e.g. "GACG" -> [0, 1, 1, 2, 3]. Makes things much faster.
        gc_totals: list[int] = [0]
        for letter in sequence:
            if letter == Base.G or letter == Base.C:
                gc_totals.append(gc_totals[-1] + 1)
            else:
                gc_totals.append(gc_totals[-1])

        # Want to center on the base - so half of either side
        half_window: int = gc_window.window_size // 2

        # Get the GC fraction of the window around every base
        gc_fractions: list[float] = []
        for position in range(len(sequence)):
            # Center the window on the base, and clip it so it doesn't run off either end of the strand
            start: int = max(0, position - half_window)
            end: int = min(len(sequence), position - half_window + gc_window.window_size)

            # GC in the window is the total at the end minus the total at the start, over how many bases are in it
            gc_in_window: int = gc_totals[end] - gc_totals[start]
            gc_fractions.append(gc_in_window / (end - start))

        # Then let the window work out the multiplier for every base at once (numpy does the whole array in one go)
        multipliers: np.ndarray = np.array(gc_window.multiplier(np.array(gc_fractions)))

        return multipliers


    def _get_advanced_error_rates(self, 
                                  sequence: str, 
                                  config: ErrorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Works out the chance of each error at every base of a strand. For each base it is:
            rate for that base * homopolymer multiplier * GC multiplier * position multiplier
        With the defaults every multiplier is 1, so it is just the rate for that base. Basically what we are doing here
        is making a list of rates for every base in the strand, a table of rates instead of figure the rate out on the
        fly for every base.

        Args:
            sequence: The strand to get the rates for.
            config: The config of the stage.

        Returns:
            A tuple containing:
                - The substitution rate at every base.
                - The insertion rate right after every base.
                - The deletion rate at every base.
        """

        # PER BASE - start with the rate the config gives each base (missing bases use the overall rate). There are only
        #   four bases, so get their rates once here instead of asking the config again for every base in the strand 
        #   (way faster). Base is a StrEnum, so the letters of the strand work as keys.
        base_rates: dict[str, tuple[float, float, float]] = {}
        for base in DNA_BASES:
            base_rates[base] = config.get_base_rates(base)

        # Now for every base in the strand, get its rates from the table we just made
        substitution_list: list[float] = []
        insertion_list: list[float] = []
        deletion_list: list[float] = []
        for letter in sequence:
            rates_for_base: tuple[float, float, float] = base_rates[letter]
            substitution_list.append(rates_for_base[0])
            insertion_list.append(rates_for_base[1])
            deletion_list.append(rates_for_base[2])

        # Convert to np arrays so we can do math on them all at once, rather than looping through every base
        substitution_rates: np.ndarray = np.array(substitution_list)
        insertion_rates: np.ndarray = np.array(insertion_list)
        deletion_rates: np.ndarray = np.array(deletion_list)

        # DEAMINATION - aging only, it is just more substitutions on top of the normal rate. The rate is per year, and a
        #   base only escapes it by making it through every year, so chance it happens at all is 1 - (1 - rate)^years.
        if config.stage == Stage.AGING:
            aging: AgingConfig = cast(AgingConfig, config)
            deamination_chance: float = self._get_aged_rate(aging.deamination_rate, aging.years)

            # Now fix sub rates to include the deamination chance. Other two are left alone
            substitution_rates = substitution_rates + deamination_chance

        # HOMOPOLYMERS - the run length counts up along the run, so only the bases far enough into a run get their
        #   rates scaled (and the further in, the more they are scaled). Each error has its own table.
        if config.homopolymer_check:
            run_lengths: np.ndarray = self._get_run_lengths(sequence)

            # Now apply the multiplier FOR EACH ERROR TYPE. Typically influences insertion and deletion more, but 
            #   just going to keep it constant all the way through
            substitution_rates *= self._get_homopolymer_multipliers(
                run_lengths=run_lengths, 
                min_length=config.homopolymer_min_length, 
                multipliers=config.homopolymer_substitution_multiplier
            )
            insertion_rates *= self._get_homopolymer_multipliers(
                run_lengths=run_lengths, 
                min_length=config.homopolymer_min_length, 
                multipliers=config.homopolymer_insertion_multiplier
            )
            deletion_rates *= self._get_homopolymer_multipliers(
                run_lengths=run_lengths, 
                min_length=config.homopolymer_min_length, 
                multipliers=config.homopolymer_deletion_multiplier
            )

        # GC CONTENT - same multiplier for all three errors
        if config.gc_content_check:
            gc_multipliers: np.ndarray = self._get_gc_multipliers(sequence, config.gc_window)

            substitution_rates *= gc_multipliers
            insertion_rates *= gc_multipliers
            deletion_rates *= gc_multipliers

        # POSITION BIAS - goes linearly from 1 at the start of the strand to 1 + slope at the end. Love you np linespace
        if config.position_slope > 0:
            position_multipliers: np.ndarray = np.linspace(1.0, 1.0 + config.position_slope, len(sequence))

            substitution_rates *= position_multipliers
            insertion_rates *= position_multipliers
            deletion_rates *= position_multipliers

        # The multipliers can push a base past a 100% chance of an error. Only one error can happen per base, so we
        #   scale the three rates of that base back down until they sum to 1 (keeps them in the same proportion).
        totals: np.ndarray = substitution_rates + insertion_rates + deletion_rates
        scales: np.ndarray = np.maximum(totals, 1.0)

        return substitution_rates / scales, insertion_rates / scales, deletion_rates / scales
    

    def _apply_truncation_errors(self, rng: np.random.Generator, index: int, sequence: str, config: ErrorConfig) -> str:
        """
        Applies truncation to one strand, cutting it down to a random length. The cut is geometric, so the chance of
        the strand surviving intact is (1 - rate)^length.

        Args:
            rng: The random generator of the stage we are in.
            index: The index of the original strand this strand came from, for the records.
            sequence: The strand to add errors to.
            config: The config of the stage.

        Returns:
            The strand after the truncation. Can be empty if the strand was cut down to nothing
        
        Raises:
            SimulatorError: If the stage has no truncation rate, so it can't apply truncation
        """
        if config.truncation_rate <= 0:
            raise SimulationError(
                message=f"Stage {config.stage} has no truncation rate, cannot apply truncation"
            )

        # Same draw as an insertion/deletion length, just starting at 0 with no cap
        kept_length: int = self._draw_length(rng, 0, config.truncation_rate, None)

        # The cut landed inside the strand (past the end means the strand made it through intact)
        if kept_length < len(sequence):
            sequence = sequence[:kept_length]
            self.error_record.record_error(index, kept_length, config.stage, ErrorType.TRUNCATION, sequence)

        return sequence


    def _apply_advanced_errors(self, index: int, sequence: str, config: ErrorConfig) -> str:
        """
        Applies truncation, then substitutions, insertions, and deletions to one strand using every advanced setting.
        Same idea as simple mode (one roll per base, cut up into a range per error), but every base has its own rates
        and insertions/deletions can be longer than one base.

        Args:
            index: The index of the original strand this strand came from, for the records.
            sequence: The strand to add errors to.
            config: The config of the stage.

        Returns:
            The strand after the errors. Can be empty if the strand was cut or deleted down to nothing.
        """

        rng: np.random.Generator = self.generators[config.stage]
        record: ErrorRecord = self.error_record

        # TRUNCATION - every base has the same chance of being where the strand is cut, so the number of bases until
        #   the cut is geometric. This gives us P(strand intact) = (1 - rate)^length. Everything past the cut is gone.
        self._apply_truncation_errors(rng, index, sequence, config)

        # Nothing left to add errors to if truncation shot and killed it (found dead in the Bronx)
        if not sequence:
            return sequence

        # Rates are worked out on the strand as it is now, before any of the errors below change it
        rates: tuple[np.ndarray, np.ndarray, np.ndarray] = self._get_advanced_error_rates(sequence, config)

        # Same as simple mode, each rate is added onto the last so the same roll can be used for all three
        substitution_rates: np.ndarray = rates[0]
        insertion_rates: np.ndarray = rates[1] + substitution_rates
        deletion_rates: np.ndarray = rates[2] + insertion_rates

        # I love gambling
        rolls: np.ndarray = rng.random(len(sequence))

        # To skip a lot of work we are only going to check the bases that got an error
        error_positions: np.ndarray = np.flatnonzero(rolls < deletion_rates)

        # Go backwards through the errors, so an insertion or deletion only shifts the bases we have already done
        for error_position in error_positions[::-1]:
            position: int = int(error_position)
            base: str = sequence[position]

            # SUBSTITUTION - what the base becomes comes from its row in the conditional table
            if rolls[position] < substitution_rates[position]:
                new_base: str = self._pick_weighted_base(rng, config.substitution_conditional[Base(base)])
                sequence = sequence[:position] + new_base + sequence[position + 1:]
                record.record_error(index, position, config.stage, ErrorType.SUBSTITUTION, sequence)

            # INSERTION - one or more bases go in right after this one
            elif rolls[position] < insertion_rates[position]:
                # Length can be longer than one - we draw it from geometric start at min length
                insertion_length: int = self._draw_length(
                    rng=rng, 
                    min_length=config.insertion_min_length, 
                    length_probability=config.insertion_length_probability, 
                    max_length=config.insertion_max_length
                )

                # Stutter (where bases are repeated) big problem for insertions
                inserted: str = ""
                previous_base: str = base
                for _ in range(insertion_length):
                    # Stutter - just repeat the base before it, otherwise pick one from the conditional table
                    inserted_base: str = previous_base
                    if rng.random() >= config.insertion_repeat_previous:
                        inserted_base = self._pick_weighted_base(rng, config.insertion_conditional)

                    inserted += inserted_base
                    previous_base = inserted_base

                # Insert the new bases after the current one
                sequence = sequence[:position + 1] + inserted + sequence[position + 1:]
                record.record_error(index, position, config.stage, ErrorType.INSERTION, sequence)

            # DELETION - this base and however many after it are removed, slicing stops at the end of the strand for us
            else:
                # Like insertions, we can be longer than 1
                deletion_length: int = self._draw_length(
                    rng=rng, 
                    min_length=config.deletion_min_length, 
                    length_probability=config.deletion_length_probability, 
                    max_length=config.deletion_max_length
                )

                # Remove em
                sequence = sequence[:position] + sequence[position + deletion_length:]
                record.record_error(index, position, config.stage, ErrorType.DELETION, sequence)

        return sequence


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def simulate(self, sequences: list[str]) -> list[str]:
        """
        Runs the strands through the four stages of the storage channel, in order: synthesis, aging, PCR, sequencing.

        Args:
            sequences: The strands going into the first stage.

        Returns:
            The strands coming out of the last stage, dropped strands are removed.
        """
        # Create a copy so we don't change the original list so we user can run multiple simulations on same strands.
        sequences = sequences.copy()

        # Fresh record every simulation, so running twice doesn't add the errors of both runs together
        self.error_record = ErrorRecord()

        # The index of every strand in the original list. Goes through every stage alongside the strands, so an error
        #   is always recorded against the original strand no matter how many strands were dropped or copied before it.
        indices: list[int] = []
        for index in range(len(sequences)):
            indices.append(index)

        for _, config in self.stages:
            # Apply our errors
            indices, sequences = self._apply_stage(indices, sequences, config)

            # Shuffle everything up after each stage. Shuffle one order and apply it to both, so the indices stay lock
            #   step with the strands and every error is still recorded against the right original strand
            order: np.ndarray = self.generators[config.stage].permutation(len(sequences))
            indices = [indices[position] for position in order]
            sequences = [sequences[position] for position in order]

        return sequences