"""
Module groups noisy reads by the strand they came from, so the consensus can be found for each strand before it is
passed along to the decoder. The clusterer takes in a list of reads and gives back the clusters as lists of read
indices, so the consensus algorithm can be swapped out freely. Clustering knows nothing about encoding. The only thing 
it is given about how the strands were made is the primers, and those are given by the user. So it works the same on 
reads from any encoder, or from real sequencing data.

Basically everything is based on VSEARCH's cluster_fast:
    Rognes, T. et al. "VSEARCH: a versatile open source tool for metagenomics" PeerJ
"""


import math

# edlib is a C++ library for edit distance, way faster than anything I could write
import edlib

# Same as the other configs - pydantic does all the validation for us, even after a user changes a field
from pydantic import BaseModel, ConfigDict, PositiveInt

from Encode.OligoFramer import Primer
from Simulate.ErrorConfig import Probability

#------------------------------------------------------------------------------#
#                                                                              #
#   CONFIGS                                                                    #
#                                                                              #
#------------------------------------------------------------------------------#


class ClusteringConfig(BaseModel):
    """
    The settings for clustering: the primers to trim off each read, and the seed-then-align settings. Same settings as
    VSEARCH's cluster_fast.
    """

    # No extra fields and validate on assignment, same as every other config
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Primers can mess up how similar each strand looks to one another, so we want to strip them off if possible. Empty
    #   means nothing is trimmed off that end
    start_primer: Primer = ""
    end_primer: Primer = ""

    # We allow for errors to be introduced on the primers themselves, so if not in allowed fraction we trim by length
    primer_error_fraction: Probability = 0.2

    # How a read joins a cluster if its identity to the centroid is at least this. Identity is the matching bases over
    #   the length of the alignment, not counting gaps hanging off either end, so a read that was cut short isn't 
    #   screwed for the missing end. Can't go too low - real encoded strands are a lot more alike than random ones 
    #   (similar indices, repeated content in the file), so it starts merging different strands together. Using 
    identity: Probability = 0.85

    # Length of the k-mers used as seeds. Longer seeds have fewer random matches, but every error destroys more of
    #   them - so the higher the error rate, the shorter this should be. Can't go too short though - encoded strands
    #   reuse the same short pattern. So short seeds fill every bucket and reads never find their own cluster.
    seed_length: PositiveInt = 16

    # How many kmers a read has to share with a centroid before we bother aligning them
    min_seed_hits: PositiveInt = 3

    # Centroids are checked most shared kmers first, and we stop after this many are rejected. With more than one 
    #   accepted, the read joins the one with the highest identity. Not sure why this is exactly done (I think to
    #   prevent too many reads joining the same cluster and not making new one), but as it is in VSEARCH we keep it. 
    #   The default is the same as VSEARCH's, and they work well enough.
    max_rejects: PositiveInt = 32


#------------------------------------------------------------------------------#
#                                                                              #
#   CLUSTERING                                                                 #
#                                                                              #
#------------------------------------------------------------------------------#



class Clustering:
    """
    Greedy centroid-based clustering, following VSEARCH's cluster_fast, an open source alternative to USEARCH's UCLUST. 
    Candidate centroids are filtered and ranked by how many kmers they share with a read (the same word-counting 
    heuristic VSEARCH uses), and only those are aligned. Using short word matches to avoid aligning everything goes back
    to FASTA (Lipman & Pearson 1985). Written in Python (with edlib for the alignment) so nothing else has to be 
    installed. Makes it way slower than the C++ version, but works well enough for the moment. Basically the steps:
        1. Reads are sorted longest first. A read that lost bases (a deletion, or cut short) is shorter, so the full
           length reads come first and become the centroids, which makes for better centroids .
        2. Each read's kmers are looked up in an index of the centroids kmers, and the centroids sharing at least
           min_seed_hits are checked most shared first.
        3. Each one is aligned (edlib) and its identity worked out. At or above the threshold its accepted, otherwise
           rejected. We stop after it is accepted or we hit max_rejects rejected.
        4. The read joins the accepted centroid with the highest identity. If none were accepted it become a new
           centroid, and its kmers are added to the index.
    """


    def __init__(self, config: ClusteringConfig):
        """
        Initializes the clusterer with its config. Configs are validated by pydantic, so there is nothing to check here.

        Args:
            config: The config for the clusterer.
        """
        self.config: ClusteringConfig = config


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE SHARED METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _find_primer(self, primer: str, window: str) -> tuple[int, int] | None:
        """
        Finds where a primer is in a window of a read with edlib, allowing for errors in it. Infix alignment (HW) lets
        the primer sit anywhere in the window for free, so only the errors in the primer itself count.

        Args:
            primer: The primer to look for.
            window: The part of the read the primer should be in.

        Returns:
            The start and end (exclusive) of the primer in the window, or None if it couldn't be found with at most
            the allowed errors.
        """
        # The most allowed errors we can have in a primer (just the fraction of its length)
        max_errors: int = math.ceil(self.config.primer_error_fraction * len(primer))

        # Get the distance and locations of the primer in the window, if there is any
        result: dict = edlib.align(
            query=primer,           # Search for the primer in the window
            target=window,          # Window is just the length of allowed error + primer
            mode="HW",              # HW = Hybrid Wunsch, basically we don't penalize sitting anywhere in the window
            task="locations",       # REturns both the edit distance and positions
            k=max_errors            # Not allow more than the allowed errors, otherwise it stops early
        )

        # If primer can't be found within allowed errors, return as None so we just cut off length instead
        if result["editDistance"] == -1:
            return None

        # edlib gives the end as inclusive, so add one. Might have multiple best spots, first is good enough
        start: int
        end: int
        start, end = result["locations"][0]

        return start, end + 1


    def _trim_primers(self, read: str) -> str:
        """
        Trims the start and end primers off a read. Errors in the primer can shift where it ends (an insertion or
        deletion), so the primer is found with edlib instead of just cutting off its length. Only the start (or end) of
        the read is searched, the length of the primer plus the errors we allow in it.

        Args:
            read: The read to trim.

        Returns:
            The read without its primers, can be empty if the read was shorter than the primers.
        """
        # Where the body (everything but the primers) of the read starts and ends
        body_start: int = 0
        body_end: int = len(read)

        # Start primer - search the start of the read, and cut everything up to where it ends
        if self.config.start_primer:
            # Window length is just the added errors
            window_length: int = len(self.config.start_primer) +\
                  math.ceil(self.config.primer_error_fraction * len(self.config.start_primer))

            # Search the start of the read, and cut everything up to where it ends
            location: tuple[int, int] | None = self._find_primer(self.config.start_primer, read[:window_length])

            # If the primer was found, we can cut up everything till where it ends
            if location:
                body_start = location[1]
            else:
                # If the primer wasn't found, just cut off its length instead
                body_start = len(self.config.start_primer)

        # End primer - search the end of the read, and cut everything from where it starts
        if self.config.end_primer:
            # Again - just add error amount to length. But start can either be at end of start primer or end of the read
            window_length: int = len(self.config.end_primer) +\
                math.ceil(self.config.primer_error_fraction * len(self.config.end_primer))
            window_start: int = max(body_start, len(read) - window_length)

            # Again search but just at the end
            location: tuple[int, int] | None = self._find_primer(self.config.end_primer, read[window_start:])

            # Primer found, new body end
            if location:
                body_end = window_start + location[0]
            else:
                # If the primer wasn't found, just cut off its length instead - same as before
                body_end = len(read) - len(self.config.end_primer)

        # Read was too short to have anything between the primers
        if body_end <= body_start:
            return ""

        return read[body_start:body_end]


    def _trim_all_primers(self, reads: list[str]) -> list[str]:
        """
        Trims the primers off every read.

        Args:
            reads: The reads to trim.

        Returns:
            The reads without their primers, lock step with reads.
        """

        # Nothing to trim, don't bother aligning
        if not self.config.start_primer and not self.config.end_primer:
            return reads.copy()

        # Trim each read (strand)
        new_reads: list[str] = []
        for read in reads:
            new_reads.append(self._trim_primers(read))

        return new_reads


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PRIVATE HELPER METHODS                                                 #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def _get_seeds(self, body: str) -> set[str]:
        """
        Gets every kmer of a read. A set - kmer the read has more than once (e.g. run of the same base) is counted once.
        We basically do this to spread out the error hits, meaning a read with a 5 or 6 errors still has a bunch of
        seeds/kmers left to decide where it belongs.

        Args:
            body: The read, without its primers.

        Returns:
            Every kmer of the read, empty if the read is shorter than the seed length.
        """
        # Get every k-mer of the read
        kmers: set[str] = set()
        for position in range(len(body) - self.config.seed_length + 1):
            kmers.add(body[position:position + self.config.seed_length])

        return kmers


    def _get_candidates(self, seeds: set[str], index: dict[str, list[int]]) -> list[int]:
        """
        Finds the clusters worth aligning against, by how many kmers the read shares with their centroid.

        Args:
            seeds: The kmers of the read.
            index: Every kmer of the centroids, and the clusters whose centroid has it.

        Returns:
            Every cluster sharing at least min_seed_hits kmers, most shared first. All of them, how many get aligned
            against is up to max_accepts and max_rejects.
        """
        # Count how many kmers we share with each centroid. Stored as {cluster: count}, so we can sort by count later
        hits: dict[int, int] = {}
        for seed in seeds:
            for cluster_index in index.get(seed, ()):
                hits[cluster_index] = hits.get(cluster_index, 0) + 1

        # Only keep the clusters that share enough kmers to be worth aligning against. Stored as (count, cluster) so
        #   sorting puts the most shared first, and ties go to the older cluster so it is the same every run
        scores: list[tuple[int, int]] = []
        for cluster_index, count in hits.items():
            # If shared enough kmers, add it to the list of candidates
            if count >= self.config.min_seed_hits:
                scores.append((count, cluster_index))
        scores.sort(reverse=True)

        # Just the clusters
        candidates: list[int] = []
        for _, cluster_index in scores:
            candidates.append(cluster_index)

        return candidates


    @staticmethod
    def _parse_cigar(cigar: str) -> list[tuple[int, str]]:
        """
        Splits an extended CIGAR string from edlib into its operations, e.g. "3=1X2=1D" into [(3, "="), (1, "X"),
        (2, "="), (1, "D")]. = is a match, X a mismatch, I an insertion, D a deletion.

        Args:
            cigar: The extended CIGAR string.

        Returns:
            Every operation as (how many, which).
        """
        operations: list[tuple[int, str]] = []
        count: str = ""

        # Go through every cigar character, building up the count until we hit operation character
        for character in cigar:
            # Digits build up the count, the letter after them is the operation
            if character.isdigit():
                count += character
            # OPeration character hit, so add it to count
            else:
                operations.append((int(count), character))
                count = ""

        return operations


    def _get_identity(self, body: str, centroid: str) -> float:
        """
        Gets the identity of a read to a centroid the same way VSEARCH does by default: the matching bases over the 
        number of columns in the alignment, not counting the gaps hanging off either end. So a read that was cut short 
        only gets scored on the part it has.

        Args:
            body: The read, without its primers.
            centroid: The cluster's centroid, without its primers.

        Returns:
            The identity between 0 and 1, 0 if it is for sure below the threshold (edlib stopped early).
        """
        # The most edits we could have and still be at the threshold. Identity is matches over columns, and there are
        #   at most as many columns as both reads put together. On top of that, the gaps hanging off the ends count as
        #   edits for edlib but not for identity, and there are at least as many of those as the difference in length.
        #   So this is a safe place for edlib to stop early
        max_edits: int = math.floor((1 - self.config.identity) * (len(body) + len(centroid))) + \
            abs(len(body) - len(centroid))

        result: dict = edlib.align(
            query=body,             # The read
            target=centroid,        # The centroid
            mode="NW",              # Global (Needleman-Wunsch), both reads are aligned end to end
            task="path",            # Need the alignment itself (the CIGAR) to count matches and columns
            k=max_edits             # Stop as soon as there are too many edits to ever reach the threshold
        )

        # Too many edits, so it is below the threshold for sure
        if result["editDistance"] == -1:
            return 0.0

        # Get our alignment as a list of operations, e.g. [(3, "="), (1, "X"), (2, "="), (1, "D")] for "3=1X2=1D"
        operations: list[tuple[int, str]] = self._parse_cigar(result["cigar"])

        # Drop the gaps hanging off either end (insertions or deletions at the very start or end of the alignment)
        start: int = 0
        end: int = len(operations)
        while start < end and operations[start][1] in ("I", "D"):
            start += 1
        while end > start and operations[end - 1][1] in ("I", "D"):
            end -= 1

        # Matches over every column that is left. Our columns are the matches, mismatches, and the insertions and 
        #   deletions in the middle of the alignment
        matches: int = 0
        columns: int = 0
        for count, operation in operations[start:end]:
            columns += count

            # We only count matches, not mismatches or gaps, for the our identity
            if operation == "=":
                matches += count

        # No matches means no identity - usually when a read is too short to have any seeds at all
        if columns == 0:
            return 0.0

        # Our identity is how close the read is to the centroid, so the matching bases over the number of columns in 
        #   the alignment. 
        return matches / columns


    def _find_cluster(self,
                      body: str,
                      seeds: set[str],
                      index: dict[str, list[int]],
                      centroids: list[str]) -> int | None:
        """
        Finds the cluster a read belongs to. Candidates are checked most shared kmers first, stopping after it finds one
        or max_rejects is hit, and the read goes to the accepted one with the highest identity.

        Args:
            body: The read, without its primers.
            seeds: The kmers of the read.
            index: Every kmer of the centroids, and the clusters whose centroid has it.
            centroids: The centroid of every cluster.

        Returns:
            The cluster the read belongs to, or None if no centroid was close enough (it should start a new cluster).
        """
        rejects: int = 0

        for cluster_index in self._get_candidates(seeds, index):
            # Get how close the read is to the centroid
            identity: float = self._get_identity(body, centroids[cluster_index])

            # IF we hit our identity standards, we accept it - otherwise its worthless to this cluster
            if identity >= self.config.identity:
                return cluster_index
            else:
                # If rejected too many, stop. Makes it faster, but a lot less accurate. Ideally we'd check every cluster
                #   but that is legit way too slow. They do this in VSEARCH too, so might as well keep it.
                rejects += 1
                if rejects >= self.config.max_rejects:
                    break

        # Couldn't find any cluster close enough, so it should start a new one
        return None


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC METHODS                                                         #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def cluster(self, reads: list[str]) -> list[list[int]]:
        """
        Groups the reads by the strand they came from. Reads go longest first, each one joins the closest accepted 
        centroid, otherwise it becomes a new centroid.

        Args:
            reads: The reads to cluster, primers and all.

        Returns:
            The clusters, each a list of indices into reads with the centroid first, in the order they were made. A
            read too short to have any seeds ends up in a cluster on its own.
        """
        # Trim all the primers so they don't mess up how similar each strand looks
        bodies: list[str] = self._trim_all_primers(reads)
       
        # Longest first, ties keep the order they came in. We want the longest first so they become the centroids, not
        #   the shorter deletion fulled or truncated strands - basically makes for a better centroid/representative.
        order: list[tuple[int, int]] = []
        for read_index, body in enumerate(bodies):
            order.append((len(body), read_index))
        order.sort(reverse=True)

        # Lock step - the centroid of each cluster (trimmed), and the reads in it
        centroids: list[str] = []
        clusters: list[list[int]] = []

        # Every kmer of the centroids, and the clusters whose centroid has it. Basically a way to say x number of seeds
        #   are shared with these clusters, so they are worth aligning against.
        index: dict[str, list[int]] = {}

        for _, read_index in order:
            body: str = bodies[read_index]

            # Get all the kmers of the read
            seeds: set[str] = self._get_seeds(body)

            # Join the closest accepted centroid if there is one 
            cluster_index: int | None = self._find_cluster(body, seeds, index, centroids)
            if cluster_index is not None:
                clusters[cluster_index].append(read_index)
                continue

            # Wasn't close to any of them, so start a new cluster with us as the centroid and add our seeds
            cluster_index = len(clusters)
            centroids.append(body)
            clusters.append([read_index])
            for seed in seeds:
                index.setdefault(seed, []).append(cluster_index)
        
        return clusters