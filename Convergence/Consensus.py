"""
Module finds the best guess of each original strand from a cluster of its noisy reads, so the decoder gets one clean
strand per index instead of a pile of noisy copies. Every consensus algorithm takes in the reads of one cluster and give
back a single strand, so it can be swapped out for any other, and paired with any clusterer. Like clustering, consensus 
knows nothing about encoding - it only ever sees strings of bases.

Based on partial order alignment (POA) from the spoa library which is a C++ verison to speed things up. Basically a 
its based on these papers - 2002 is origin and 2017 is what made it fast:
    (2002): Lee, C. et al. "Multiple sequence alignment using partial order graphs"
    (2017): Vaser, Sović, Nagarajan, Šikić "Fast and accurate de novo genome assembly from long uncorrected reads".
"""

from spoa import poa

from Convergence.Clustering import Clustering
from Helpers.Errors import ConsensusError



class Consensus:
    """
    Holds what every consensus algorithm shares: running it on every cluster, and tying it together with a clusterer.
    Each algorithm only has to say how one cluster is turned into one strand.
    """

    def __init__(self, alignment_type: int = 1):
        """
        Initializes the POA consensus algorithm.

        Args:
            alignment_type: The type of alignment to use (0 = local, 1 = global, 2 = semi-global).
        """

        # spoa's alignment types are 0 = local (Smith-Waterman), 1 = global (Needleman-Wunsch), 2 = semi-global. spoa 
        #   has defaults and they are explained in the C++ help function, but we aren't really going to change them, so
        #   just keep it to the obvious.
        self.alignment_type: int = alignment_type

        if alignment_type not in {0, 1, 2}:
            raise ConsensusError(
                message=f"Invalid alignment type: {alignment_type}. Must be 0 (local), 1 (global), or 2 (semi-global)."
            )


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   STATIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    @staticmethod
    def _group_reads(reads: list[str], clusters: list[list[int]]) -> list[list[str]]:
        """
        Turns clusters of read indices into clusters of the reads themselves, what the consensus algorithms take in.

        Args:
            reads: The reads that were clustered (the same list given to cluster).
            clusters: The clusters as lists of read indices, from cluster.

        Returns:
            The clusters as lists of reads, lock step with clusters.
        """
        groups: list[list[str]] = []
        # Go through all the clusters
        for cluster in clusters:
            # One group per cluster, holding every read in it
            group: list[str] = []
            for index in cluster:
                group.append(reads[index])

            groups.append(group)

        return groups


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def consensus(self, reads: list[str]) -> str:
        """
        Finds the best guess of the original strand by building a partial order graph of the reads, and taking the
        heaviest path through it.

        Args:
            reads: The reads of one cluster.

        Returns:
            The best guess of the original strand, empty if there are no reads.
        """
        # Nothing to build from
        if not reads:
            return ""

        # A single read has nothing to agree or disagree with, so it is its own best guess
        if len(reads) == 1:
            return reads[0]

        consensus: str
        # The library barely gives any info on the the parameters, so just keeping it to whats obivious.
        #   sequences = reads to align
        #   algorithm = alignment type (0 = local (Smith-Waterman), 1 = global (Needleman-Wunsch), 2 = semi-global)
        consensus, _ = poa(
            sequences=reads,                    # The reads to align against the graph
            algorithm=self.alignment_type,      # Align each read end to end against the graph
            genmsa=False                        # Don't need the full alignment, only the consensus
        )

        return consensus


    def consensus_all(self, groups: list[list[str]]) -> list[str]:
        """
        Finds the consensus of every cluster.

        Args:
            groups: The clusters as lists of reads, from Clustering.group_reads.

        Returns:
            One strand per cluster, lock step with groups.
        """
        strands: list[str] = []
        for reads in groups:
            strands.append(self.consensus(reads))

        return strands


    def cluster_and_consensus(self, reads: list[str], clusterer: Clustering) -> list[str]:
        """
        Ties clustering and consensus together. Clusters the reads, then finds the consensus of every cluster, so a
        pile of noisy reads goes in and one strand per cluster comes out - ready for the decoder.

        Args:
            reads: The reads to cluster, primers and all.
            clusterer: The clusterer to group the reads with.

        Returns:
            One strand per cluster, primers and all.
        """
        # Group the reads by the strand they came from
        clusters: list[list[int]] = clusterer.cluster(reads)

        # Clusters are read indices, so swap them for the reads themselves. These still have their primers - the
        #   primers were only trimmed for clustering, the decoder wants the full strand
        groups: list[list[str]] = self._group_reads(reads, clusters)

        return self.consensus_all(groups)