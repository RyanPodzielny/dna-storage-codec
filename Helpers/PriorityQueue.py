"""
A priority queue implementation using a min-heap, used for building Huffman tree. Stores tuples of 
(frequency, character) to facilitate the Huffman encoding process.
"""

import heapq
from typing import Any


class PriorityQueue:
    """
    A priority queue implementation using a min-heap.
    """

    def __init__(self) -> None:
        """
        Initializes an empty priority queue.
        """

        # Min heap is a list of tuples = (frequency, order, character), where 'order' is used to maintain the
        #   order of insertion for elements with the same frequency - basically a tie breaker. Heapq also makes this 
        #   easier to implement and a lot faster (especially for Pgrams)
        self.minheap: list[tuple[int, int, Any]] = []
        self.size: int = 0

        # Used to maintain insertion order for elements with same frequency
        self._order: int = 0


    #--------------------------------------------------------------------------#
    #                                                                          #
    #   PUBLIC HELPER METHODS                                                  #
    #                                                                          #
    #--------------------------------------------------------------------------#


    def build_queue(self, frequencies: dict[Any, int]) -> None:
        """
        Builds a priority queue from a dictionary of character frequencies.

        Args:
            frequencies: A dictionary mapping characters to their frequencies.

        Returns:
            None
        """

        self.minheap.clear()
        self.size = 0
        self._order = 0

        for char, freq in frequencies.items():
            self.enqueue((freq, char))


    def is_empty(self) -> bool:
        """
        Checks if the priority queue is empty.

        Returns:
            True if the priority queue is empty, False otherwise.
        """
        return self.size == 0


    def get_size(self) -> int:
        """
        Returns the number of elements in the priority queue.

        Returns:
            The number of elements in the priority queue.
        """
        return self.size


    def top(self) -> tuple[int, Any]:
        """
        Returns the top element of the priority queue without removing it.

        Returns:
            The top element of the priority queue.

        Raises:
            IndexError: If the priority queue is empty.
        """
        # Can't return the top element if the heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        entry: tuple[int, int, Any] = self.minheap[0]
        return entry[0], entry[2]


    def dequeue(self) -> tuple[int, Any]:
        """
            Removes and returns the top element of the priority queue.

        Returns:
            The top element of the priority queue.

        Raises:
            IndexError: If the priority queue is empty.
        """
        # Can't dequeue if heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")

        frequency: int
        payload: Any

        frequency, _, payload = heapq.heappop(self.minheap)
        self.size -= 1
        return frequency, payload


    def enqueue(self, element: tuple[int, Any]) -> None:
        """
        Adds a new element to the priority queue.
        
        Args:
            element: The element to be added to the priority queue.

        Returns:
            None
        """
        frequency: int
        payload: Any
        frequency, payload = element

        # Add the new element to the heap with the current order as a tie breaker
        heapq.heappush(self.minheap, (frequency, self._order, payload))
        self._order += 1
        self.size += 1