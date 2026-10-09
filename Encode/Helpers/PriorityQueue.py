"""
PriorityQueue.py

A priority queue implementation using a min-heap, used for building Huffman tree. Stores tuples of 
(frequency, character) to facilitate the Huffman encoding process.
"""


class PriorityQueue:
    def __init__(self) -> None:
        """
        Initializes an empty priority queue.
        """
        # Since we need the character and its frequency, easier to store them as a tuple (frequency, character) now
        self.minheap: list[tuple[int, str]] = []
        self.size: int = 0


    def parent(self, index: int) -> int:
        """
        Gets the index of the parrent of the element at the given index in the min-heap.

        Args:
            index: The index of the element in the min-heap.

        Returns:
            Index of the parent of the element at the given index.
        """
        return (index - 1) // 2


    def left_child(self, index: int) -> int:
        """
        Gets the index of the left child of the element at the given index in the min-heap.

        Args:
            index: The index of the element in the min-heap.

        Returns:
            Index of the left child of the element at the given index.
        """
        return 2 * index + 1


    def right_child(self, index: int) -> int:
        """
        Gets the index of the right child of the element at the given index in the min-heap.

        Args:
            index: The index of the element in the min-heap.

        Returns:
            Index of the right child of the element at the given index.
        """
        return 2 * index + 2


    def heapify(self, index: int) -> None:
        """
        Maintains the min-heap property for the subtree rooted at the given index.

        Args:
            index: The index of the root of the subtree to heapify.

        Returns:
            None
        """

        # Get our left and right child indices
        left: int = self.left_child(index)
        right: int = self.right_child(index)

        smallest: int = index

        # If left is smaller, update
        if left < self.size and self.minheap[left][0] < self.minheap[smallest][0]:
            smallest = left
        # If right is smaller, update
        if right < self.size and self.minheap[right][0] < self.minheap[smallest][0]:
            smallest = right

        # If smallest is not the current index, swap and heapify
        if smallest != index:
            self.minheap[index], self.minheap[smallest] = self.minheap[smallest], self.minheap[index]
            self.heapify(smallest)


    def build_queue(self, frequencies: dict[str, int]) -> None:
        """
        Builds a priority queue from a dictionary of character frequencies.

        Args:
            frequencies: A dictionary mapping characters to their frequencies.

        Returns:
            None
        """

        # Add all the frequency-character pairs to the heap
        for char, freq in frequencies.items():
            self.minheap.append((freq, char))
            self.size += 1

        # Start from the last non-leaf node and heapify each node
        for i in range(self.size // 2 - 1, -1, -1):
            self.heapify(i)


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


    def top(self) -> tuple[int, str]:
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
        
        return self.minheap[0]


    def dequeue(self) -> tuple[int, str]:
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

        # Our top
        top_element: tuple[int, str] = self.top()

        # Move the last element to our root, removing it from the end
        self.minheap[0] = self.minheap[self.size - 1]
        self.size -= 1
        self.minheap.pop()

        # Heapify the root to maintain the min-heap property
        self.heapify(0)
        return top_element


    def enqueue(self, element: tuple[int, str]) -> None:
        """
        Adds a new element to the priority queue.
        
        Args:
            element: The element to be added to the priority queue.

        Returns:
            None
        """
        # Add the new element to the end of the heap
        self.minheap.append(element)
        self.size += 1

        # Bubble up the new element to maintain the min-heap property
        index: int = self.size - 1
        parent: int = self.parent(index)

        # While not at root and parent is greater, swap
        while index != 0 and self.minheap[parent][0] > self.minheap[index][0]:
            # Swap with parent
            self.minheap[index], self.minheap[parent] = self.minheap[parent], self.minheap[index]

            # Update index and parent for the next iteration
            index = self.parent(index)
            parent = self.parent(index)