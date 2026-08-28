class PriorityQueue:
    def __init__(self) -> None:
        # Since we need the character and its frequency, easier to store them as a tuple (frequency, character) now
        self.minheap: list[tuple[int, str]] = []
        self.size: int = 0


    def parent(self, index: int) -> int:
        return (index - 1) // 2


    def left_child(self, index: int) -> int:
        return 2 * index + 1


    def right_child(self, index: int) -> int:
        return 2 * index + 2


    def heapify(self, index: int) -> None:
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
        # Add all the frequency-character pairs to the heap
        for char, freq in frequencies.items():
            self.minheap.append((freq, char))
            self.size += 1

        # Start from the last non-leaf node and heapify each node
        for i in range(self.size // 2 - 1, -1, -1):
            self.heapify(i)


    def is_empty(self) -> bool:
        return self.size == 0


    def top(self) -> tuple:
        # Can't return the top element if the heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")
        
        return self.minheap[0]


    def dequeue(self) -> tuple:
        # Can't dequeue if heap is empty
        if self.is_empty():
            raise IndexError("Priority queue is empty")

        # Our top
        top_element: tuple = self.top()

        # Move the last element to our root, removing it from the end
        self.minheap[0] = self.minheap[self.size - 1]
        self.size -= 1
        self.minheap.pop()

        # Heapify the root to maintain the min-heap property
        self.heapify(0)
        return top_element


    def enqueue(self, element: tuple) -> None:
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