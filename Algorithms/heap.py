from typing import List


class Heap:
    
    def __init__(self, comparator = lambda x,y: x < y):
        """Heap data structure which can handle any type as long as the user tells it how to handle said types

        Args:
            comparator (_type_, optional): Specify if the first input should be above the second input in the heap. Defaults to x < y.
        """
        self.__underlying_array = []
        self.__comparator = comparator

    def __len__(self) -> int:
        return len(self.__underlying_array)

    def empty(self) -> bool:
        return len(self.__underlying_array) == 0

    def push(self, val: any) -> None:
        self.__underlying_array.append(val)
        self.__fix_heap_up(idx = len(self.__underlying_array) - 1)

    def pop(self) -> any:
        if len(self.__underlying_array) == 0:
            raise IndexError("pop from empty heap")
        v = self.__underlying_array[0]
        if len(self.__underlying_array) == 1:
            self.__underlying_array = []
        else:
            last_value = self.__underlying_array[len(self.__underlying_array)-1]
            self.__underlying_array.pop()
            self.__underlying_array[0] = last_value
            self.__fix_heap_down(idx = 0)
        return v
    
    def __fix_heap_up(self, idx: int):
        if idx > 0:
            current_value = self.__underlying_array[idx]
            parent_idx = (idx - 1) // 2
            parent_value = self.__underlying_array[parent_idx]
            if self.__comparator(current_value, parent_value):
                # Current value is less than parent - switch the parent and current values and make a recursive call on the parent index
                self.__swap(idx, parent_idx)
                self.__fix_heap_up(idx=parent_idx)
    
    def __fix_heap_down(self, idx: int):
        current_value = self.__underlying_array[idx]
        left_child_idx = 2*idx + 1
        right_child_idx = 2*(idx + 1)
        if left_child_idx < len(self.__underlying_array) and right_child_idx < len(self.__underlying_array):
            # Both children exist
            left_child_value = self.__underlying_array[left_child_idx]
            right_child_value = self.__underlying_array[right_child_idx]
            if self.__comparator(left_child_value, current_value) or self.__comparator(right_child_value, current_value):
                # At least one child has a value less than the current parent
                min_child_idx = left_child_idx if self.__comparator(left_child_value, right_child_value) else right_child_idx
                self.__swap(idx, min_child_idx)
                # After switching values with the min child and current parent, make a recursive call on the lower child index
                self.__fix_heap_down(idx=min_child_idx)
        elif left_child_idx < len(self.__underlying_array):
            # Only the left child exists
            left_child_value = self.__underlying_array[left_child_idx]
            if self.__comparator(left_child_value, current_value):
                # Then switch the left child value with the current parent value
                self.__swap(idx, left_child_idx)
                # And make a recursive call on the left child
                self.__fix_heap_down(idx=left_child_idx)

    def top(self) -> any:
        if self.empty():
            raise IndexError("top from empty heap")
        return self.__underlying_array[0]

    def heapify(self, values: List) -> None:
        self.__underlying_array = values[:]
        for i in reversed(range(len(self.__underlying_array) // 2)):
            self.__fix_heap_down(i)

    def __swap(self, i, j):
        self.__underlying_array[i], self.__underlying_array[j] = self.__underlying_array[j], self.__underlying_array[i]

    def clear(self) -> None:
        self.__underlying_array = []

    def __str__(self):
        if not self.__underlying_array:
            return "Heap(empty)"

        def draw_tree(index=0, indent="", last=True):
            if index >= len(self.__underlying_array):
                return ""

            output = indent
            if index != 0:
                output += "└── " if last else "├── "
            output += str(self.__underlying_array[index]) + "\n"

            indent += "    " if last else "│   "
            left_index = 2 * index + 1
            right_index = 2 * index + 2
            has_left = left_index < len(self.__underlying_array)
            has_right = right_index < len(self.__underlying_array)

            if has_left or has_right:
                output += draw_tree(left_index, indent, not has_right)
                output += draw_tree(right_index, indent, True)

            return output

        return "Heap:\n" + draw_tree()