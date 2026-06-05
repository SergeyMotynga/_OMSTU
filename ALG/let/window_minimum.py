"Минимум в скользящем окне"

class QueueWithMinimum:
    def __init__(self):
        self.left_stack = []
        self.right_stack = []

    def _add_to_stack(self, stack, element):
        current_minimum = element if not stack else min(element, stack[-1][1])
        stack.append((element, current_minimum))

    def add(self, element):
        self._add_to_stack(self.right_stack, element)

    def _transfer(self):
        while self.right_stack:
            element, _ = self.right_stack.pop()
            self._add_to_stack(self.left_stack, element)

    def remove(self):
        if not self.left_stack:
            self._transfer()

        if self.left_stack:
            self.left_stack.pop()

    def minimum(self):
        minimums = []

        if self.left_stack:
            minimums.append(self.left_stack[-1][1])

        if self.right_stack:
            minimums.append(self.right_stack[-1][1])

        return min(minimums) if minimums else None


def window_minimums(numbers, window_size):
    queue = QueueWithMinimum()
    answer = []

    for index, number in enumerate(numbers):
        queue.add(number)

        if index >= window_size - 1:
            answer.append(queue.minimum())
            queue.remove()

    return answer


numbers = [11, 18, 14, 3, 5, 17, 4, 15, 6, 17, 25, 3]
window_size = 5

print(window_minimums(numbers, window_size))