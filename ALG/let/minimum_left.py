"Ближайший меньший элемент слева"

def find_previous_less_indices(array):
    result = [-1] * len(array)
    stack = []

    for current_index, current_value in enumerate(array):
        while stack and array[stack[-1]] >= current_value:
            stack.pop()

        if stack:
            result[current_index] = stack[-1] + 1

        stack.append(current_index)

    return result


array = [14, 20, 21, 13, 5, 87, 9, 61, 3, 45, 10, 36]

print("Массив:", array)
print("Индексы ближайших меньших слева:", find_previous_less_indices(array))