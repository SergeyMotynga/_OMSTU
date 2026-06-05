"Разбиение числа на сумму слагаемых"

def generate_partitions(number, minimum_part=1):
    if number == 0:
        yield []
        return

    for first_part in range(minimum_part, number + 1):
        for other_parts in generate_partitions(number - first_part, first_part):
            yield [first_part] + other_parts


def print_number_partitions(number):
    total_count = 0

    for partition in generate_partitions(number):
        total_count += 1
        expression = " + ".join(map(str, partition))
        print(f"{number} = {expression}")

    print()
    print(f"Всего разбиений: {total_count}")


print_number_partitions(10)