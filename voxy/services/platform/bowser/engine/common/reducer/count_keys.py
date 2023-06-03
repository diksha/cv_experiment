from typing import Tuple

from pyflink.datastream import AggregateFunction


class CountKeys(AggregateFunction):
    def create_accumulator(self) -> Tuple[Tuple, int]:
        """Create a new accumulator to count unique key

        :returns: Tuple[Tuple, int]
        - Tuple = key
        - int = counter

        """
        return (), 0

    def add(
        self, value: Tuple, accumulator: Tuple[Tuple, int]
    ) -> Tuple[Tuple, int]:
        """Increment the counter inside the Tuple if the key is the same

        :param Tuple value: unique key
        :param Tuple accumulator: Tuple[Tuple( Key) , int ( Counter) ]
        :returns: the new incremented unique key
        :rtype: Tuple[Tuple, int]

        """
        return value, accumulator[1] + 1

    def get_result(self, accumulator: Tuple[Tuple, int]) -> Tuple[Tuple, int]:
        """Display the result of the accumulator

        :param Tuple accumulator: Tuple[Tuple ( Key ) , int ( Counter )]
        :returns: Tuple[Tuple ( Key ) , int ( Counter )]
        :rtype: Tuple[Tuple, int]

        """
        return accumulator

    def merge(
        self,
        accumulator_a: Tuple[Tuple, int],
        accumulator_b: Tuple[Tuple, int],
    ) -> Tuple[Tuple, int]:
        """


        :param Tuple accumulator_a: Tuple[Tuple ( Key ) , int ( Counter )]
        :param Tuple accumulator_b: Tuple[Tuple ( Key ) , int ( Counter )]
        :returns: a Tuple[Tuple ( Key ) , int ( Counter )]
        :rtype: Tuple

        """
        return (
            accumulator_a[0] + accumulator_b[0],
            accumulator_a[1] + accumulator_b[1],
        )
