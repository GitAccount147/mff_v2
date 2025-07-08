#include <stdio.h>
int read_int()
{
	int number = 0;
	int digit_char = getchar();
	int minus = 0;
	if (digit_char == 45)
	{
		minus = 1;
		digit_char = getchar();
	}
	while (digit_char >= 48 && digit_char <= 57)
	{
		number = (number * 10) + (digit_char - 48);
		digit_char = getchar();
	}
	if (minus == 1)
	{
		number = number * (-1);
	}
	return number;
}
int main() {
	int sequence[1000];
	int length = read_int();
	int max = 0;
	int first = 1;
	for (int i = 0; i < length; i++)
	{
		sequence[i] = read_int();
		if (sequence[i] > max)
		{
			max = sequence[i];
		}
	}
	printf("%d\n", max);
	for (int i = 0; i < length; i++)
	{
		if (sequence[i] == max)
		{
			if (first == 1)
			{
				printf("%d", i + 1);
				first = 0;
			}
			else
			{
				printf(" %d", i + 1);
			}
		}
	}
	return 0;
}