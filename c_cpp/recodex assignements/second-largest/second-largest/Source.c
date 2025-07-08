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
int find_second_largest()
{
	int first = read_int();
	int second = read_int();
	if (second > first)
	{
		int a = first;
		first = second;
		second = a;
	}
	int number = read_int();
	while (number != -1)
	{
		number = read_int();
		if (number > first)
		{
			second = first;
			first = number;
		}
		else if (number > second)
		{
			second = number;
		}
	}
	return second;
}
int main()
{
	printf("%d", find_second_largest());
	return 0;
}