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
void properties(int number)
{
	int divisor = 1;
	int remainder;
	int perfect_sum = 0;
	int square = 0;
	int cube = 0;
	int quotient;
	while (divisor < number)
	{
		remainder = number % divisor;
		if (remainder == 0)
		{
			perfect_sum += divisor;
			quotient = number / divisor;
			if (quotient == divisor)
			{
				square = 1;
			}
			if (quotient == divisor * divisor)
			{
				cube = 1;
			}
		}
		divisor++;
	}
	if (perfect_sum == number)
	{
		printf("P");
	}
	if (square == 1)
	{
		printf("C");
	}
	if (cube == 1)
	{
		printf("K");
	}
}
int main()
{
	properties(read_int());
	return 0;
}