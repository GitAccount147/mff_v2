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
int main()
{
	int number_a = read_int();
	int number_b = read_int();
	if (number_b != 0)
	{
		printf("%d", number_a / number_b);
	}
	else
	{
		printf("NELZE");
	};
	return 0;
}