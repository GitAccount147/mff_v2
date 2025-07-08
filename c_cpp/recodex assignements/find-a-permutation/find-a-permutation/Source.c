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
	while (digit_char >= '0' && digit_char <= '9')
	{
		number = (number * 10) + (digit_char - '0');
		digit_char = getchar();
	}
	if (minus)
	{
		number = number * (-1);
	}
	return number;
}
void read_data(int length, int* data)
{
	for (int i = 0; i < length; i++)
	{
		data[i] = read_int();
	}
}
int get_factorial(int number)
{
	int result = 1;
	for (int i = 2; i <= number; i++)
	{
		result = result * i;
	}
	return result;
}
int factorial_to_decimal(int length, int* data)
{
	int number = 0;
	int used[13];
	for (int i = 0; i < 13; i++)
	{
		used[i] = 0;
	}
	for (int i = 0; i < length; i++)
	{
		used[data[i] - 1] = 1;
		int q = 0;
		for (int j = 0; j < data[i] - 1; j++)
		{
			q = q + used[j];
		}
		number = number + (data[i] - 1 - q) * get_factorial(length - i - 1);
	}
	return number;
}
void decimal_to_factorial(int number, int length, int* data)
{
	for (int i = length - 1; i > -1; i--)
	{
		int fact = get_factorial(i);
		int quotient = number / fact;
		number = number - (quotient * fact);
		data[length - i - 1] = quotient;
	}
	int used[13];
	for (int i = 0; i < 13; i++)
	{
		used[i] = 0;
	}
	for (int i = 0; i < length; i++)
	{
		int destination = data[i];
		int shift = 0;
		for (int j = 0; j < data[i]; j++)
		{
			shift = shift + used[j];
		}
		while (shift != 0)
		{
			while (used[destination] == 1)
			{
				destination++;
			}
			shift--;
			destination++;
		}
		while (used[destination] == 1)
		{
			destination++;
		}
		used[destination] = 1;
		data[i] = destination + 1;
	}
}
int main()
{
	int length = read_int();
	int shift = read_int();
	int data[13];
	int new_data[13];
	read_data(length, data);
	decimal_to_factorial(factorial_to_decimal(length, data) + shift, length, new_data);
	for (int i = 0; i < length; i++)
	{
		if (i == 0)
		{
			printf("%d", new_data[0]);
		}
		else
		{
			printf(" %d", new_data[i]);
		}
	}
	return 0;
}