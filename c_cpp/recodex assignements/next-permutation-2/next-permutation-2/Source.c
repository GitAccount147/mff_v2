#include <stdio.h>
int read_int();
void read_data(int length, int* data);
void next_permutation(int length, int* data);
int get_stop(int length, int* data, int* numbers_log);
void change_first(int stop, int* data, int* numbers_log);
void change_and_print(int length, int stop, int* data, int* numbers_log);
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
void read_data(int length, int* data)
{
	for (int i = 0; i < length; i++)
	{
		data[i] = read_int();
	}
}
void next_permutation(int length, int* data)
{
	int numbers_log[100];
	for (int i = 0; i < length; i++)
	{
		numbers_log[i] = 0;
	}
	int stop = get_stop(length, data, numbers_log);
	if (stop != -1)
	{
		numbers_log[data[stop] - 1] = 1;
		change_first(stop, data, numbers_log);
		change_and_print(length, stop, data, numbers_log);
	}
	else
	{
		printf("NEEXISTUJE");
	}
}
int get_stop(int length, int* data, int* numbers_log)
{
	int stop = 0;
	int i = length - 1;
	while (stop == 0)
	{
		numbers_log[data[i] - 1] = 1;
		if (i == 0)
		{
			stop = -1;
		}
		else
		{
			if (data[i] > data[i - 1])
			{
				stop = i - 1;
			}
		}
		i--;
	}
	return stop;
}
void change_first(int stop, int* data, int* numbers_log)
{
	int replacement = 0;
	int i = data[stop];
	while (replacement == 0)
	{
		if (numbers_log[i] == 1)
		{
			replacement = i + 1;
			numbers_log[i] = 0;
		}
		i++;
	}
	data[stop] = replacement;
}
void change_and_print(int length, int stop, int* data, int* numbers_log)
{
	for (int j = 0, k = stop + 1; j < length; j++)
	{
		if (numbers_log[j] == 1)
		{
			data[k] = j + 1;
			k++;
		}
	}
	for (int i = 0; i < length; i++)
	{
		if (i == 0)
		{
			printf("%d", data[i]);
		}
		else
		{
			printf(" %d", data[i]);
		}
	}
}
int main()
{
	int length = read_int();
	int data[100];
	read_data(length, data);
	next_permutation(length, data);
	return 0;
}