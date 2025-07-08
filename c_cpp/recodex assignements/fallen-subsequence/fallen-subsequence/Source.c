#include <stdio.h>
typedef struct sequences
{
	char used;
	int last_number;
} sequence;
int read_int()
{
	int number = 0;
	int minus = 0;
	char c = getchar();
	if (c == '-')
	{
		minus = 1;
		c = getchar();
	}
	while (!(c >= '0' && c <= '9') && c != '-')
	{
		c = getchar();
		if (c == '-')
		{
			minus = 1;
		}
	}
	while (c >= '0' && c <= '9')
	{
		number = 10 * number + c - '0';
		c = getchar();
	}
	if (minus)
	{
		return (-1) * number;
	}
	return number;
}
int main()
{
	int n = read_int();
	int current;

	sequence* without_fall[10000];
	sequence* with_fall[10000];
	for (int i = 0; i < n; i++)
	{
		without_fall[i] = malloc(sizeof(sequence));
		without_fall[i]->used = 'n';
		with_fall[i] = malloc(sizeof(sequence));
		with_fall[i]->used = 'n';
	}
	
	for (int i = 0; i < n; i++)
	{
		current = read_int();
		//printf("(%d)", current);
		for (int j = i - 1; j > -1; j--)
		{
			if (without_fall[j]->used == 'y')
			{
				if (current >= without_fall[j]->last_number)
				{
					if (without_fall[j + 1]->used == 'n' || current < without_fall[j + 1]->last_number)
					{
						without_fall[j + 1]->last_number = current;
						without_fall[j + 1]->used = 'y';
					}
				}
				if (current < without_fall[j]->last_number)
				{
					if (with_fall[j + 1]->used == 'n' || current < with_fall[j + 1]->last_number)
					{
						with_fall[j + 1]->last_number = current;
						with_fall[j + 1]->used = 'y';
					}
				}
			}
			if (with_fall[j]->used == 'y')
			{
				if (current >= with_fall[j]->last_number)
				{
					if (with_fall[j + 1]->used == 'n' || current < with_fall[j + 1]->last_number)
					{
						with_fall[j + 1]->last_number = current;
						with_fall[j + 1]->used = 'y';
					}
				}
			}
		}
		if (without_fall[0]->used == 'n' || current < without_fall[0]->last_number)
		{
			without_fall[0]->used = 'y';
			without_fall[0]->last_number = current;
		}
	}

	/*
	for (int i = 0; i < n; i++)
	{
		if (without_fall[i]->used == 'y')
		{
			printf("(%d)", without_fall[i]->last_number);
		}
		else
		{
			printf("(_)");
		}
	}
	printf("\n");
	for (int i = 0; i < n; i++)
	{
		if (with_fall[i]->used == 'y')
		{
			printf("(%d)", with_fall[i]->last_number);
		}
		else
		{
			printf("(_)");
		}
	}
	*/
	for (int i = n - 1; i > -1; i--)
	{
		if (with_fall[i]->used == 'y' || without_fall[i]->used == 'y')
		{
			printf("%d", i + 1);
			i = -1;
		}
	}
	return 0;
}