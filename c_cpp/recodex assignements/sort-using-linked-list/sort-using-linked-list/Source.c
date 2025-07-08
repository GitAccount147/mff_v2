#include <stdio.h>
typedef struct blocks
{
	int value;
	struct blocks* next;
} block;
block* add_mag(int value, block* head)
{
	block* new_block = malloc(sizeof(block));
	new_block->next = head;
	new_block->value = value;
	return new_block;
}
block* add_queue(int value, block* tail)
{
	block* new_block = malloc(sizeof(block));
	tail->next = new_block;
	new_block->value = value;
	new_block->next = NULL;
	return new_block;
}
void bubble_sort(block* head)
{
	int switched = 1;
	while (switched)
	{
		switched = 0;
		block* pom = head;
		while (pom->next != NULL)
		{
			if (pom->value > pom->next->value)
			{
				int temp = pom->next->value;
				pom->next->value = pom->value;
				pom->value = temp;
				switched = 1;
			}
			pom = pom->next;
		}
	}
}
int read_int_c(char c)
{
	int number = 0;
	int digit_char = c;
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
void read_data(block* tail)
{
	char c = getchar();
	while (c > 44)
	{
		int number = read_int_c(c);
		tail = add_queue(number, tail);
		c = getchar();
	}
}
void read_data2(block* tail)
{
	char c = getchar();
	int number = read_int_c(c);
	tail = add_queue(number, tail);
	c = getchar();
	while (c != EOF)
	{
		if (c > 44)
		{
			int number = read_int_c(c);
			tail = add_queue(number, tail);
		}
		c = getchar();
	}
}
int main()
{
	block* head = malloc(sizeof(block));
	head->next = NULL;
	head->value = -1;
	block* tail = malloc(sizeof(block));
	tail->next = NULL;
	tail->value = -1;

	char c = getchar();
	int first = read_int_c(c);
	tail = add_queue(first, tail);
	head = tail;

	read_data2(tail);
	bubble_sort(head);

	block* pom;
	for (int i = 0; i < 1; i++)
	{
		pom = head;
		while (pom != NULL)
		{
			printf("%d ", pom->value);
			pom = pom->next;
		}
	}
	return 0;
}