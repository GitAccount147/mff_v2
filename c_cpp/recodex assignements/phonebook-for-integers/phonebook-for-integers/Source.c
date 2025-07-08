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
		if (pom->next != NULL)
		{
			while (pom->next->next != NULL)
			{
				if (pom->value < pom->next->value)
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
}
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
block* delete_block(int number, block* head)
{
	while (head->value == number && head->next != NULL)
	{
		block* p2 = head;
		head = head->next;
		free(p2);
	}
	block* pom = head;
	if (pom->next != NULL)
	{
		while (pom->next->next != NULL)
		{
			if (pom->next->value == number)
			{
				block* pompom = pom->next;
				pom->next = pom->next->next;
				free(pompom);
			}
			else
			{
				pom = pom->next;
			}
		}
	}
	return head;
}
void print_blocks(block* head)
{
	block* pom = head;
	while (pom->next != NULL)
	{
		printf("%d\n", pom->value);
		pom = pom->next;
	}
}
void use_phonebook(block* head)
{
	int command = read_int();
	int number;
	while (command != 6)
	{
		switch (command)
		{
		case 1:
			number = read_int();
			head = add_mag(number, head);
			break;
		case 2:
			number = read_int();
			head = delete_block(number, head);
			break;
		case 4:
			bubble_sort(head);
			break;
		case 5:
			print_blocks(head);
			break;
		}
		command = read_int();
	}
}
int main()
{
	block* head = malloc(sizeof(block));
	head->next = NULL;
	head->value = -1;

	use_phonebook(head);
	return 0;
}